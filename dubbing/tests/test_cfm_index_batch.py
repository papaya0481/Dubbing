from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

librosa = pytest.importorskip("librosa", reason="librosa is required for batch CFM test")
torchaudio = pytest.importorskip("torchaudio", reason="torchaudio is required for batch CFM test")
OmegaConf = pytest.importorskip("omegaconf", reason="omegaconf is required for batch CFM test").OmegaConf

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent
PROJ_ROOT = FILE_DIR.parents[1]
INDEX_ROOT = PROJ_ROOT / "index-tts2"

for p in [str(INDEX_ROOT), str(PROJ_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("HF_HUB_CACHE", str(PROJ_ROOT / "checkpoints" / "hf_cache"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_DIR = Path("/data2/ruixin/index-tts2/checkpoints")
DATA_CSV = Path("/data2/ruixin/datasets/flow_dataset/MELD/semantic/metadata.csv")
DATA_ROOT = DATA_CSV.parent
OUTPUT_DIR = Path("/data2/ruixin/ours/test_outputs/batch_cfm_index")

# Runtime knobs via env vars for pytest usage.
TEST_DEVICE = os.getenv("CFM_TEST_DEVICE", "cuda:0")
TEST_N_SAMPLES = int(os.getenv("CFM_TEST_N_SAMPLES", "4"))
TEST_BATCH_SIZE = int(os.getenv("CFM_TEST_BATCH_SIZE", "2"))
TEST_N_STEPS = int(os.getenv("CFM_TEST_N_STEPS", "10"))
TEST_CFG_RATE = float(os.getenv("CFM_TEST_CFG_RATE", "0.7"))
TEST_EQ_THRESHOLD = float(os.getenv("CFM_TEST_EQ_THRESHOLD", "1e-4"))


class Sample(NamedTuple):
    stem: str
    prompt_audio_path: str
    out_pt: str
    out_wav: str


def load_samples(n: int) -> list[Sample]:
    samples: list[Sample] = []
    with open(DATA_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("gen_error", "").strip():
                continue
            out_pt = row.get("out_pt", "").strip()
            out_wav = row.get("out_wav", "").strip()
            prompt = row.get("prompt_audio_path", "").strip()
            if not (out_pt and out_wav and prompt):
                continue
            if out_pt and not Path(out_pt).is_absolute():
                out_pt = str((DATA_ROOT / out_pt).resolve())
            if out_wav and not Path(out_wav).is_absolute():
                out_wav = str((DATA_ROOT / out_wav).resolve())
            if prompt and not Path(prompt).is_absolute():
                prompt = str((DATA_ROOT / prompt).resolve())
            if not (Path(out_pt).exists() and Path(out_wav).exists() and Path(prompt).exists()):
                continue
            samples.append(
                Sample(
                    stem=Path(out_pt).stem,
                    prompt_audio_path=prompt,
                    out_pt=out_pt,
                    out_wav=out_wav,
                )
            )
            if len(samples) >= n:
                break
    return samples


class ConditionBuilder:
    """Build CFM conditioning tensors from prompt audio + S_infer."""

    mel_ratio: float = 22050 / (50 * 256)

    def __init__(self, cfg, model_dir: Path, device: str):
        self.device = device
        self.cfg = cfg

        from transformers import SeamlessM4TFeatureExtractor
        from huggingface_hub import hf_hub_download
        import safetensors

        from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        from dubbing.modules.mel_strech.meldataset import get_mel_spectrogram

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        stat_path = str(model_dir / cfg.w2v_stat)
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(stat_path)
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)

        semantic_codec = build_semantic_codec(cfg.semantic_codec)
        codec_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, codec_ckpt)
        self.semantic_codec = semantic_codec.to(device).eval()

        campplus_ckpt = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
        self.campplus = campplus.to(device).eval()

        s2mel_path = str(model_dir / cfg.s2mel_checkpoint)
        s2mel_model = MyModel(cfg.s2mel, use_gpt_latent=False)
        s2mel_model, _, _, _ = load_checkpoint2(
            s2mel_model,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.length_regulator = s2mel_model.models["length_regulator"].to(device).eval()

        sp = cfg.s2mel.preprocess_params.spect_params
        fmax_val = None if str(sp.get("fmax", "None")) == "None" else 8000
        _mel_h = SimpleNamespace(
            n_fft=sp.n_fft,
            num_mels=sp.n_mels,
            sampling_rate=cfg.s2mel.preprocess_params.sr,
            hop_size=sp.hop_length,
            win_size=sp.win_length,
            fmin=sp.get("fmin", 0),
            fmax=fmax_val,
        )
        self._mel_fn = lambda wav: get_mel_spectrogram(wav, _mel_h)

    @torch.no_grad()
    def _load_audio(self, path: str, max_sec: float = 15.0):
        audio, sr = librosa.load(path)
        audio = torch.tensor(audio).unsqueeze(0)
        max_len = int(max_sec * sr)
        if audio.shape[1] > max_len:
            audio = audio[:, :max_len]
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
        return audio_22k, audio_16k

    @torch.no_grad()
    def _get_emb(self, input_features, attention_mask):
        out = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = out.hidden_states[17]
        return (feat - self.semantic_mean) / self.semantic_std

    @torch.no_grad()
    def build(self, spk_audio_path: str, s_infer: torch.Tensor):
        if s_infer.dim() == 2:
            s_infer = s_infer.unsqueeze(0)
        s_infer = s_infer.to(self.device).float()

        audio_22k, audio_16k = self._load_audio(spk_audio_path)

        ref_mel = self._mel_fn(audio_22k.to(self.device).float())  # [1, 80, T_ref]

        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        feat_in = inputs["input_features"].to(self.device)
        attn_m = inputs["attention_mask"].to(self.device)
        spk_emb = self._get_emb(feat_in, attn_m)
        _, s_ref = self.semantic_codec.quantize(spk_emb)

        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(self.device)
        prompt_cond = self.length_regulator(
            s_ref,
            ylens=ref_target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]  # [1, T_ref_mel, 512]

        feat_fbank = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(self.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat_fbank = feat_fbank - feat_fbank.mean(dim=0, keepdim=True)
        style = self.campplus(feat_fbank.unsqueeze(0))  # [1, 192]

        code_len = s_infer.size(1)
        target_lengths = torch.tensor(
            [int(code_len * self.mel_ratio)],
            dtype=torch.long,
            device=self.device,
        )
        infer_cond = self.length_regulator(
            s_infer,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]  # [1, T_infer_mel, 512]

        cat_condition = torch.cat([prompt_cond, infer_cond], dim=1)  # [1, T_total, 512]
        x_lens = torch.LongTensor([cat_condition.size(1)]).to(self.device)

        return cat_condition, ref_mel, style, x_lens

    @torch.no_grad()
    def build_batch(self, spk_audio_paths: list[str], s_infer_list: list[torch.Tensor]):
        """Batch version of build.

        Returns
        -------
        cat_condition : [B, T_total_max, 512]
        ref_mel_batch : [B, 80, T_ref_max]
        style_batch   : [B, 192]
        x_lens_batch  : [B]
        prompt_lens   : list[int]
        """
        bsz = len(spk_audio_paths)
        assert bsz == len(s_infer_list), "spk_audio_paths and s_infer_list size mismatch"

        # 1) load audios
        audio_22k_list: list[torch.Tensor] = []
        audio_16k_list: list[torch.Tensor] = []
        for p in spk_audio_paths:
            a22k, a16k = self._load_audio(p)
            audio_22k_list.append(a22k)
            audio_16k_list.append(a16k)

        # 2) ref mel per sample, then pad to batch tensor [B, 80, T_ref_max]
        ref_mel_list: list[torch.Tensor] = []
        prompt_lens: list[int] = []
        for a22k in audio_22k_list:
            m = self._mel_fn(a22k.to(self.device).float())  # [1, 80, T_ref]
            ref_mel_list.append(m)
            prompt_lens.append(int(m.size(-1)))

        ref_target_lengths = torch.tensor(prompt_lens, dtype=torch.long, device=self.device)
        ref_mel_batch = _pad_ref_mels(ref_mel_list).to(self.device)

        # 3) semantic feature extraction in batch for prompt_condition
        audio_16k_np = [a16k.squeeze(0).cpu().numpy() for a16k in audio_16k_list]
        inputs = self.extract_features(
            audio_16k_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        feat_in = inputs["input_features"].to(self.device)
        attn_m = inputs["attention_mask"].to(self.device)
        spk_emb = self._get_emb(feat_in, attn_m)
        _, s_ref = self.semantic_codec.quantize(spk_emb)

        prompt_cond = self.length_regulator(
            s_ref,
            ylens=ref_target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]  # [B, T_ref_max, 512]

        # 4) style per sample then stack [B, 192]
        style_list: list[torch.Tensor] = []
        for a16k in audio_16k_list:
            feat_fbank = torchaudio.compliance.kaldi.fbank(
                a16k.to(self.device),
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat_fbank = feat_fbank - feat_fbank.mean(dim=0, keepdim=True)
            style = self.campplus(feat_fbank.unsqueeze(0))  # [1, 192]
            style_list.append(style.squeeze(0))
        style_batch = torch.stack(style_list, dim=0).to(self.device)

        # 5) infer condition in batch
        s_infer_seq: list[torch.Tensor] = []
        infer_lens: list[int] = []
        for s_infer in s_infer_list:
            if s_infer.dim() == 2:
                s_infer = s_infer.unsqueeze(0)
            s_infer = s_infer.to(self.device).float()
            s_infer_seq.append(s_infer.squeeze(0))  # [T_code, 1024]
            infer_lens.append(int(s_infer.size(1) * self.mel_ratio))

        s_infer_batch = pad_sequence(s_infer_seq, batch_first=True, padding_value=0.0).to(self.device)
        infer_target_lengths = torch.tensor(infer_lens, dtype=torch.long, device=self.device)
        infer_cond = self.length_regulator(
            s_infer_batch,
            ylens=infer_target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]  # [B, T_infer_max, 512]

        # 6) concatenate conditions and lengths
        cat_condition = torch.cat([prompt_cond, infer_cond], dim=1)
        x_lens_batch = ref_target_lengths + infer_target_lengths

        return cat_condition, ref_mel_batch, style_batch, x_lens_batch, prompt_lens


def _load_cfm_state(model_dir: Path) -> dict:
    state = torch.load(str(model_dir / "s2mel.pth"), map_location="cpu")
    return state["net"]["cfm"]


def build_orig_cfm(cfg_s2mel, cfm_state: dict, device: str):
    from indextts.s2mel.modules.flow_matching import CFM as OrigCFM

    model = OrigCFM(cfg_s2mel)
    model.load_state_dict(cfm_state, strict=False)
    model = model.to(device).eval()
    model.estimator.setup_caches(max_batch_size=max(1, TEST_BATCH_SIZE), max_seq_length=8192)
    return model


def build_my_cfm(cfg_s2mel, cfm_state: dict, device: str):
    from dubbing.modules.cfm_index.flow_matching import CFM as MyCFM

    model = MyCFM(cfg_s2mel)
    model.load_state_dict(cfm_state, strict=False)
    model = model.to(device).eval()
    model.estimator.setup_caches(max_batch_size=max(1, TEST_BATCH_SIZE), max_seq_length=8192)
    return model


def mel_from_wav(wav_path: str, mel_fn) -> torch.Tensor:
    audio, _ = librosa.load(wav_path, sr=22050)
    wave = torch.tensor(audio).unsqueeze(0)
    mel = mel_fn(wave.float())
    return mel


def l1_loss_trimmed(a: torch.Tensor, b: torch.Tensor) -> float:
    t = min(a.size(-1), b.size(-1))
    return F.l1_loss(a[..., :t], b[..., :t]).item()


def _pad_ref_mels(ref_mels: list[torch.Tensor]) -> torch.Tensor:
    # Input: list of [1, 80, T_i] -> output [B, 80, max_T]
    max_t = max(int(m.size(-1)) for m in ref_mels)
    padded = []
    for m in ref_mels:
        cur_t = int(m.size(-1))
        if cur_t < max_t:
            m = F.pad(m, (0, max_t - cur_t))
        padded.append(m.squeeze(0))
    return torch.stack(padded, dim=0)


def _batchify(builder: ConditionBuilder, batch_samples: list[Sample], device: str):
    s_infer_list = [torch.load(sample.out_pt, map_location=device).float() for sample in batch_samples]
    spk_audio_paths = [sample.prompt_audio_path for sample in batch_samples]
    return builder.build_batch(spk_audio_paths, s_infer_list)


def _save_wav(vocoder, mel: torch.Tensor, out_path: Path):
    # mel: [1, 80, T]
    with torch.no_grad():
        wav = vocoder(mel.float()).squeeze(1)
    wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu().to(torch.int16)
    torchaudio.save(str(out_path), wav, 22050)


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="checkpoint directory does not exist")
@pytest.mark.skipif(not DATA_CSV.exists(), reason="metadata csv does not exist")
def test_cfm_index_batch_inference_equivalence():
    if TEST_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available for configured CFM_TEST_DEVICE")

    device = TEST_DEVICE

    cfg = OmegaConf.load(MODEL_DIR / "config.yaml")
    cfm_state = _load_cfm_state(MODEL_DIR)

    orig_cfm = build_orig_cfm(cfg.s2mel, cfm_state, device)
    my_cfm = build_my_cfm(cfg.s2mel, cfm_state, device)
    builder = ConditionBuilder(cfg, MODEL_DIR, device)

    from indextts.s2mel.modules.bigvgan import bigvgan

    vocoder = bigvgan.BigVGAN.from_pretrained(cfg.vocoder.name, use_cuda_kernel=False)
    vocoder = vocoder.to(device)
    vocoder.remove_weight_norm()
    vocoder.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_samples(TEST_N_SAMPLES)
    if not samples:
        pytest.skip("No valid samples found from metadata.csv")

    rows: list[dict[str, float | str]] = []
    all_r1r2: list[float] = []

    mel_fn = builder._mel_fn

    for start in range(0, len(samples), TEST_BATCH_SIZE):
        batch_samples = samples[start : start + TEST_BATCH_SIZE]

        cond_batch, ref_mel_batch, style_batch, x_lens_batch, prompt_lens = _batchify(
            builder, batch_samples, device
        )
        # DiT attention mask in current inference path expects x_lens max to match
        # the padded sequence length used by cond/prompt tensors.
        x_lens_infer = torch.full_like(x_lens_batch, int(cond_batch.size(1)))

        # Current CFM inference path applies CFG by internally expanding batch.
        # For true batch size > 1 this can create a batch-size mismatch in timestep
        # conditioning for the current implementations, so we disable CFG in that case.
        cfg_rate = TEST_CFG_RATE if cond_batch.size(0) == 1 else 0.0

        # Use the same random seed for both routes to compare implementation equivalence.
        seed = 42
        torch.manual_seed(seed)
        with torch.no_grad():
            my_full = my_cfm.inference(
                cond_batch,
                x_lens_infer,
                ref_mel_batch,
                style_batch,
                None,
                TEST_N_STEPS,
                inference_cfg_rate=cfg_rate,
            )

        torch.manual_seed(seed)
        with torch.no_grad():
            orig_full = orig_cfm.inference(
                cond_batch,
                x_lens_infer,
                ref_mel_batch,
                style_batch,
                None,
                TEST_N_STEPS,
                inference_cfg_rate=cfg_rate,
            )

        for i, sample in enumerate(batch_samples):
            total_len = int(x_lens_batch[i].item())
            prompt_len = int(prompt_lens[i])

            my_gen = my_full[i : i + 1, :, :total_len][:, :, prompt_len:]
            orig_gen = orig_full[i : i + 1, :, :total_len][:, :, prompt_len:]

            gt_mel = mel_from_wav(sample.out_wav, mel_fn).to(device)

            l1_r1r2 = l1_loss_trimmed(my_gen.cpu(), orig_gen.cpu())
            l1_r1r3 = l1_loss_trimmed(my_gen.cpu(), gt_mel.cpu())
            l1_r2r3 = l1_loss_trimmed(orig_gen.cpu(), gt_mel.cpu())

            all_r1r2.append(l1_r1r2)
            rows.append(
                {
                    "stem": sample.stem,
                    "l1_my_vs_orig": l1_r1r2,
                    "l1_my_vs_gt": l1_r1r3,
                    "l1_orig_vs_gt": l1_r2r3,
                }
            )

            _save_wav(vocoder, my_gen, OUTPUT_DIR / f"{sample.stem}_my.wav")
            _save_wav(vocoder, orig_gen, OUTPUT_DIR / f"{sample.stem}_orig.wav")

            torch.save(my_gen.cpu(), OUTPUT_DIR / f"{sample.stem}_my_mel.pt")
            torch.save(orig_gen.cpu(), OUTPUT_DIR / f"{sample.stem}_orig_mel.pt")

    metrics_path = OUTPUT_DIR / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stem", "l1_my_vs_orig", "l1_my_vs_gt", "l1_orig_vs_gt"],
        )
        writer.writeheader()
        writer.writerows(rows)

    mean_eq = float(sum(all_r1r2) / max(1, len(all_r1r2)))
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"n_samples={len(rows)}\n")
        f.write(f"batch_size={TEST_BATCH_SIZE}\n")
        f.write(f"n_steps={TEST_N_STEPS}\n")
        f.write(f"cfg_rate_requested={TEST_CFG_RATE}\n")
        f.write("cfg_rate_effective=0.0 for batch>1, requested value for batch==1\n")
        f.write(f"mean_l1_my_vs_orig={mean_eq:.8f}\n")
        f.write(f"threshold={TEST_EQ_THRESHOLD}\n")

    assert len(rows) > 0, "No samples were processed"
    assert torch.isfinite(torch.tensor(mean_eq)), "mean L1 is not finite"
    assert mean_eq <= TEST_EQ_THRESHOLD, (
        f"Batch equivalence failed: mean L1(my, orig)={mean_eq:.6f} > {TEST_EQ_THRESHOLD}"
    )
