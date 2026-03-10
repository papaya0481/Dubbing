"""
测试：比较 dubbing/modules/cfm_index/CFM（自写移植版）
      与 IndexTTS2 原版 CFM 的输出一致性。

运行方式（从项目根目录）：
    conda run -n dub3 python -m dubbing.tests.test_cfm_index \
        [--device cuda:0] [--n-samples 3] [--n-steps 10]

测试逻辑
--------
对 metadata.csv 中若干样本：
  1. 从 out_pt 加载 S_infer         （GPT 语义 + gpt_layer 已融合的 1024-dim 特征序列）
  2. 从 prompt_audio_path 加载参考音频，提取：
       ref_mel       ── mel 频谱（作为 CFM 的 prompt 输入）
       style         ── CAMPPlus 全局 speaker embedding
       prompt_cond   ── length_regulator(S_ref) 上采样后的条件
  3. 对 S_infer 做 length_regulator 上采样得到 infer_cond
  4. cat_condition = cat([prompt_cond, infer_cond], dim=1)

  result1 ── 自写 CFM (dubbing/modules/cfm_index) 用 s2mel.pth 权重推理
  result2 ── 原版 CFM (index-tts2/indextts)        用 s2mel.pth 权重推理
  result3 ── out_wav 的 mel 频谱（IndexTTS2 全流程生成结果，作为参考）

  比较 L1 损失（只作参考，不训练）：
    loss(result1, result2) ── 应趋近 0（实现等价性验证）
    loss(result1, result3) ── 参考质量
    loss(result2, result3) ── 原版与生成结果的参考质量
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
FILE_DIR   = Path(__file__).resolve().parent          # dubbing/tests/
PROJ_ROOT  = FILE_DIR.parents[1]                      # /home/ruixin/Dubbing
INDEX_ROOT = PROJ_ROOT / "index-tts2"                 # /home/ruixin/Dubbing/index-tts2

# index-tts2 放最前面，保证 from indextts.* 导入原版模块
for p in [str(INDEX_ROOT), str(PROJ_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("HF_HUB_CACHE", str(PROJ_ROOT / "checkpoints" / "hf_cache"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
import torchaudio
import librosa
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
MODEL_DIR = Path("/data2/ruixin/index-tts2/checkpoints")
DATA_CSV  = Path("/data2/ruixin/datasets/flow_dataset/MELD_semantic/metadata.csv")
DATA_ROOT = DATA_CSV.parent           # 用于解析相对路径


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CFM 实现等价性测试")
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--n-samples", type=int, default=3,
                   help="从 CSV 中抽取的样本数量（跳过 gen_error 不空的行）")
    p.add_argument("--n-steps",   type=int, default=10,
                   help="CFM 扩散步数（减小可加快测试，10 已足够验证等价性）")
    p.add_argument("--cfg-rate",  type=float, default=0.7, help="Classifier-free guidance 强度")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 样本加载
# ---------------------------------------------------------------------------
class Sample(NamedTuple):
    stem:              str
    prompt_audio_path: str
    out_pt:            str
    out_wav:           str


def load_samples(n: int = 5) -> list[Sample]:
    """从 metadata.csv 加载 n 条有效样本（out_pt 和 out_wav 均存在）。"""
    samples: list[Sample] = []
    with open(DATA_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("gen_error", "").strip():
                continue
            out_pt  = row.get("out_pt", "").strip()
            out_wav = row.get("out_wav", "").strip()
            prompt  = row.get("prompt_audio_path", "").strip()
            if not (out_pt and out_wav and prompt):
                continue
            # 相对路径 → 绝对路径
            if out_pt and not Path(out_pt).is_absolute():
                out_pt = str((DATA_ROOT / out_pt).resolve())
            if out_wav and not Path(out_wav).is_absolute():
                out_wav = str((DATA_ROOT / out_wav).resolve())
            if not Path(out_pt).exists() or not Path(out_wav).exists():
                continue
            stem = Path(out_pt).stem
            samples.append(Sample(stem=stem, prompt_audio_path=prompt,
                                  out_pt=out_pt, out_wav=out_wav))
            if len(samples) >= n:
                break
    return samples


# ---------------------------------------------------------------------------
# 条件构建器：加载冻结子模型，为 CFM 准备输入条件
# ---------------------------------------------------------------------------
class ConditionBuilder:
    """
    加载 s2mel 推理所需的冻结模型（跳过 GPT），并从音频 + S_infer 构建 CFM 条件。

    加载的模型：
      - SeamlessM4TFeatureExtractor + w2v-bert-2.0 (semantic_model)
      - semantic_codec (RepCodec) – 仅用于量化参考音频特征
      - CAMPPlus (speaker style encoder)
      - InterpolateRegulator (length_regulator，来自 s2mel.pth)
      - mel_spectrogram 函数
    """

    mel_ratio: float = 22050 / (50 * 256)   # ≈ 1.7227，与 infer_v2 一致

    def __init__(self, cfg, model_dir: Path, device: str):
        self.device = device
        self.cfg    = cfg

        # ---- SeamlessM4TFeatureExtractor --------------------------------
        from transformers import SeamlessM4TFeatureExtractor
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        # ---- w2v-bert semantic model ------------------------------------
        from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
        stat_path = str(model_dir / cfg.w2v_stat)
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(stat_path)
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean  = self.semantic_mean.to(device)
        self.semantic_std   = self.semantic_std.to(device)

        # ---- semantic_codec (RepCodec) ----------------------------------
        from huggingface_hub import hf_hub_download
        import safetensors
        semantic_codec = build_semantic_codec(cfg.semantic_codec)
        codec_ckpt = hf_hub_download("amphion/MaskGCT",
                                     filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, codec_ckpt)
        self.semantic_codec = semantic_codec.to(device).eval()

        # ---- CAMPPlus ---------------------------------------------------
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        campplus_ckpt = hf_hub_download("funasr/campplus",
                                        filename="campplus_cn_common.bin")
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
        self.campplus = campplus.to(device).eval()

        # ---- length_regulator（从 s2mel.pth 加载）-----------------------
        #  使用原版 MyModel 仅加载 length_regulator 子模块
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        s2mel_path = str(model_dir / cfg.s2mel_checkpoint)
        s2mel_model = MyModel(cfg.s2mel, use_gpt_latent=False)
        s2mel_model, _, _, _ = load_checkpoint2(
            s2mel_model, None, s2mel_path,
            load_only_params=True, ignore_modules=[], is_distributed=False,
        )
        self.length_regulator = s2mel_model.models["length_regulator"].to(device).eval()

        # ---- mel_spectrogram 函数 ----------------------------------------
        from indextts.s2mel.modules.audio import mel_spectrogram
        sp = cfg.s2mel.preprocess_params.spect_params
        fmax_val = None if str(sp.get("fmax", "None")) == "None" else 8000
        self._mel_fn = lambda wav: mel_spectrogram(
            wav,
            n_fft=sp.n_fft,
            num_mels=sp.n_mels,
            sampling_rate=cfg.s2mel.preprocess_params.sr,
            hop_size=sp.hop_length,
            win_size=sp.win_length,
            fmin=sp.get("fmin", 0),
            fmax=fmax_val,
            center=False,
        )

        print("[ConditionBuilder] 全部冻结子模型加载完成。")

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _load_audio(self, path: str, max_sec: float = 15.0):
        audio, sr = librosa.load(path)
        audio = torch.tensor(audio).unsqueeze(0)           # [1, T]
        max_len = int(max_sec * sr)
        if audio.shape[1] > max_len:
            audio = audio[:, :max_len]
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
        return audio_22k, audio_16k

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        out = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = out.hidden_states[17]                       # [B, T, 1024]
        return (feat - self.semantic_mean) / self.semantic_std

    @torch.no_grad()
    def build(
        self,
        spk_audio_path: str,
        s_infer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对一条样本构建 CFM 的全部输入条件。

        Parameters
        ----------
        spk_audio_path : str
            说话人参考音频路径。
        s_infer : torch.Tensor
            从 .pt 文件加载的 S_infer 特征，形状 [1, T_codes, 1024]。

        Returns
        -------
        cat_condition : Tensor [1, T_ref_mel + T_infer_mel, 512]
        ref_mel       : Tensor [1, 80, T_ref_mel]
        style         : Tensor [1, 192]
        x_lens        : LongTensor [1]  总 mel 帧数（= cat_condition.size(1)）
        """
        device = self.device
        if s_infer.dim() == 2:
            s_infer = s_infer.unsqueeze(0)           # [1, T, 1024]
        s_infer = s_infer.to(device).float()

        # -------- 处理参考音频 ------------------------------------------
        audio_22k, audio_16k = self._load_audio(spk_audio_path)

        # mel（作为 CFM prompt）
        ref_mel = self._mel_fn(audio_22k.to(device).float())   # [1, 80, T_ref]

        # w2v-bert 特征 → 量化得 S_ref（用于 prompt_condition）
        inputs = self.extract_features(audio_16k, sampling_rate=16000,
                                       return_tensors="pt")
        feat_in = inputs["input_features"].to(device)
        attn_m  = inputs["attention_mask"].to(device)
        spk_emb = self.get_emb(feat_in, attn_m)         # [1, T_feat, 1024]
        _, S_ref = self.semantic_codec.quantize(spk_emb) # S_ref: discrete codes or embeddings

        # prompt_condition
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(device)
        prompt_cond = self.length_regulator(
            S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
        )[0]                                             # [1, T_ref_mel, 512]

        # CAMPPlus style embedding
        feat_fbank = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(device), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat_fbank = feat_fbank - feat_fbank.mean(dim=0, keepdim=True)
        style = self.campplus(feat_fbank.unsqueeze(0))   # [1, 192]

        # -------- 处理 S_infer -----------------------------------------
        # mel_ratio = 1/50 * sr/hop = 22050 / (50*256) ≈ 1.7227
        code_len = s_infer.size(1)
        target_lengths = torch.tensor(
            [int(code_len * self.mel_ratio)], dtype=torch.long, device=device
        )
        infer_cond = self.length_regulator(
            s_infer, ylens=target_lengths, n_quantizers=3, f0=None
        )[0]                                             # [1, T_infer_mel, 512]

        # -------- 拼接条件 ---------------------------------------------
        cat_condition = torch.cat([prompt_cond, infer_cond], dim=1)   # [1, T_total, 512]
        x_lens = torch.LongTensor([cat_condition.size(1)]).to(device)

        return cat_condition, ref_mel, style, x_lens


# ---------------------------------------------------------------------------
# CFM 构建与权重加载
# ---------------------------------------------------------------------------

def _load_cfm_state(model_dir: Path) -> dict:
    """从 s2mel.pth 中提取 CFM 子模块的 state_dict。"""
    state = torch.load(str(model_dir / "s2mel.pth"), map_location="cpu")
    return state["net"]["cfm"]


def build_orig_cfm(cfg_s2mel, cfm_state: dict, device: str):
    """加载原版 IndexTTS2 CFM（index-tts2/indextts）。"""
    from indextts.s2mel.modules.flow_matching import CFM as OrigCFM

    model = OrigCFM(cfg_s2mel)
    missing, unexpected = model.load_state_dict(cfm_state, strict=False)
    if missing:
        print(f"  [orig_cfm] 缺少 key: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [orig_cfm] 多余 key: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model = model.to(device).eval()
    model.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    return model


def build_my_cfm(cfg_s2mel, cfm_state: dict, device: str):
    """加载自写 CFM（dubbing/modules/cfm_index）。"""
    from dubbing.modules.cfm_index.flow_matching import CFM as MyCFM

    model = MyCFM(cfg_s2mel)
    missing, unexpected = model.load_state_dict(cfm_state, strict=False)
    if missing:
        print(f"  [my_cfm]   缺少 key: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [my_cfm]   多余 key: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model = model.to(device).eval()
    model.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    return model


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def mel_from_wav(wav_path: str, mel_fn) -> torch.Tensor:
    """加载 wav 文件并计算其 mel 频谱，返回 [1, 80, T]。"""
    audio, sr = librosa.load(wav_path, sr=22050)
    wave = torch.tensor(audio).unsqueeze(0)      # [1, T]
    mel  = mel_fn(wave.float())                  # [1, 80, T_mel]
    return mel


def l1_loss_trimmed(a: torch.Tensor, b: torch.Tensor) -> float:
    """对两个形状可能不同的 mel 频谱，取最短长度后计算 L1 损失。
    输入均为 [1, 80, T]，输出标量 float。
    """
    t = min(a.size(-1), b.size(-1))
    return F.l1_loss(a[..., :t], b[..., :t]).item()


# ---------------------------------------------------------------------------
# 主测试逻辑
# ---------------------------------------------------------------------------

def run_test(args: argparse.Namespace) -> None:
    device = args.device

    # ---- 加载配置 & 权重 -----------------------------------------------
    cfg = OmegaConf.load(MODEL_DIR / "config.yaml")
    print(f"\n[Config] 已加载：{MODEL_DIR / 'config.yaml'}")

    print("\n[Loading] 提取 s2mel.pth 中的 CFM state_dict …")
    cfm_state = _load_cfm_state(MODEL_DIR)
    print(f"  CFM keys 数量：{len(cfm_state)}")

    # ---- 构建两个 CFM --------------------------------------------------
    print("\n[Build] 加载原版 IndexTTS2 CFM …")
    orig_cfm = build_orig_cfm(cfg.s2mel, cfm_state, device)
    print("  [orig_cfm] 加载完成。")

    print("\n[Build] 加载自写 CFM …")
    my_cfm = build_my_cfm(cfg.s2mel, cfm_state, device)
    print("  [my_cfm] 加载完成。")

    # ---- 加载条件构建器 ------------------------------------------------
    print("\n[Build] 初始化 ConditionBuilder（加载冻结子模型）…")
    builder = ConditionBuilder(cfg, MODEL_DIR, device)

    # ---- 加载 BigVGAN（用于将 result1 转换回音频）--------------------
    print("\n[Build] 加载 BigVGAN vocoder …")
    from indextts.s2mel.modules.bigvgan import bigvgan
    bigvgan_name = cfg.vocoder.name
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
    bigvgan_model = bigvgan_model.to(device)
    bigvgan_model.remove_weight_norm()
    bigvgan_model.eval()
    print(f"  BigVGAN 加载完成：{bigvgan_name}")

    output_dir = PROJ_ROOT / "test_output"
    output_dir.mkdir(exist_ok=True)

    # mel_fn 直接从 builder 复用
    mel_fn  = builder._mel_fn

    # ---- 加载样本 -------------------------------------------------------
    samples = load_samples(args.n_samples)
    if not samples:
        print("[ERROR] 未加载到任何有效样本，请检查 DATA_CSV 路径。")
        return
    print(f"\n[Data] 加载 {len(samples)} 条样本。")

    # ---- 逐样本推理 & 比较 ---------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'stem':<30} {'L1(r1,r2)':>12} {'L1(r1,r3)':>12} {'L1(r2,r3)':>12}")
    print("=" * 70)

    all_r1r2, all_r1r3, all_r2r3 = [], [], []
    first_sample_done = False

    for sample in samples:
        # 加载 S_infer
        s_infer = torch.load(sample.out_pt, map_location=device)  # [1, T, 1024]
        if s_infer.dim() == 2:
            s_infer = s_infer.unsqueeze(0)
        s_infer = s_infer.float()

        # 构建条件
        try:
            cat_condition, ref_mel, style, x_lens = builder.build(
                sample.prompt_audio_path, s_infer
            )
        except Exception as e:
            print(f"  [{sample.stem}] 条件构建失败：{e}")
            continue

        ref_prompt_len = ref_mel.size(-1)   # prompt（参考音频）mel 帧数

        # 固定随机种子：两个 CFM 必须从相同的噪声 z 出发，否则随机性会掩盖实现差异
        SEED = 42

        # ---- result1：自写 CFM ----------------------------------------
        torch.manual_seed(SEED)
        with torch.no_grad():
            r1_full = my_cfm.inference(
                cat_condition, x_lens, ref_mel, style, None,
                args.n_steps, inference_cfg_rate=args.cfg_rate
            )
        result1 = r1_full[:, :, ref_prompt_len:]         # 去掉 prompt 部分 [1, 80, T_gen]

        # ---- result2：原版 CFM ----------------------------------------
        torch.manual_seed(SEED)
        with torch.no_grad():
            r2_full = orig_cfm.inference(
                cat_condition, x_lens, ref_mel, style, None,
                args.n_steps, inference_cfg_rate=args.cfg_rate
            )
        result2 = r2_full[:, :, ref_prompt_len:]         # [1, 80, T_gen]

        # ---- result3：out_wav 的 mel 频谱（参考）---------------------
        try:
            result3 = mel_from_wav(sample.out_wav, mel_fn).to(device)   # [1, 80, T_ref]
        except Exception as e:
            print(f"  [{sample.stem}] mel_from_wav 失败：{e}")
            result3 = None

        # ---- 计算损失 --------------------------------------------------
        l1_r1r2 = l1_loss_trimmed(result1.cpu(), result2.cpu())
        l1_r1r3 = l1_loss_trimmed(result1.cpu(), result3.cpu()) if result3 is not None else float("nan")
        l1_r2r3 = l1_loss_trimmed(result2.cpu(), result3.cpu()) if result3 is not None else float("nan")

        all_r1r2.append(l1_r1r2)
        if result3 is not None:
            all_r1r3.append(l1_r1r3)
            all_r2r3.append(l1_r2r3)

        print(f"{sample.stem:<30} {l1_r1r2:>12.6f} {l1_r1r3:>12.6f} {l1_r2r3:>12.6f}")

        # ---- 第一个样本：将 result1 经 BigVGAN 声码器保存为音频 ------
        if not first_sample_done:
            first_sample_done = True
            with torch.no_grad():
                wav = bigvgan_model(result1.float()).squeeze(1)  # [1, T_wav]
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu().to(torch.int16)
            out_wav_path = output_dir / f"{sample.stem}.wav"
            torchaudio.save(str(out_wav_path), wav, 22050)
            print(f"  [Audio] result1 已保存 → {out_wav_path}")

    # ---- 汇总统计 -------------------------------------------------------
    print("=" * 70)
    if all_r1r2:
        mean_r1r2 = sum(all_r1r2) / len(all_r1r2)
        mean_r1r3 = sum(all_r1r3) / len(all_r1r3) if all_r1r3 else float("nan")
        mean_r2r3 = sum(all_r2r3) / len(all_r2r3) if all_r2r3 else float("nan")
        print(f"{'平均':<30} {mean_r1r2:>12.6f} {mean_r1r3:>12.6f} {mean_r2r3:>12.6f}")
    print("=" * 70)
    print()
    print("说明：")
    print("  L1(r1, r2)  ── 自写 CFM vs 原版 CFM（理应 ≈ 0，验证实现等价性）")
    print("  L1(r1, r3)  ── 自写 CFM vs 已生成音频 mel（信息性参考）")
    print("  L1(r2, r3)  ── 原版 CFM vs 已生成音频 mel（BigVGAN 逆变换的误差下限）")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_test(args)
