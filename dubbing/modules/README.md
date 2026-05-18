# dubbing/modules/

Model implementations and processing utilities used by the training package. The two main model families are `cfm/` for the original phoneme-conditioned experiment and `cfm_index/` for the IndexTTS2-style experiment.

## Submodules

### cfm/

Original LipSyncCFM model — a DiT-based continuous flow matching model for mel spectrogram generation conditioned on phonemes and stretched mel input.

| File | Purpose |
|---|---|
| `DiT.py` | Diffusion Transformer backbone (self-attention + FiLM conditioning) |
| `attn.py` | Custom attention modules |
| `flow_matching.py` | `LipSyncCFM` class: CFM training loss + ODE inference (Euler/Dopri5) |
| `__init__.py` | Package init |

Training forward signature: `(clean_mel, stretched_mel, phoneme_ids, lip_embedding, x_lens, cond=None, spks=None)` -> loss. In the current experiment path, `lip_embedding` is passed as `None`; lips conditioning is implemented in `cfm_index/CrossAttnCFM`.

### cfm_index/

Port of IndexTTS2's `s2mel` CFM for mel generation from semantic conditions.

| File | Purpose |
|---|---|
| `DiT.py` | DiT backbone adapted for IndexTTS2-style content conditioning |
| `transformer.py` | Transformer encoder components |
| `wavenet.py` | WaveNet final layer (convolution-based mel decoder) |
| `encodec.py` | EnCodec-based audio codec wrapper |
| `cross_attn.py` | Cross-attention layer used by `CrossAttnCFM` for lips conditioning |
| `flow_matching.py` | `CFM` and `CrossAttnCFM` classes |
| `__init__.py` | Package init |

`CFM.forward(...)` trains on `x1`, `x_lens`, `prompt_lens`, `cond`, and `style`. `CrossAttnCFM.forward(...)` splits the condition into `prompt_cond` and `infer_cond`, then cross-attends `infer_cond` to `lips_feat` before calling the DiT estimator.

### mel_strech/

Mel spectrogram warping based on TextGrid alignment. Used to stretch/compress audio timing between source and target phoneme durations.

| File | Purpose |
|---|---|
| `meldataset.py` | `get_mel_spectrogram()` — extract log-mel spectrograms from audio |
| `mel_transform.py` | `GlobalWarpTransformer` — main warping class: loads audio, extracts mel, applies PCHIP-interpolated time warping from source→target TextGrid, saves warped audio via BigVGAN vocoder |
| `__init__.py` | Package init |

### lips/

Lip-reading and phoneme-lips alignment code kept as a Git submodule at `dubbing/modules/lips`.

| Directory | Purpose |
|---|---|
| `data/` | Dataset classes for MELD and LRS2, phoneme vocabulary, TextGrid parser |
| `models/` | `conformer_xattn.py` (Conformer with cross-attention), `phoneme_split_net.py`, `resnet_new.py` |
| `scripts/` | Training and evaluation scripts: Viterbi alignment, duration prediction, data preparation |
| `utils/` | Data augmentation, logging, optimizer utilities, sequence decoder |
| `tests/` | Unit test for MELD TextGrid word-phone mapping |
| `train_phoneme_split.py` | Main training entry point for phoneme split network |
| `finetune_meld.py` | Fine-tuning script for MELD dataset |

### semantic_stretch/

Semantic token warping — aligns GPT semantic codes (50 fps) to a target TextGrid using `F.grid_sample` for fully vectorized temporal interpolation.

| File | Purpose |
|---|---|
| `semantic_transform.py` | `SemanticTransformer` class — warps latent sequences from source to target timing, with support for semantic-rate or mel-rate input |
| `__init__.py` | Package init |

Design mirrors `GlobalWarpTransformer` from `mel_strech/` but operates on discrete latent codes rather than mel spectrograms.

### Other files

| File | Purpose |
|---|---|
| `mfa_alinger.py` | `MFAAligner` — wraps Montreal Forced Aligner for per-utterance phoneme alignment. Uses Kalpy + MFA internals for G2P, lexicon compilation, and acoustic alignment |
| `english_us_arpa_300.json` | Small ARPA phoneme/category mapping used by the original CFM data path |
