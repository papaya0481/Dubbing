"""dubbing/data_provider/collate_funcs.py

Collate functions for all registered DataLoader pipelines.

Moved here from data_loader.py to keep dataset classes separate from
batching logic.

Exported:
  collate_cfm_phase1           – for Dataset_CFM_Phase1 / Dataset_CFM_Phase1_StretchEntireMel
  collate_cfm_index_phase1     – for Dataset_CFM_Index_Phase1
        collate_cfm_index_phase1_for_lipsfeat – for Dataset_CFM_Index_Phase1_ForLipsFeat
"""

from __future__ import annotations

from typing import Dict, List

import torch


# =============================================================================
# cfm_phase1 / cfm_phase1_stretch
# =============================================================================


def collate_cfm_phase1(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    lengths = torch.stack([item["x_len"] for item in batch], dim=0)
    max_len = int(lengths.max().item())
    n_mels  = batch[0]["cond_mel"].shape[0]

    cond_mel    = torch.zeros(len(batch), n_mels, max_len, dtype=batch[0]["cond_mel"].dtype)
    x1          = torch.zeros(len(batch), n_mels, max_len, dtype=batch[0]["x1"].dtype)
    phoneme_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    x_mean      = torch.stack([item["x_mean"] for item in batch], dim=0)  # [B]
    x_std       = torch.stack([item["x_std"]  for item in batch], dim=0)  # [B]

    pair_keys    = []
    mse_list     = []
    text_r1_list = []

    for i, item in enumerate(batch):
        t = int(item["x_len"].item())
        cond_mel[i, :, :t]    = item["cond_mel"][:, :t]
        x1[i, :, :t]          = item["x1"][:, :t]
        phoneme_ids[i, :t]    = item["phoneme_ids"][:t]
        pair_keys.append(item["pair_key"])
        mse_list.append(item["mse"])
        text_r1_list.append(item["text_r1"])

    return {
        "pair_key":    pair_keys,
        "cond_mel":    cond_mel,
        "x1":          x1,
        "phoneme_ids": phoneme_ids,
        "x_lens":      lengths,
        "x_mean":      x_mean,
        "x_std":       x_std,
        "mse":         mse_list,
        "text_r1":     text_r1_list,
    }


# =============================================================================
# cfm_index_phase1
# =============================================================================


def collate_cfm_index_phase1(batch: List[Dict]) -> Dict:
    """Collate pre-computed CFM conditions from Dataset_CFM_Index_Phase1.

    ``prompt_cond`` and ``infer_cond`` are returned as **separate** padded
    tensors.  The Exp is responsible for assembling the full condition tensor
    (e.g. via ``_assemble_cond``) before calling CFM.forward / CFM.inference.

    Keys returned
    -------------
    stems       list[str]
    x1_full     [B, num_mels, T_max]       ref_mel ++ x1_mel, zero-padded
    ref_mels    [B, num_mels, T_ref_max]   reference mel for CFM.inference
    prompt_cond [B, T_ref_max, 512]        length_regulator(S_ref), zero-padded
    infer_cond  [B, T_gen_max, 512]        length_regulator(S_infer), zero-padded
    style       [B, 192]
    x_lens      [B]   T_ref + T_gen per sample  (total CFM sequence length)
    prompt_lens [B]   T_ref per sample
    infer_lens  [B]   T_gen per sample
    """
    B        = len(batch)
    num_mels = batch[0]["ref_mel"].size(0)

    T_refs   = [item["ref_mel"].size(-1) for item in batch]
    T_gens   = [item["x1_mel"].size(-1)  for item in batch]
    T_totals = [r + g for r, g in zip(T_refs, T_gens)]

    T_ref_max   = max(T_refs)
    T_gen_max   = max(T_gens)
    T_total_max = max(T_totals)

    x1_full     = torch.zeros(B, num_mels, T_total_max)
    ref_mels    = torch.zeros(B, num_mels, T_ref_max)
    prompt_cond = torch.zeros(B, T_ref_max, 512)
    infer_cond  = torch.zeros(B, T_gen_max, 512)

    for i, item in enumerate(batch):
        T_r = T_refs[i]
        T_g = T_gens[i]
        x1_full[i, :, :T_r]        = item["ref_mel"]
        x1_full[i, :, T_r:T_r+T_g] = item["x1_mel"]
        ref_mels[i, :, :T_r]       = item["ref_mel"]
        prompt_cond[i, :T_r, :]    = item["prompt_cond"]
        infer_cond[i, :T_g, :]     = item["infer_cond"]

    style       = torch.stack([item["style"] for item in batch])   # [B, 192]
    x_lens      = torch.tensor(T_totals, dtype=torch.long)
    prompt_lens = torch.tensor(T_refs,   dtype=torch.long)
    infer_lens  = torch.tensor(T_gens,   dtype=torch.long)
    stems       = [item["stem"] for item in batch]

    return {
        "stems":       stems,
        "x1_full":     x1_full,
        "ref_mels":    ref_mels,
        "prompt_cond": prompt_cond,
        "infer_cond":  infer_cond,
        "style":       style,
        "x_lens":      x_lens,
        "prompt_lens": prompt_lens,
        "infer_lens":  infer_lens,
    }


def collate_cfm_index_phase1_for_lipsfeat(batch: List[Dict]) -> Dict:
    """Collate for Dataset_CFM_Index_Phase1_ForLipsFeat.

    Same outputs as ``collate_cfm_index_phase1`` plus padded lips features.
    """
    base = collate_cfm_index_phase1(batch)

    B = len(batch)
    hs_dim = int(batch[0]["lips_hidden_states"].size(-1))
    hs_lens = [int(item["lips_hidden_states"].size(0)) for item in batch]
    hs_max = max(hs_lens)

    lips_hidden_states = torch.zeros(B, hs_max, hs_dim, dtype=batch[0]["lips_hidden_states"].dtype)
    for i, item in enumerate(batch):
        t = hs_lens[i]
        lips_hidden_states[i, :t, :] = item["lips_hidden_states"]

    base["lips_hidden_states"] = lips_hidden_states
    base["lips_lens"] = torch.tensor(hs_lens, dtype=torch.long)
    base["lips_textgrids"] = [item["lips_textgrid"] for item in batch]
    base["source_textgrids"] = [item["source_textgrid"] for item in batch]
    return base
