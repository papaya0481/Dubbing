from data_provider.data_loader import (
    Dataset_CFM_Phase1,
    Dataset_CFM_Phase1_StretchEntireMel,
    Dataset_CFM_Index_Phase1,
    Dataset_CMF_Index_Phase1_ForLipsFeat,
)
from data_provider.collate_funcs import (
    collate_cfm_phase1,
    collate_cfm_index_phase1,
    collate_cmf_index_phase1_for_lipsfeat,
)

from torch.utils.data import DataLoader

data_dict = {
    "cfm_phase1":         Dataset_CFM_Phase1,
    "cfm_phase1_stretch": Dataset_CFM_Phase1_StretchEntireMel,
    "cfm_index_phase1":   Dataset_CFM_Index_Phase1,
    "cmf_index_phase1_for_lipsfeat": Dataset_CMF_Index_Phase1_ForLipsFeat,
}

collate_dict = {
    "cfm_phase1":         collate_cfm_phase1,
    "cfm_phase1_stretch": collate_cfm_phase1,
    "cfm_index_phase1":   collate_cfm_index_phase1,
    "cmf_index_phase1_for_lipsfeat": collate_cmf_index_phase1_for_lipsfeat,
}


def data_provider(args, flag: str):
    dataset_name = args.data.dataset
    Data         = data_dict[dataset_name]
    collate_fn   = collate_dict[dataset_name]

    if flag in {'val', 'test'}:
        shuffle_flag = False
        drop_last    = False
        batch_size   = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last    = False
        batch_size   = args.data.batch_size

    # cfm_index_phase1 family uses a different constructor signature
    if dataset_name == "cfm_index_phase1":
        data_set = Data(
            csv_path=args.data.csv_path,
            mel_h=args.preprocess.mel,
            preprocess=args.preprocess,
            sr_ref_16k=args.preprocess.sr_ref_16k,
            split=flag,
            split_ratio=args.data.train_split_ratio,
            seed=args.system.seed,
            max_ref_sec=args.data.max_ref_sec,
            max_gen_sec=args.data.max_gen_sec,
            max_code_len=args.data.max_code_len,
            cache_dir=getattr(args.data, "cache_dir", None),
            cache_batch_size=getattr(args.preprocess, "cache_batch_size", 16),
        )
    elif dataset_name == "cmf_index_phase1_for_lipsfeat":
        data_set = Data(
            flow_dataset_path=args.data.flow_dataset_path,
            mel_h=args.preprocess.mel,
            preprocess=args.preprocess,
            sr_ref_16k=args.preprocess.sr_ref_16k,
            split=flag,
            split_ratio=args.data.train_split_ratio,
            seed=args.system.seed,
            max_ref_sec=args.data.max_ref_sec,
            max_gen_sec=args.data.max_gen_sec,
            max_code_len=args.data.max_code_len,
            cache_dir=getattr(args.data, "cache_dir", None),
            cache_batch_size=getattr(args.preprocess, "cache_batch_size", 16),
            tier_name=getattr(args.data, "tier_name", "phones"),
        )
    else:
        data_set = Data(
            root_dir=args.data.root,
            split=flag,
            split_ratio=args.data.train_split_ratio,
            seed=args.system.seed,
            filter_enabled=args.data.filter_by_mse,
            mse_threshold=args.data.mse_threshold,
            tier_name=args.data.tier_name,
            phoneme_map_path=args.data.phoneme_map_path,
        )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.data.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return data_set, data_loader