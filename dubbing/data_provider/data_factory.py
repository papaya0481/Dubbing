from data_provider.data_loader import (
    Dataset_CFM_Phase1,
    collate_cfm_phase1,
)

from torch.utils.data import DataLoader

data_dict = {
    "cfm_phase1": Dataset_CFM_Phase1,
}



def data_provider(args, flag: str):
    Data = data_dict[args.data.dataset]

    if flag in {'val', 'test'}:
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.data.batch_size

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
        collate_fn=collate_cfm_phase1,
    )
    return data_set, data_loader