from data_provider.data_loader import (
    Dataset_CFM_Phase1,
    collate_cfm_phase1,
)

from torch.utils.data import DataLoader

data_dict = {
    "cfm_phase1": Dataset_CFM_Phase1,
}



def data_provider(args, flag: str):
    Data = data_dict[args.data]

    if flag in {'val', 'test'}:
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid

    data_set = Data(
        root_dir=args.data_root,
        split=flag,
        split_ratio=args.train_split_ratio,
        seed=args.seed,
        filter_enabled=args.filter_by_mse,
        mse_threshold=args.mse_threshold,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        num_mels=args.num_mels,
        hop_size=args.hop_size,
        win_size=args.win_size,
        fmin=args.fmin,
        fmax=args.fmax,
        tier_name=args.tier_name,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_cfm_phase1,
    )
    return data_set, data_loader