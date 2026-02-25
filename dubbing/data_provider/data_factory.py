from data_provider.data_loader import (
    Dataset_CFM_Phase1,
)

from torch.utils.data import DataLoader

data_dict = {
    "cfm_phase1": Dataset_CFM_Phase1,
}



def data_provider(args, flag: str):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid

    data_set = Data(

    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader