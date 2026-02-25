from data_provider.data_loader import (
    Dataset_CFM_Phase1,
)

from torch.utils.data import DataLoader

data_dict = {
    "cfm_phase1": Dataset_CFM_Phase1,
}



def data_provider(args, flag: str):
    pass