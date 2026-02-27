import os
import torch
from logger import get_logger


logger = get_logger("dubbing.exp")

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            logger.info(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device('cpu')
            logger.info("Use CPU")
        return device
    
    def _build_scheduler(self, optimizer):
        lr_min = getattr(self.args, 'lr_min', 1e-5)
        sched_type = getattr(self.args, 'lr_scheduler', 'cosine').lower()
        n = self.args.train_epochs

        if sched_type == 'linear':
            end_factor = lr_min / max(self.args.learning_rate, 1e-12)
            logger.info(f"Scheduler: LinearLR  end_factor={end_factor:.2e}  total_iters={n}")
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=n,
            )
        elif sched_type == 'cosine':
            logger.info(f"Scheduler: CosineAnnealingLR  eta_min={lr_min:.2e}  T_max={n}")
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=n,
                eta_min=lr_min,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler '{sched_type}'. Choose 'linear' or 'cosine'.")

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass