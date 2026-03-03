import os
import torch
import json
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
        
    def _save_args(self, path):
        # 保存训练配置
        args_path = os.path.join(path, "args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=2, default=str)
        logger.info(f"Args saved to: {args_path}")
        
    def _save_state(self, path, epoch, model_state, optimizer_state=None, scheduler_state=None):
        # 保存模型和优化器状态
        state = {
            "epoch": epoch,
            "model_state": model_state,
        }
        if optimizer_state is not None:
            state["optimizer_state"] = optimizer_state
        if scheduler_state is not None:
            state["scheduler_state"] = scheduler_state
        state_path = os.path.join(path, f"checkpoint_epoch_{epoch}.pth")
        torch.save(state, state_path)
        logger.info(f"Checkpoint saved to: {state_path}")

    def _save_training_log(self, path, logs: list):
        """将所有 epoch 的训练指标（loss、lr 等）以 JSON 列表形式保存到文件。"""
        log_path = os.path.join(path, "training_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, default=str)
        logger.info(f"Training log saved to: {log_path}")


    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass