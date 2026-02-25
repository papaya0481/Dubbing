import os
import time
import torch

from exp.basic import Exp_Basic
from data_provider.data_factory import data_provider
from modules.cfm.flow_matching import LipSyncCFM
from logger import get_logger


logger = get_logger("dubbing.exp.cfm")


class Exp_CFM_Phase1(Exp_Basic):
	def __init__(self, args):
		self.best_ckpt_path = None
		super().__init__(args)

	def _build_model(self):
		model = LipSyncCFM(self.args)
		if self.args.use_multi_gpu and self.args.use_gpu:
			model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
		return model

	def _get_data(self, flag: str):
		data_set, data_loader = data_provider(self.args, flag)
		return data_set, data_loader

	def _run_one_epoch(self, loader, train: bool):
		total_loss = 0.0
		count = 0

		if train:
			self.model.train()
		else:
			self.model.eval()

		ctx = torch.enable_grad() if train else torch.no_grad()
		with ctx:
			for batch in loader:
				x0 = batch["x0"].to(self.device)
				x1 = batch["x1"].to(self.device)
				phoneme_ids = batch["phoneme_ids"].to(self.device)
				x_lens = batch["x_lens"].to(self.device)

				if train:
					self.optimizer.zero_grad(set_to_none=True)

				loss = self.model(
					clean_mel=x1,
					stretched_mel=x0,
					phoneme_ids=phoneme_ids,
					lip_embedding=None,
					x_lens=x_lens,
					cond=None,
					spks=None,
				)

				if train:
					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
					self.optimizer.step()

				total_loss += float(loss.item())
				count += 1

		return total_loss / max(count, 1)

	def train(self, setting: str):
		train_data, train_loader = self._get_data("train")
		val_data, val_loader = self._get_data("val")
		test_data, test_loader = self._get_data("test")

		os.makedirs(self.args.checkpoints, exist_ok=True)
		ckpt_dir = os.path.join(self.args.checkpoints, setting)
		os.makedirs(ckpt_dir, exist_ok=True)
		self.best_ckpt_path = os.path.join(ckpt_dir, "best.pth")

		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

		best_val = float("inf")
		stale_epochs = 0

		logger.info(f"Train samples: {len(train_data)} | Val samples: {len(val_data)} | Test samples: {len(test_data)}")
		for epoch in range(1, self.args.train_epochs + 1):
			t0 = time.time()
			train_loss = self._run_one_epoch(train_loader, train=True)
			val_loss = self._run_one_epoch(val_loader, train=False)

			logger.info(
				f"Epoch {epoch}/{self.args.train_epochs} | "
				f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
				f"time={time.time()-t0:.1f}s"
			)

			if val_loss < best_val:
				best_val = val_loss
				stale_epochs = 0
				torch.save({"model": self.model.state_dict(), "args": vars(self.args)}, self.best_ckpt_path)
				logger.info(f"Saved best checkpoint: {self.best_ckpt_path}")
			else:
				stale_epochs += 1

			if stale_epochs >= self.args.patience:
				logger.warning(f"Early stopping triggered at epoch {epoch}")
				break

		return self.model

	def test(self, setting: str, test: int = 0):
		_, test_loader = self._get_data("test")

		ckpt_dir = os.path.join(self.args.checkpoints, setting)
		best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
		if os.path.exists(best_ckpt_path):
			state = torch.load(best_ckpt_path, map_location=self.device)
			self.model.load_state_dict(state["model"], strict=False)
			logger.info(f"Loaded checkpoint: {best_ckpt_path}")

		test_loss = self._run_one_epoch(test_loader, train=False)
		logger.info(f"Test loss: {test_loss:.6f}")
		return test_loss
