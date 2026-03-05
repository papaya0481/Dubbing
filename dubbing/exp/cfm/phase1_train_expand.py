import os
import time
import torch
from tqdm.auto import tqdm

from exp.cfm.phase1 import Exp_CFM_Phase1
from data_provider.data_factory import data_provider
from modules.cfm.flow_matching import LipSyncCFM
from modules.mel_strech.mel_transform import GlobalWarpTransformer
from logger import get_logger


logger = get_logger("dubbing.exp.cfm")


class Exp_CFM_Phase1_TrainExpand(Exp_CFM_Phase1):
	def __init__(self, args):
		self.best_ckpt_path = None
		super().__init__(args)

	def train(self, setting: str):
		train_data, train_loader = self._get_data("train")
		val_data, val_loader = self._get_data("val")
		test_data, test_loader = self._get_data("test")

		os.makedirs(self.args.system.checkpoints, exist_ok=True)
		ckpt_dir = os.path.join(self.args.system.checkpoints, setting)
		os.makedirs(ckpt_dir, exist_ok=True)
		self.best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
		self._save_args(ckpt_dir)

		t = self.args.training
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)
		self.scheduler = self._build_scheduler(self.optimizer)

		best_val = float("inf")
		stale_epochs = 0
		early_stop_patience = t.early_stop_patience
		self.epoch_logs = []

		logger.info(f"Train samples: {len(train_data)} | Val samples: {len(val_data)} | Test samples: {len(test_data)}")
		for epoch in range(1, t.epochs + 1):
			t0 = time.time()
			train_loss = self._run_one_epoch(train_loader, train=True, stage=f"Train {epoch}/{t.epochs}")
			val_loss = self._run_one_epoch(val_loader, train=False, stage=f"Val {epoch}/{t.epochs}")

			cur_lr = self.optimizer.param_groups[0]['lr']
			self.scheduler.step()
			new_lr = self.optimizer.param_groups[0]['lr']
			lr_tag = f" LR={cur_lr:.2e} -> {new_lr:.2e}" if new_lr != cur_lr else f" LR={cur_lr:.2e}]"
			logger.info(
				f"Epoch {epoch}/{t.epochs} | "
				f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} |"
				f"{lr_tag} | time={time.time()-t0:.1f}s"
			)
			self.epoch_logs.append({
				"epoch": epoch,
				"train_loss": train_loss,
				"val_loss": val_loss,
				"lr": cur_lr,
				"time": round(time.time() - t0, 1),
			})

			if val_loss < best_val:
				best_val = val_loss
				stale_epochs = 0
				torch.save({"model": self.model.state_dict(), "args": self.args}, self.best_ckpt_path)
				logger.info(f"Saved best checkpoint: {self.best_ckpt_path}")
			else:
				stale_epochs += 1

			if stale_epochs >= early_stop_patience:
				logger.warning(f"Early stopping triggered at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
				break

			if epoch % 5 == 0 and os.path.exists(self.best_ckpt_path):
				logger.info(f"Periodic test at epoch {epoch} ...")
				self.test(setting, epoch=epoch)
				train_output_dir = os.path.join(ckpt_dir, f"train_outputs@train_{epoch}")
				logger.info(f"Periodic train-subset inference at epoch {epoch} ...")
				self._infer_and_save(
					loader=train_loader,
					output_dir=train_output_dir,
					stage_name="TrainInfer+Save",
					max_batches=t.train_infer_max_batches,
				)

		return self.model

	def test(self, setting: str, test: int = 0, epoch: int = 0):
		_, test_loader = self._get_data("test")

		ckpt_dir = os.path.join(self.args.system.checkpoints, setting)
		best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
		if os.path.exists(best_ckpt_path):
			state = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
			self.model.load_state_dict(state["model"], strict=False)
			logger.info(f"Loaded checkpoint: {best_ckpt_path}")

		# logger.debug(f"Model structure:\n{self.model}")

		test_loss = self._run_one_epoch(test_loader, train=False, stage="Test")
		logger.info(f"Test loss: {test_loss:.6f}")

		# 保存所有 epoch 的训练指标
		if hasattr(self, 'epoch_logs') and self.epoch_logs:
			self._save_training_log(ckpt_dir, self.epoch_logs)

		# --- Inference + audio output ---
		output_dir = os.path.join(ckpt_dir, f"test_outputs@test{test}_{epoch}")
		self._infer_and_save(
			loader=test_loader,
			output_dir=output_dir,
			stage_name="Infer+Save",
			max_batches=self.args.training.test_infer_max_batches,
		)

		logger.info(f"Test outputs saved to: {output_dir}")
		return test_loss

	def _infer_and_save(self, loader, output_dir: str, stage_name: str, max_batches: int = 16):
		os.makedirs(output_dir, exist_ok=True)
		vocoder = self._get_vocoder()

		self.model.eval()
		progress = tqdm(loader, desc=stage_name, dynamic_ncols=True)
		with torch.no_grad():
			for j, batch in enumerate(progress):
				cond_mel = batch["cond_mel"].to(self.device)
				phoneme_ids = batch["phoneme_ids"].to(self.device)
				x_lens = batch["x_lens"].to(self.device)
				x1 = batch["x1"].to(self.device)
				pair_keys = batch["pair_key"]
				x_mean = batch["x_mean"].to(self.device)
				x_std = batch["x_std"].to(self.device)

				pred_mel = self.model.inference(
					stretched_mel=cond_mel,
					phoneme_ids=phoneme_ids,
					lip_embedding=None,
					x_lens=x_lens,
					steps=self.args.training.inference_steps,
					cfg_scale=self.args.model.CFM.inference_cfg_rate,
					temperature=self.args.model.CFM.training_temperature,
				)

				scale = x_std[:, None, None]
				bias = x_mean[:, None, None]
				pred_mel_out  = pred_mel  * scale + bias
				cond_mel_out  = cond_mel  * scale + bias
				x1_out        = x1        * scale + bias

				for i, key in enumerate(pair_keys):
					safe_key = str(key).replace("/", "_").replace(" ", "_")
					vocoder.save_audio(cond_mel_out[i : i + 1], os.path.join(output_dir, f"{safe_key}_cond.wav"))
					vocoder.save_audio(pred_mel_out[i : i + 1], os.path.join(output_dir, f"{safe_key}_pred.wav"))
					vocoder.save_audio(x1_out[i : i + 1], os.path.join(output_dir, f"{safe_key}_x1.wav"))

				if j >= max_batches - 1:
					break
