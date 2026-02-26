import os
import time
import torch
from tqdm.auto import tqdm

from exp.basic import Exp_Basic
from data_provider.data_factory import data_provider
from modules.cfm.flow_matching import LipSyncCFM
from modules.mel_strech.mel_transform import GlobalWarpTransformer
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

	def _run_one_epoch(self, loader, train: bool, stage: str):
		total_loss = 0.0
		count = 0

		if train:
			self.model.train()
		else:
			self.model.eval()

		ctx = torch.enable_grad() if train else torch.no_grad()
		with ctx:
			progress = tqdm(loader, desc=stage, leave=False, dynamic_ncols=True)
			for batch in progress:
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
				avg_loss = total_loss / max(count, 1)
				progress.set_postfix(loss=f"{float(loss.item()):.6f}", avg_loss=f"{avg_loss:.6f}")

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
		self.scheduler = torch.optim.lr_scheduler.LinearLR(
			self.optimizer,
			start_factor=1.0,
			end_factor=getattr(self.args, 'lr_end_factor', 0.1),
			total_iters=self.args.train_epochs,
		)

		best_val = float("inf")
		stale_epochs = 0

		logger.info(f"Train samples: {len(train_data)} | Val samples: {len(val_data)} | Test samples: {len(test_data)}")
		for epoch in range(1, self.args.train_epochs + 1):
			t0 = time.time()
			train_loss = self._run_one_epoch(train_loader, train=True, stage=f"Train {epoch}/{self.args.train_epochs}")
			val_loss = self._run_one_epoch(val_loader, train=False, stage=f"Val {epoch}/{self.args.train_epochs}")

			cur_lr = self.optimizer.param_groups[0]['lr']
			self.scheduler.step()
			logger.info(
				f"Epoch {epoch}/{self.args.train_epochs} | "
				f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
				f"lr={cur_lr:.2e} | time={time.time()-t0:.1f}s"
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

	def _get_vocoder(self) -> GlobalWarpTransformer:
		"""Lazily build a vocoder-enabled warper for mel-to-wav conversion."""
		if not hasattr(self, "_vocoder") or self._vocoder is None:
			device = "cuda" if self.args.use_gpu else "cpu"
			self._vocoder = GlobalWarpTransformer(
				use_vocoder=True,
				device=device,
				verbose=False,
			)
		return self._vocoder

	def test(self, setting: str, test: int = 0):
		_, test_loader = self._get_data("test")

		ckpt_dir = os.path.join(self.args.checkpoints, setting)
		best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
		if os.path.exists(best_ckpt_path):
			state = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
			self.model.load_state_dict(state["model"], strict=False)
			logger.info(f"Loaded checkpoint: {best_ckpt_path}")

		test_loss = self._run_one_epoch(test_loader, train=False, stage="Test")
		logger.info(f"Test loss: {test_loss:.6f}")

		# --- Inference + audio output ---
		output_dir = os.path.join(ckpt_dir, "test_outputs")
		os.makedirs(output_dir, exist_ok=True)
		vocoder = self._get_vocoder()

		self.model.eval()
		progress = tqdm(test_loader, desc="Infer+Save", dynamic_ncols=True)
		with torch.no_grad():
			for j, batch in enumerate(progress):
				x0 = batch["x0"].to(self.device)
				phoneme_ids = batch["phoneme_ids"].to(self.device)
				x_lens = batch["x_lens"].to(self.device)
				x1 = batch["x1"].to(self.device)
				pair_keys = batch["pair_key"]
				x_mean = batch["x_mean"].to(self.device)  # [B]
				x_std  = batch["x_std"].to(self.device)   # [B]

				pred_mel = self.model.inference(
					stretched_mel=x0,
					phoneme_ids=phoneme_ids,
					lip_embedding=None,
					x_lens=x_lens,
					steps=getattr(self.args, 'inference_steps', 32),
					cfg_scale=getattr(self.args, 'inference_cfg_rate', 0.5),
				)

				# Denormalize: bring mels back to original log-mel scale for vocoder
				scale = x_std[:, None, None]   # [B,1,1]
				bias  = x_mean[:, None, None]  # [B,1,1]
				pred_mel_out = pred_mel * scale + bias
				x0_out       = x0      * scale + bias
				x1_out       = x1      * scale + bias

				for i, key in enumerate(pair_keys):
					safe_key = str(key).replace("/", "_").replace(" ", "_")
					vocoder.save_audio(x0_out[i : i + 1],   os.path.join(output_dir, f"{safe_key}_x0.wav"))
					vocoder.save_audio(pred_mel_out[i : i + 1], os.path.join(output_dir, f"{safe_key}_pred.wav"))
					vocoder.save_audio(x1_out[i : i + 1],   os.path.join(output_dir, f"{safe_key}_x1.wav"))
				
				if j > 6:
					break

		logger.info(f"Test outputs saved to: {output_dir}")
		return test_loss
