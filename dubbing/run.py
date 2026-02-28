import argparse
import random
from types import SimpleNamespace

import numpy as np
import torch

from exp.cfm.phase1 import Exp_CFM_Phase1
from logger import get_logger, set_log_level, show_setting


logger = get_logger("dubbing.run")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="CFM Phase1 Trainer")

	parser.add_argument("--is_training", type=int, required=True, default=1, help="1 for train+test, 0 for test only")
	parser.add_argument("--model_id", type=str, required=True, default="cfm_phase1", help="experiment model id")
	parser.add_argument("--model", type=str, required=True, default="LipSyncCFM", help="model name")
	parser.add_argument("--exp_name", type=str, help="experiment entry name")

	parser.add_argument("--data", type=str, default="cfm_phase1", help="dataset key in data_factory")
	parser.add_argument("--data_root", type=str, required=True, help="root dir containing ost/ and aligned/")
	parser.add_argument("--filter_by_mse", action="store_true", default=True, help="filter by mse in metadata csv")
	parser.add_argument("--mse_threshold", type=float, default=8, help="mse threshold for filtering")
	parser.add_argument("--train_split_ratio", type=float, default=0.9, help="train/test split ratio")
	parser.add_argument("--tier_name", type=str, default="phones", help="TextGrid tier used for alignment")
	parser.add_argument("--phoneme_map_path", type=str, default="dubbing/modules/english_us_arpa_300.json", help="phoneme id mapping json")

	parser.add_argument("--hidden_dim", type=int, default=512)
	parser.add_argument("--num_heads", type=int, default=8)
	parser.add_argument("--depth", type=int, default=8)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--ff_mult", type=int, default=4)
	parser.add_argument("--cond_dim", type=int, default=128)
	parser.add_argument("--mu_dim", type=int, default=80)
	parser.add_argument("--phoneme_vocab_size", type=int, default=72)
	parser.add_argument("--lip_dim", type=int, default=0)
	parser.add_argument("--long_skip_connection", action="store_true", default=False)
	parser.add_argument("--generate_from_noise", action="store_true", default=False, help="whether to generate from pure noise instead of stretched mel + noise")

	parser.add_argument("--t_scheduler", type=str, default="linear", choices=["linear", "cosine"])
	parser.add_argument("--training_cfg_rate", type=float, default=0.2)
	parser.add_argument("--inference_cfg_rate", type=float, default=0.7)
	parser.add_argument("--training_temperature", type=float, default=0.1, help="noise scale added to stretched mel during training")
	parser.add_argument("--inference_steps", type=int, default=25, help="number of Euler steps for inference")

	parser.add_argument("--checkpoints", type=str, default="./checkpoints")
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--itr", type=int, default=1)
	parser.add_argument("--train_epochs", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--learning_rate", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--max_grad_norm", type=float, default=1.0)
	parser.add_argument("--lr_reduce_factor", type=float, default=0.6,
		help="ReduceLROnPlateau: factor to multiply lr by on plateau")
	parser.add_argument("--lr_reduce_patience", type=int, default=3,
		help="ReduceLROnPlateau: epochs with no improvement before reducing lr")
	parser.add_argument("--lr_min", type=float, default=1e-5,
		help="ReduceLROnPlateau: minimum lr")
	parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine"],
		help="learning rate scheduler type (linear or cosine)")
	parser.add_argument("--early_stop_patience", type=int, default=10,
		help="stop training after this many epochs with no val improvement")

	parser.add_argument("--seed", type=int, default=2026)
	parser.add_argument("--log_level", type=str, default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		help="logging verbosity; DEBUG also prints full model structure")
	parser.add_argument("--use_gpu", type=bool, default=True)
	parser.add_argument("--gpu", type=int, default=0)
	parser.add_argument("--use_multi_gpu", action="store_true", default=False)
	parser.add_argument("--devices", type=str, default="0,1,2,3")

	return parser


def inject_nested_cfg(args: argparse.Namespace) -> argparse.Namespace:
	args.DiT = SimpleNamespace(
		in_channels=80,
		hidden_dim=args.hidden_dim,
		num_heads=args.num_heads,
		depth=args.depth,
		cond_dim=args.cond_dim,
		mu_dim=args.mu_dim,
		dropout=args.dropout,
		ff_mult=args.ff_mult,
		long_skip_connection=args.long_skip_connection,
		phoneme_vocab_size=args.phoneme_vocab_size,
		lip_dim=args.lip_dim,
		static_chunk_size=50,
		num_decoding_left_chunks=2,
	)
	args.CFM = SimpleNamespace(
		t_scheduler=args.t_scheduler,
		training_cfg_rate=args.training_cfg_rate,
		inference_cfg_rate=args.inference_cfg_rate,
		training_temperature=args.training_temperature,
		generate_from_noise=args.generate_from_noise,
	)
	return args


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	import logging
	set_log_level(getattr(logging, args.log_level.upper(), logging.INFO))

	args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
	if args.use_gpu and args.use_multi_gpu:
		args.devices = args.devices.replace(" ", "")
		device_ids = args.devices.split(",")
		args.device_ids = [int(d) for d in device_ids]
		args.gpu = args.device_ids[0]

	args = inject_nested_cfg(args)

	logger.info("Args in experiment:")
	logger.info(str(args))

	Exp = Exp_CFM_Phase1
 
	if args.exp_name is None:
		# use date + hash as default exp_name to avoid overwriting previous results
		import datetime
		import hashlib
		args.exp_name = datetime.datetime.now().strftime("%m%d%H") + "_" + hashlib.md5(str(args).encode()).hexdigest()[:4]

	if args.is_training:
		for ii in range(args.itr):
			setting_parts = [args.model_id, args.model, args.exp_name, ii]
			show_setting("Training Setting", setting_parts)
			setting = "_".join([str(x) for x in setting_parts])
			exp = Exp(args)
			logger.info(f"Start training: {setting}")
			exp.train(setting)
			logger.info(f"Testing: {setting}")
			exp.test(setting)
			torch.cuda.empty_cache()
	else:
		setting_parts = [args.model_id, args.model, args.exp_name, 0]
		show_setting("Testing Setting", setting_parts)
		setting = "_".join([str(x) for x in setting_parts])
		exp = Exp(args)
		logger.info(f"Testing: {setting}")
		exp.test(setting, test=1)
		torch.cuda.empty_cache()
