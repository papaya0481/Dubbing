import argparse
import random
from types import SimpleNamespace

import numpy as np
import torch

from exp.cfm.phase1 import Exp_CFM_Phase1


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="CFM Phase1 Trainer")

	parser.add_argument("--is_training", type=int, required=True, default=1, help="1 for train+test, 0 for test only")
	parser.add_argument("--model_id", type=str, required=True, default="cfm_phase1", help="experiment model id")
	parser.add_argument("--model", type=str, required=True, default="LipSyncCFM", help="model name")
	parser.add_argument("--exp_name", type=str, default="cfm_phase1", help="experiment entry name")

	parser.add_argument("--data", type=str, default="cfm_phase1", help="dataset key in data_factory")
	parser.add_argument("--data_root", type=str, required=True, help="root dir containing ost/ and aligned/")
	parser.add_argument("--filter_by_mse", action="store_true", default=True, help="filter by mse in metadata csv")
	parser.add_argument("--mse_threshold", type=float, default=0.08, help="mse threshold for filtering")
	parser.add_argument("--train_split_ratio", type=float, default=0.95, help="train/test split ratio")
	parser.add_argument("--tier_name", type=str, default="words", help="TextGrid tier used for alignment")

	parser.add_argument("--sample_rate", type=int, default=22050)
	parser.add_argument("--n_fft", type=int, default=1024)
	parser.add_argument("--num_mels", type=int, default=80)
	parser.add_argument("--hop_size", type=int, default=256)
	parser.add_argument("--win_size", type=int, default=1024)
	parser.add_argument("--fmin", type=int, default=0)
	parser.add_argument("--fmax", type=int, default=11025)

	parser.add_argument("--hidden_dim", type=int, default=512)
	parser.add_argument("--num_heads", type=int, default=8)
	parser.add_argument("--depth", type=int, default=8)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--ff_mult", type=int, default=4)
	parser.add_argument("--cond_dim", type=int, default=80)
	parser.add_argument("--mu_dim", type=int, default=80)
	parser.add_argument("--phoneme_vocab_size", type=int, default=8194)
	parser.add_argument("--lip_dim", type=int, default=512)
	parser.add_argument("--long_skip_connection", action="store_true", default=False)

	parser.add_argument("--t_scheduler", type=str, default="linear", choices=["linear", "cosine"])
	parser.add_argument("--training_cfg_rate", type=float, default=0.1)
	parser.add_argument("--inference_cfg_rate", type=float, default=0.5)

	parser.add_argument("--checkpoints", type=str, default="./checkpoints")
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--itr", type=int, default=1)
	parser.add_argument("--train_epochs", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--patience", type=int, default=5)
	parser.add_argument("--learning_rate", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--max_grad_norm", type=float, default=1.0)

	parser.add_argument("--seed", type=int, default=2026)
	parser.add_argument("--use_gpu", type=bool, default=True)
	parser.add_argument("--gpu", type=int, default=0)
	parser.add_argument("--use_multi_gpu", action="store_true", default=False)
	parser.add_argument("--devices", type=str, default="0,1,2,3")

	return parser


def inject_nested_cfg(args: argparse.Namespace) -> argparse.Namespace:
	args.DiT = SimpleNamespace(
		in_channels=args.num_mels,
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
	)
	return args


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
	if args.use_gpu and args.use_multi_gpu:
		args.devices = args.devices.replace(" ", "")
		device_ids = args.devices.split(",")
		args.device_ids = [int(d) for d in device_ids]
		args.gpu = args.device_ids[0]

	args = inject_nested_cfg(args)

	print("Args in experiment:")
	print(args)

	Exp = Exp_CFM_Phase1

	if args.is_training:
		for ii in range(args.itr):
			setting = f"{args.model_id}_{args.model}_{args.exp_name}_{ii}"
			exp = Exp(args)
			print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
			exp.train(setting)
			print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
			exp.test(setting)
			torch.cuda.empty_cache()
	else:
		setting = f"{args.model_id}_{args.model}_{args.exp_name}_0"
		exp = Exp(args)
		print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
		exp.test(setting, test=1)
		torch.cuda.empty_cache()
