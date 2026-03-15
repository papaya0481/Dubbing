import argparse
import datetime
import hashlib
import logging
import random

import numpy as np
import torch

from config import apply_overrides, config_to_dict, load_config
from exp.cfm.phase1_train_expand import Exp_CFM_Phase1_TrainExpand
from exp.cfm.cfm_index_phase1_train_expand import Exp_CFM_Index_Phase1_TrainExpand
from exp.cfm.cfm_index_phase1_lips import Exp_CFM_Index_Phase1_Lips
from logger import get_logger, set_log_level, show_setting

EXP_MAP = {
    "LipSyncCFM": Exp_CFM_Phase1_TrainExpand,
    "CFM_Index":  Exp_CFM_Index_Phase1_TrainExpand,
    "CFM_Index_Lips": Exp_CFM_Index_Phase1_Lips,
}

logger = get_logger("dubbing.run")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="CFM Phase1 Trainer",
		formatter_class=argparse.RawTextHelpFormatter,
	)
	parser.add_argument(
		"--config", required=True, metavar="PATH",
		help="Path to a YAML config file (e.g. dubbing/configs/default.yaml)",
	)
	parser.add_argument(
		"overrides", nargs="*", metavar="key=value",
		help=(
			"Optional dotted overrides applied on top of the YAML, e.g.:\n"
			"  data.root=/my/data  training.learning_rate=5e-4  system.gpu=1"
		),
	)
	return parser


if __name__ == "__main__":
	parser = build_parser()
	cli = parser.parse_args()

	cfg = load_config(cli.config)
	if cli.overrides:
		apply_overrides(cfg, cli.overrides)

	# --- Reproducibility ---
	random.seed(cfg.system.seed)
	np.random.seed(cfg.system.seed)
	torch.manual_seed(cfg.system.seed)

	# --- Logging ---
	set_log_level(getattr(logging, cfg.system.log_level.upper(), logging.INFO))

	# --- GPU setup ---
	cfg.system.use_gpu = bool(torch.cuda.is_available() and cfg.system.use_gpu)
	if cfg.system.use_gpu and cfg.system.use_multi_gpu:
		device_ids = [int(d) for d in cfg.system.devices.replace(" ", "").split(",")]
		cfg.system.device_ids = device_ids
		cfg.system.gpu = device_ids[0]

	# --- Auto exp_name ---
	if cfg.exp_name is None:
		time_now = datetime.datetime.now().strftime("%m%d%H")
		cfg.exp_name = time_now + "_" + hashlib.md5(
			(str(config_to_dict(cfg)) + time_now).encode()
		).hexdigest()[:4]

	logger.info(f"Config:\n{config_to_dict(cfg)}")

	Exp = EXP_MAP.get(cfg.model_name)
	if Exp is None:
		raise ValueError(
			f"Unknown model_name '{cfg.model_name}'. "
			f"Available: {list(EXP_MAP.keys())}"
		)

	if cfg.is_training:
		for ii in range(cfg.itr):
			setting_parts = [cfg.model_id, cfg.model_name, cfg.exp_name, ii]
			show_setting("Training Setting", setting_parts)
			setting = "_".join(str(x) for x in setting_parts)
			exp = Exp(cfg)
			logger.info(f"Start training: {setting}")
			exp.train(setting)
			logger.info(f"Testing: {setting}")
			exp.test(setting, test=1)
			torch.cuda.empty_cache()
	else:
		setting_parts = [cfg.model_id, cfg.model_name, cfg.exp_name, 0]
		show_setting("Testing Setting", setting_parts)
		setting = "_".join(str(x) for x in setting_parts)
		exp = Exp(cfg)
		logger.info(f"Testing: {setting}")
		exp.test(setting, test=1)
		torch.cuda.empty_cache()
