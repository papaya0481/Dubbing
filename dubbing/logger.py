import logging
from typing import Mapping, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table


_CONSOLE = Console(color_system="auto")


def get_logger(name: str = "dubbing", level: int = logging.INFO) -> logging.Logger:
	logger = logging.getLogger(name)
	if logger.handlers:
		logger.setLevel(level)
		return logger

	handler = RichHandler(
		console=_CONSOLE,
		rich_tracebacks=True,
		show_time=True,
		show_level=True,
		show_path=False,
		markup=False,
  		log_time_format="%H:%M:%S",
	)
	formatter = logging.Formatter("[%(name)s] %(message)s")
	handler.setFormatter(formatter)

	logger.addHandler(handler)
	logger.setLevel(level)
	logger.propagate = False
	return logger


def set_log_level(level: int) -> None:
	logging.getLogger("dubbing").setLevel(level)


def show_setting(title: str, setting: Mapping[str, object] | Sequence[object]) -> None:
	table = Table(title=title, show_header=True, header_style="bold cyan")

	if isinstance(setting, Mapping):
		table.add_column("Field", style="green", no_wrap=True)
		table.add_column("Value", style="magenta")
		for k, v in setting.items():
			table.add_row(str(k), str(v))
	else:
		table.add_column("Index", style="green", no_wrap=True)
		table.add_column("Value", style="magenta")
		for i, v in enumerate(setting):
			table.add_row(str(i), str(v))

	_CONSOLE.print(table)

