"""
gen_once.py
===========
gen.py 的单次推理变体：每条样本只生成一次（repeat=1）。

仅覆盖 parse_args 中 repeat 的值，其余逻辑与 gen.py 完全一致。
"""

import gen

_original_parse_args = gen.parse_args


def _parse_args_once():
    args = _original_parse_args()
    args.repeat = 1
    return args


# 替换为固定 repeat=1 的版本
gen.parse_args = _parse_args_once

if __name__ == "__main__":
    gen.main()
