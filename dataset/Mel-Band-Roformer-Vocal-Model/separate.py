"""
separate.py — 通用音频人声/背景音分离脚本

用法:
    python separate.py --ost /path/to/audio/ost

输入: ost/ 文件夹（存放原始混合音频，.wav 格式）
输出:
    audio/vocals/  — 分离出的人声
    audio/ins/     — 分离出的背景音（伴奏）

两个输出文件夹与 ost/ 位于同级目录。
"""

import os
import shutil
import argparse
import subprocess

import soundfile as sf
import librosa


# ── 默认路径（可通过命令行参数覆盖）────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = "/data2/ruixin/downloads/melbandroformer/MelBandRoformer.ckpt"
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", "config_vocals_mel_band_roformer.yaml")
DEFAULT_INFERENCE_SCRIPT = os.path.join(SCRIPT_DIR, "inference.py")


def convert_to_mono(src_path: str, dst_path: str) -> bool:
    """将多声道音频转为单声道并保存，返回是否成功。"""
    try:
        y, sr = librosa.load(src_path, sr=None, mono=True)
        sf.write(dst_path, y, sr)
        return True
    except Exception as e:
        print(f"  [警告] 单声道转换失败 {src_path}: {e}")
        return False


def calculate_energy_db(audio_path: str) -> float:
    """
    计算音频文件的能量（单位：dB）。
    返回能量值（dB），如果出错则返回 None。
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if len(y) == 0:
            return float('-inf')
        # 计算 RMS 能量
        rms = librosa.feature.rms(y=y)[0]
        # 将 RMS 转换为 dB，然后取平均值
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        mean_rms_db = rms_db.mean()
        return mean_rms_db
    except Exception as e:
        print(f"  [警告] 计算能量失败 {audio_path}: {e}")
        return None


def preprocess_mono(input_folder: str) -> int:
    """将 input_folder 内所有多声道 .wav 就地转换为单声道，返回转换数量。"""
    converted = 0
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".wav"):
            continue
        fpath = os.path.join(input_folder, fname)
        try:
            info = sf.info(fpath)
            if info.channels > 1:
                tmp = fpath + ".mono.tmp.wav"
                if convert_to_mono(fpath, tmp):
                    shutil.move(tmp, fpath)
                    converted += 1
                    print(f"  已转换为单声道: {fname}")
        except Exception as e:
            print(f"  [警告] 无法读取 {fname}: {e}")
    return converted


def run_inference(
    input_folder: str,
    store_dir: str,
    model_path: str,
    config_path: str,
    inference_script: str,
    device_ids: list,
) -> None:
    """调用 inference.py 进行人声分离。"""
    cmd = [
        "python", inference_script,
        "--config_path", config_path,
        "--model_path", model_path,
        "--input_folder", input_folder,
        "--store_dir", store_dir,
    ]
    if device_ids:
        cmd += ["--device_ids"] + [str(d) for d in device_ids]

    print(f"\n[推理] 执行命令: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def postprocess(store_dir: str, ins_dir: str, energy_threshold_db: float = -50.0) -> tuple[int, int, int]:
    """
    inference.py 将 *_vocals.wav 和 *_instrumental.wav 都输出到 store_dir (vocals/)。
    本函数将 *_instrumental.wav 移动到 ins_dir，并根据能量阈值过滤低能量文件。
    返回 (vocals_count, ins_count, filtered_count)。
    """
    os.makedirs(ins_dir, exist_ok=True)
    vocals_count = 0
    ins_count = 0
    filtered_count = 0

    # 收集所有 vocals 文件并计算能量
    vocals_files = {}
    for fname in os.listdir(store_dir):
        if fname.endswith("_vocals.wav"):
            vocals_files[fname] = os.path.join(store_dir, fname)

    # 对每个 vocals 文件进行能量检测
    for vocals_fname, vocals_path in vocals_files.items():
        # 计算 vocals 能量
        vocals_energy = calculate_energy_db(vocals_path)

        if vocals_energy is None:
            print(f"  [警告] 无法计算能量，跳过: {vocals_fname}")
            filtered_count += 1
            continue

        # 过滤低能量样本
        if vocals_energy < energy_threshold_db:
            print(f"  [过滤] {vocals_fname}: 能量={vocals_energy:.2f}dB (阈值={energy_threshold_db}dB)")
            # 删除 vocals 文件
            if os.path.exists(vocals_path):
                os.remove(vocals_path)
            # 删除对应的 instrumental 文件
            base_name = vocals_fname.replace("_vocals.wav", "")
            instrumental_fname = f"{base_name}_instrumental.wav"
            instrumental_path = os.path.join(store_dir, instrumental_fname)
            if os.path.exists(instrumental_path):
                os.remove(instrumental_path)
            filtered_count += 1
            continue

        vocals_count += 1

    # 移动剩余的 instrumental 文件
    for fname in os.listdir(store_dir):
        if fname.endswith("_instrumental.wav"):
            src = os.path.join(store_dir, fname)
            dst = os.path.join(ins_dir, fname)
            shutil.move(src, dst)
            ins_count += 1

    return vocals_count, ins_count, filtered_count


def move_txt_files(src_dir: str, dst_dir: str) -> int:
    """将 src_dir 中所有 .txt 文件移动到 dst_dir，返回移动数量。"""
    txt_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".txt")]
    if not txt_files:
        print(f"[警告] ost 文件夹中没有 .txt 文件: {src_dir}")
        return 0

    moved_count = 0
    for fname in txt_files:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        try:
            shutil.move(src, dst)
            moved_count += 1
            print(f"  已移动文本文件: {fname} -> {dst_dir}")
        except Exception as e:
            print(f"  [警告] 文本文件移动失败 {fname}: {e}")

    print(f"[信息] 共移动 {moved_count} 个 .txt 文件到 vocals 文件夹。")
    return moved_count


def main():
    parser = argparse.ArgumentParser(
        description="使用 Mel-Band-RoFormer 分离人声与背景音"
    )
    parser.add_argument(
        "--ost", type=str, required=True,
        help="原始混合音频文件夹路径（例如 audio/ost）"
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help=f"模型权重路径（默认: {DEFAULT_MODEL_PATH}）"
    )
    parser.add_argument(
        "--config_path", type=str, default=DEFAULT_CONFIG_PATH,
        help=f"模型配置文件路径（默认: {DEFAULT_CONFIG_PATH}）"
    )
    parser.add_argument(
        "--inference_script", type=str, default=DEFAULT_INFERENCE_SCRIPT,
        help=f"inference.py 的路径（默认: {DEFAULT_INFERENCE_SCRIPT}）"
    )
    parser.add_argument(
        "--device_ids", nargs="+", type=int, default=None,
        help="使用的 GPU ID 列表，例如 --device_ids 0 1（默认由 inference.py 决定）"
    )
    parser.add_argument(
        "--skip_mono_convert", action="store_true",
        help="跳过单声道预处理步骤"
    )
    parser.add_argument(
        "--energy_threshold", type=float, default=-50.0,
        help="能量阈值（dB），低于此值的音频将被过滤（默认: -50.0）"
    )
    args = parser.parse_args()

    # ── 路径推导 ─────────────────────────────────────────────────────────────
    ost_dir = os.path.abspath(args.ost)
    if not os.path.isdir(ost_dir):
        print(f"[错误] ost 文件夹不存在: {ost_dir}")
        return

    audio_dir = os.path.dirname(ost_dir)   # ost 的父目录
    vocals_dir = os.path.join(audio_dir, "vocals")
    ins_dir = os.path.join(audio_dir, "ins")

    os.makedirs(vocals_dir, exist_ok=True)
    os.makedirs(ins_dir, exist_ok=True)

    print("=" * 60)
    print(f"  输入 (ost)  : {ost_dir}")
    print(f"  输出 (vocals): {vocals_dir}")
    print(f"  输出 (ins)  : {ins_dir}")
    print(f"  模型        : {args.model_path}")
    print(f"  配置        : {args.config_path}")
    print("=" * 60)

    # ── 步骤 0: 将 ost 内的 .txt 文件移动到 vocals/ ─────────────────────────
    print("\n[步骤 0] 检查并移动 ost 中的 .txt 文件...")
    move_txt_files(ost_dir, vocals_dir)

    # ── 检查 .wav 文件 ───────────────────────────────────────────────────────
    wav_files = [f for f in os.listdir(ost_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print(f"[错误] ost 文件夹中没有 .wav 文件: {ost_dir}")
        return
    print(f"\n共找到 {len(wav_files)} 个 .wav 文件待处理。")

    # ── 步骤 1: 单声道预处理 ─────────────────────────────────────────────────
    if not args.skip_mono_convert:
        print("\n[步骤 1] 单声道预处理...")
        converted = preprocess_mono(ost_dir)
        print(f"  完成，共转换 {converted} 个文件。")
    else:
        print("\n[步骤 1] 跳过单声道预处理。")

    # ── 步骤 2: 推理分离 ─────────────────────────────────────────────────────
    print("\n[步骤 2] 运行 Mel-Band-RoFormer 推理...")
    try:
        run_inference(
            input_folder=ost_dir,
            store_dir=vocals_dir,   # inference 先全部输出到 vocals/
            model_path=args.model_path,
            config_path=args.config_path,
            inference_script=args.inference_script,
            device_ids=args.device_ids or [],
        )
    except subprocess.CalledProcessError as e:
        print(f"[错误] 推理失败: {e}")
        return

    # ── 步骤 3: 后处理 — 将 instrumental 文件移入 ins/ ──────────────────────
    print("\n[步骤 3] 后处理：能量过滤与文件分类...")
    vocals_count, ins_count, filtered_count = postprocess(
        vocals_dir, ins_dir, energy_threshold_db=args.energy_threshold
    )

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  分离完成！")
    print(f"  过滤的低能量文件              : {filtered_count} 个 (阈值: {args.energy_threshold}dB)")
    print(f"  人声文件 (*_vocals.wav)       : {vocals_count} 个 → {vocals_dir}")
    print(f"  背景音文件 (*_instrumental.wav): {ins_count} 个 → {ins_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
