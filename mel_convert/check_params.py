"""
检查 BigVGAN 模型参数配置
用于验证 nvidia/bigvgan_v2_22khz_80band_256x 的参数设置
"""
import torch
import sys
sys.path.append('/home/ruixin/Dubbing/mel_convert')

def check_bigvgan_params():
    """检查 BigVGAN 模型配置参数"""
    try:
        import bigvgan
    except ImportError:
        print("❌ 请先安装 BigVGAN: pip install bigvgan")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    
    model_id = "nvidia/bigvgan_v2_22khz_80band_256x"
    print(f"加载模型: {model_id}")
    
    # 加载模型
    model = bigvgan.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False)
    model = model.to(device)
    
    print("\n" + "="*60)
    print("模型配置参数:")
    print("="*60)
    
    # 从模型配置中获取参数
    h = model.h
    
    params = {
        "sampling_rate": h.sampling_rate,
        "n_fft": h.n_fft,
        "hop_size": h.hop_size,
        "win_size": h.win_size,
        "num_mels": h.num_mels,
        "fmin": h.fmin,
        "fmax": h.fmax,
    }
    
    # 期望值 (根据 Hugging Face 文档)
    expected = {
        "sampling_rate": 22050,
        "n_fft": 1024,
        "hop_size": 256,
        "win_size": 1024,
        "num_mels": 80,
        "fmin": 0,
        "fmax": 11025,
    }
    
    print(f"{'参数名':<20} {'实际值':<15} {'期望值':<15} {'状态'}")
    print("-"*60)
    
    all_match = True
    for key in params:
        actual = params[key]
        expect = expected.get(key, "N/A")
        match = "✅" if actual == expect else "❌"
        if actual != expect:
            all_match = False
        print(f"{key:<20} {actual:<15} {expect:<15} {match}")
    
    print("="*60)
    
    if all_match:
        print("\n✅ 所有参数匹配正确!")
    else:
        print("\n⚠️  警告: 部分参数与期望值不符")
    
    print("\n其他重要配置:")
    print("-"*60)
    if hasattr(h, 'resblock'):
        print(f"resblock: {h.resblock}")
    if hasattr(h, 'upsample_rates'):
        print(f"upsample_rates: {h.upsample_rates}")
        upsampling_ratio = 1
        for rate in h.upsample_rates:
            upsampling_ratio *= rate
        print(f"总上采样倍率: {upsampling_ratio}x")
    if hasattr(h, 'upsample_kernel_sizes'):
        print(f"upsample_kernel_sizes: {h.upsample_kernel_sizes}")
    if hasattr(h, 'upsample_initial_channel'):
        print(f"upsample_initial_channel: {h.upsample_initial_channel}")
    
    print("\n推荐的 Mel 提取配置 (用于 torchaudio.transforms.MelSpectrogram):")
    print("-"*60)
    print(f"sample_rate={params['sampling_rate']},")
    print(f"n_fft={params['n_fft']},")
    print(f"win_length={params['win_size']},")
    print(f"hop_length={params['hop_size']},")
    print(f"n_mels={params['num_mels']},")
    print(f"f_min={params['fmin']},")
    print(f"f_max={params['fmax']},")
    print(f"power=2.0,  # 官方使用 magnitude (即 power=2 后开根号)")
    print(f"normalized=False,")
    print(f"center=True  # 官方使用 center padding")

if __name__ == "__main__":
    check_bigvgan_params()
