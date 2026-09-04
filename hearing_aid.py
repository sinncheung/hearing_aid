import sys
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, butter, sosfilt
import matplotlib.pyplot as plt


SR_DEFAULT = 16000
N_FFT = 512
HOP = 128
NOISE_UPDATE = 0.03


def rms_db(x, eps=1e-10):
    return 20 * np.log10(np.sqrt(np.mean(x * x) + eps))


def make_demo(sr=SR_DEFAULT, seconds=6):
    """生成可重复测试的 demo：语音样调制信号 + 风扇/空调样噪声。"""
    rng = np.random.default_rng(7)
    t = np.arange(int(sr * seconds)) / sr

    # 用多个谐波 + 缓慢包络模拟“人声样”信号，而不是伪造真实语音。
    f0 = 125 + 12 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    speech = (
        0.55 * np.sin(phase)
        + 0.20 * np.sin(2 * phase)
        + 0.10 * np.sin(3 * phase)
    )
    envelope = 0.15 + 0.85 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.2 * t)) ** 2
    speech *= envelope

    # 宽带噪声 + 低频电机/风扇成分
    white = rng.normal(0, 1, len(t))
    sos = butter(2, 3500, btype="lowpass", fs=sr, output="sos")
    broadband = sosfilt(sos, white)
    fan = 0.22 * np.sin(2 * np.pi * 120 * t) + 0.12 * np.sin(2 * np.pi * 240 * t)
    noise = 0.65 * broadband / (np.std(broadband) + 1e-9) + fan

    # 中间一段提高噪声，模拟环境变化
    noise[2 * sr:4 * sr] *= 1.8
    return (speech + 0.75 * noise).astype(np.float32), speech.astype(np.float32)


def vad_from_spectrum(Zxx, sr):
    """非常简单的频谱 VAD：估计 300 Hz–4 kHz 的能量。"""
    power = np.abs(Zxx) ** 2
    freqs = np.linspace(0, sr / 2, power.shape[0])
    band = (freqs >= 300) & (freqs <= 4000)
    energy = np.mean(power[band], axis=0)
    # 自适应阈值；这里只用于 prototype。
    threshold = np.percentile(energy, 35) * 2.2 + 1e-12
    return energy > threshold, energy


def adaptive_wiener(Zxx, speech_mask):
    """Wiener-style suppression。

    噪声谱来自 VAD 判定为“非语音”的帧，并用最小统计量保持稳定。
    """
    power = np.abs(Zxx) ** 2
    T = power.shape[1]

    noise_frames = np.where(~speech_mask)[0]
    if len(noise_frames) < 3:
        noise_power = np.percentile(power, 25, axis=1, keepdims=True)
        noise_power = np.repeat(noise_power, T, axis=1)
    else:
        noise_power = np.median(power[:, noise_frames], axis=1, keepdims=True)
        noise_power = np.repeat(noise_power, T, axis=1)

    # 平滑噪声估计，避免参数跳变。
    alpha = 0.90
    for k in range(1, T):
        if not speech_mask[k]:
            noise_power[:, k] = alpha * noise_power[:, k - 1] + (1 - alpha) * noise_power[:, k]

    snr = np.maximum((power - noise_power) / (noise_power + 1e-10), 0.0)

    # Wiener-like gain，限制最低增益，避免音乐噪声/“水下感”。
    gain = snr / (snr + 1.0)
    gain = np.clip(gain, 0.18, 1.0)

    # 语音帧保护：不要过度压低疑似语音。
    gain[:, speech_mask] = np.maximum(gain[:, speech_mask], 0.55)

    return Zxx * gain, gain


def hearing_eq(Zxx, sr):
    """非常温和的高频补偿示例；真实 hearing profile 应由听力学测试决定。"""
    freqs = np.linspace(0, sr / 2, Zxx.shape[0])
    eq_db = np.zeros_like(freqs)

    # 从 1.5 kHz 开始轻微提升，最高 +6 dB。
    hi = freqs >= 1500
    eq_db[hi] = np.minimum(6.0, (freqs[hi] - 1500) / 1500 * 3.0)

    gain = 10 ** (eq_db / 20)
    return Zxx * gain[:, None]


def process(x, sr):
    x = x.astype(np.float64)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.8

    f, tt, Z = stft(
        x, fs=sr, window="hann",
        nperseg=N_FFT, noverlap=N_FFT - HOP,
        boundary="zeros"
    )

    speech_mask, band_energy = vad_from_spectrum(Z, sr)
    Z_nr, gain = adaptive_wiener(Z, speech_mask)
    Z_out = hearing_eq(Z_nr, sr)

    _, y = istft(
        Z_out, fs=sr, window="hann",
        nperseg=N_FFT, noverlap=N_FFT - HOP,
        input_onesided=True
    )

    y = y[:len(x)]
    y /= max(np.max(np.abs(y)), 1e-9)
    y *= 0.75
    return y.astype(np.float32), speech_mask, gain, tt, f


def plot_results(x, y, sr, speech_mask, gain, tt, f):
    fig = plt.figure(figsize=(11, 8))

    ax1 = fig.add_subplot(3, 1, 1)
    tx = np.arange(len(x)) / sr
    ax1.plot(tx, x)
    ax1.set_title("Input waveform")
    ax1.set_xlabel("Time (s)")

    ax2 = fig.add_subplot(3, 1, 2)
    ty = np.arange(len(y)) / sr
    ax2.plot(ty, y)
    ax2.set_title("Enhanced waveform")
    ax2.set_xlabel("Time (s)")

    ax3 = fig.add_subplot(3, 1, 3)
    im = ax3.pcolormesh(tt, f, 20 * np.log10(np.maximum(gain, 1e-4)), shading="auto")
    ax3.set_ylim(0, 8000)
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Applied suppression gain (dB)")
    fig.colorbar(im, ax=ax3)

    fig.tight_layout()
    plt.savefig("hearing_aid_v0_1_result.png", dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
        x, sr = sf.read(input_path)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        print(f"Loaded: {input_path}, {sr} Hz, {len(x)/sr:.2f} s")
        clean_reference = None
    else:
        print("No input WAV supplied; generating demo signal.")
        x, clean_reference = make_demo()
        sr = SR_DEFAULT
        sf.write("demo_input.wav", x, sr)

    print(f"Input RMS: {rms_db(x):.1f} dBFS")
    y, speech_mask, gain, tt, f = process(x, sr)
    print(f"Output RMS: {rms_db(y):.1f} dBFS")
    print(f"Detected speech-like frames: {100*np.mean(speech_mask):.1f}%")

    output_path = sys.argv[2] if len(sys.argv) >= 3 else "demo_enhanced.wav"
    sf.write(output_path, y, sr)
    plot_results(x, y, sr, speech_mask, gain, tt, f)

    print(f"Written: {output_path}")
    print("Written: hearing_aid_v0_1_result.png")


if __name__ == "__main__":
    main()
