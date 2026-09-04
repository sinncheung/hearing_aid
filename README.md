# Adaptive Hearing Aid V0.1

这是一个 PC/Python 原型，用来验证“自适应环境降噪 + 语音保护”的核心 DSP。

## 功能
- WAV 输入/输出
- STFT
- 简单 VAD（语音活动检测）
- 噪声谱估计
- Wiener-style spectral suppression
- 自适应增益
- 频段 EQ
- 频谱/波形对比图

## 使用

安装依赖：

    pip install -r requirements.txt

处理 WAV：

    python hearing_aid.py input.wav output.wav

如果没有输入文件，程序会自动生成一个“语音样信号 + 风扇样宽带噪声”的 demo：

    python hearing_aid.py

注意：这是工程原型，不是医疗器械，也不要把输出直接大音量送入耳朵。
