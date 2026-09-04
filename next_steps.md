# V0.2 / V0.3 路线

## V0.2
1. 改进 VAD：WebRTC VAD 或能量 + 谱特征
2. 使用 minimum statistics noise estimator
3. 加入 attack/release smoothing
4. 加入 loudness normalization
5. 加入可配置 hearing profile
6. 做客观指标：SNR、segmental SNR、STOI（如环境允许）

## V0.3
1. 双通道输入
2. GCC-PHAT / TDOA 方向估计
3. Delay-and-sum beamformer
4. 左右耳独立增益
5. 场景分类器：quiet / speech / traffic / cafe / wind
6. 根据场景自动选择 DSP profile

## V1 hardware
PC 原型验证后再迁移到 STM32H7 + audio codec。
