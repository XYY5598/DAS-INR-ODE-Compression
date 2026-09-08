# 阶段三：增强 ODE 解码器与分频段损失

## 3.1 SpatioTemporalDecoderHead 设计

**旧版问题**：`SignalDecoderHead(h(t)) → ℝ^C`，输出秩-1 时空分离。

**新版**：`f(h(t), z, x, t) → scalar`，在 `neural_ode_decoder.py` 中实现：

```
输入拼接: [h, z, x, t, sin(2πt), cos(2πt)]  → MLP(128) → 标量
重建: 向量化 (C×T) 网格批量前向
```

## 3.2 分频段加权损失

`das_inr_ode/losses.py`：

| 频段 | 范围 (Hz) | 默认权重 |
|------|-----------|----------|
| 地震预警 | 3–10 | 8.0 |
| 背景 | 0.1–3 | 1.0 |
| 噪声 | >10 | 0.5 |

损失 = 0.5×空域 MSE + 0.5×频域加权误差 + λ×PINN（λ=0.05）

## 3.3 实验观察

- 新 ODE 头 + 分频段损失：**消除水平条纹**，hybrid 模式 PSNR 7.6→16.1 dB
- ODE 单独仍弱于 INR（26 vs 16 dB），需更多 ODE epoch 或 hybrid_alpha↑（建议 0.8–0.9）
- 时频图预期改善：需对 B_time_INR 重绘验证（PSNR 26 dB 应保留更多 3–10 Hz 能量）

## 3.4 后续调参建议

```yaml
hybrid_alpha: 0.85        # 以 INR 为主
ode:
  epochs: 150
  seismic_weight: 10.0
```
