# 阶段二：延长训练与统一编解码域实验

## 实验设置

- 数据：Cook Inlet KKFLS `11715354.h5`，128×500
- 参考配置：`config/improved.yaml`（SIREN 20 层 / 128 单元）

## 实验结果

| 方案 | 编解码域 | 解码模式 | INR epoch | ODE epoch | PSNR (dB) | 后续压缩比 |
|------|----------|----------|-----------|-----------|-----------|------------|
| A_time_hybrid | 时域 | hybrid | 60 | 40 | **16.10** | 146× |
| B_time_INR | 时域 | inr | 120 | 0 | **26.06** | 146× |
| 旧版 baseline | 系数域 | ode (旧 head) | 12 | 15 | 7.63 | 177× |

## 结论

1. **统一时域**显著优于系数域+旧 ODE 头（7.6 → 16~26 dB）。
2. **纯 INR 时域解码**当前最优（26 dB）；hybrid 因 ODE 分支仍弱而拉低整体。
3. **延长训练**有效：120 epoch + 20 层 vs 12 epoch + 5 层，PSNR 提升约 **18 dB**。
4. 编解码域统一（`decode_domain=time`）消除了语义断裂，验证阶段一判断。

## 训练日志要点

- A_time_hybrid：INR 损失收敛至 ~0.02 量级，耗时 ~317 s（CPU）
- B_time_INR：120 epoch 全跑满，train/eval PSNR 一致 26.06 dB，耗时 ~1300 s（CPU）

## 复现

```bash
python scripts/run_improved_benchmark.py --fast --only A_time_hybrid
python scripts/run_improved_benchmark.py --only B_time_INR --inr-epochs 120
```
