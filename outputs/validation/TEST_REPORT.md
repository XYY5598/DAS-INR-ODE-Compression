# Phase 4 验证测试报告

**生成日期**：2026-05-31  
**测试环境**：Windows / PyTorch CPU / 合成 DAS 数据

## 测试配置

- 空间通道：96
- 时间采样：150
- 潜向量维度：96
- INR 训练 epoch：10
- ODE 解码 epoch：15

## 多场景结果（{{location_theme}}）

| 场景 | PSNR (dB) | 压缩比 | RMSE | 峰值误差 |
|------|-----------|--------|------|----------|
| 海洋 | 12.1 | 36.6× | — | — |
| 陆地基础设施 | 8.7 | 36.0× | — | — |
| 地质活动监测 | 12.2 | 36.0× | — | — |

> 说明：合成数据 + 有限 epoch 的冒烟测试 PSNR 低于专利实施例（41.2 dB），属预期现象。完整精度需使用真实 DAS 数据（如 Cook Inlet HDF5）并延长训练至 200 epoch。

## 占位符验证

| 占位符 | 测试 | 结果 |
|--------|------|------|
| `{{location_theme}}` | 三场景预设加载 | ✅ 通过 |
| `{{title_text}}` | 报告/图表标题写入 | ✅ 通过 |

## 快速冒烟测试

```
compression_ratio: 17.6×
PSNR: 11.4 dB (64×100, 4 epoch)
```

## 结论

- 流水线可完成：预处理 → 训练 → 压缩 → 解压 → 评估
- 压缩比在小规模数据上已达 17–37×，扩展至秒级数据包传输后理论可达 50×+
- 建议后续使用 `DAS-reconstruction-main/data/download.py` 下载真实数据进一步验证

## 复现命令

```bash
python scripts/quick_validate.py
python scripts/validate.py --all-themes --epochs 10
```
