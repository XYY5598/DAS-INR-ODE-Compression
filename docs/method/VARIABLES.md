# 可替换变量说明

## {{location_theme}}

**含义**：分布式光纤应变监测的应用场景主题，用于自动选择算法预设。

**可选值**（定义于 `config/default.yaml` → `scenarios`）：

| 值 | 说明 | 自动调整项 |
|----|------|------------|
| `海洋` | 海底地震/海啸监测 | SIREN 激活、弹性波 PINN、严格地震频段量化 |
| `陆地基础设施` | 桥梁/隧道健康监测 | ReLU 激活、热扩散 PINN、准静态优化 |
| `地质活动监测` | 火山/断层长期监测 | SIREN、双物理场 PINN、三级异常触发 |

**替换示例**：

```bash
python scripts/train.py --location-theme 地质活动监测
```

```yaml
# config/default.yaml
location_theme: "陆地基础设施"
```

```python
cfg = load_config(location_theme="海洋")
```

---

## {{title_text}}

**含义**：可视化输出与 JSON 报告中的标题字符串。

**用途位置**：
- `plot_strain_field()` / `plot_comparison()` 图表标题
- 压缩数据包 `metadata.title_text`
- 验证报告 `validation_report.json`

**替换示例**：

```bash
python scripts/decompress.py --packet outputs/packet.bin --title-text "2024-06-01 14:30:00 海底DAS监测"
```

```yaml
title_text: "FORGE 地热区 DAS 应变场"
```

**嵌套替换**：配置文件中可写 `"监测报告 — {{location_theme}}"`，加载时由 `apply_placeholders()` 一并替换。

---

## 其他配置变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `latent_dim` | 128 | 潜向量维度，可由光纤长度自动推算 |
| `fiber_length_km` | 10 | 用于 `d_z = α·L` 公式 |
| `ode.pinn_weight` | 0.5 | PINN 物理约束权重 |
| `monitoring.anomaly_energy_factor` | 5.0 | 异常能量触发倍数 |
