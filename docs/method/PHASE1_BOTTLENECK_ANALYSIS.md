# 阶段一：技术瓶颈分析报告

## 1. 当前重建质量不足的主要技术点

### 1.1 ODE 时域动态重建失效（核心瓶颈）

**现象**（见 `03_comparison.png`、`04_waveform_channel.png`）：
- 重建场呈水平条纹，重建波形近似常数；
- 散点图呈 y≈0 水平带，与 y=x 无关。

**根因**（代码级）：
- 旧版 `SignalDecoderHead` 仅实现 `Linear(hidden_dim → n_channels)`，输入仅为 `h(t)`；
- 输出结构为 **u(c,t) ≈ W[c] · φ(h(t))**，为秩-1 时空分离模型；
- 无法表达斜线状地震波场所需的 **(c,t) 耦合**。

### 1.2 编解码域不一致

- INR 在 **预处理系数域** (C × n_coeff) 训练；
- ODE 在 **原始时域** (C × T) 上拟合 MSE；
- 两域无逆变换衔接，潜向量 z 的语义在解码端丢失。

### 1.3 训练配置不足

| 项目 | 旧配置 | 参考/专利 |
|------|--------|-----------|
| INR 层数 | 4–5 | 20（SIREN 参考） |
| INR epoch | 12 | 200 |
| ODE epoch | 15 | 100+ |
| 潜向量 z | 随机 batch，非联合优化 | 应联合训练 + 精调 |
| 损失 | 全局 MSE | 应含 3–10 Hz 加权 |

### 1.4 压缩比统计口径偏乐观

- 仅统计潜向量包（~1.4 KB），未计入共享 INR/ODE 权重（~500 KB）；
- 首任务部署压缩比被高估。

---

## 2. 架构配置不足评估清单

1. **解码器表达能力**：缺少 (h, z, x, t) 四元输入 → **已改进为 SpatioTemporalDecoderHead**
2. **域对齐**：系数域/时域混训 → **新增 decode_domain=time 统一方案**
3. **训练周期**：epoch 不足 → **improved.yaml: 200/120**
4. **损失函数**：无频段先验 → **新增 band_weighted_mse + seismic_weight**
5. **解码模式单一**：仅 ODE → **新增 inr / hybrid 模式**
6. **部署指标缺失**：无全链路压缩比 → **新增 deployment_metrics**

---

## 3. 至少三项影响重建性能的关键因素

1. **解码器设计**：秩-1 结构导致时间动态丢失（影响最大）
2. **训练周期与网络深度**：浅层+少 epoch 无法拟合 128×500 高维场
3. **编解码域不统一**：系数域编码、时域解码造成语义断裂

---

## 4. 改进路线（对应阶段二–四）

| 改进项 | 实现文件 |
|--------|----------|
| SpatioTemporalDecoderHead | `neural_ode_decoder.py` |
| 统一时域 + hybrid 解码 | `pipeline_v2.py` |
| 联合优化 z | `LearnableLatentTrainer` |
| 分频段损失 | `losses.py` |
| 部署级压缩比 | `deployment_metrics.py` |
| 基准实验 | `scripts/run_improved_benchmark.py` |
