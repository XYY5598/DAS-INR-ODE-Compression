# 方法概述与算法设计

## 一、方法概述（Phase 1）

### 1.1 三阶段协同架构

```
原始 DAS 应变场 (C×T)
    ↓ 物理预处理
频域系数矩阵 + 元数据
    ↓ INR 编码器
潜向量 z ∈ R^{64..256} + 共享网络权重 θ
    ↓ 数据包传输
    ↓ Neural ODE 解码
连续时间重建应变场 (C×T)
```

### 1.2 核心步骤

| 步骤 | 模块 | 输入 | 输出 |
|------|------|------|------|
| 1 | 均值剥离 | 原始信号 | 去基线信号 + 均值元数据 |
| 2 | DWT/DCT 融合 | 去基线信号 | 能量集中系数 |
| 3 | LPC + 受控量化 | 系数 | 残差 + 量化步长 |
| 4 | INR 编码 | 坐标 + 系数 | 潜向量 z |
| 5 | 数据包封装 | z + 元数据 | 二进制压缩包 |
| 6 | ODE 积分 | z → h₀ → h(t) | 隐藏状态轨迹 |
| 7 | 信号重建 | h(t) | 时空应变场 |

### 1.3 可替换变量

| 占位符 | 用途 | 配置位置 |
|--------|------|----------|
| `{{location_theme}}` | 选择监测场景预设（海洋/陆地基础设施/地质活动监测），影响 PINN 类型、激活函数、劣化阈值 | `config/default.yaml`, CLI `--location-theme` |
| `{{title_text}}` | 可视化图表与 JSON 报告标题 | `config/default.yaml`, CLI `--title-text` |

### 1.4 关键超参数

- 潜向量维度：`d_z = clamp(α·L_km, 64, 256)`，α 随光纤长度分段
- ODE 求解器：dopri5（精度）/ euler（实时）
- PINN 权重 λ ∈ [0.1, 1.0]
- 异常能量阈值：基线能量 × (5~10)

---

## 二、算法伪代码（Phase 2）

### 2.1 物理预处理

```
function Preprocess(data[C,T], cfg):
    for each channel c:
        μ_c ← mean(data[c,:])
        data[c,:] ← data[c,:] - μ_c
    for each channel c:
        coeffs ← DWT(data[c,:], wavelet=db4, level=L)
        coeffs[0] ← DCT(coeffs[0])
    residuals ← LPC(coeff_matrix, order=P)   // Levinson-Durbin
    q_data ← ControlledQuantize(residuals, ε_band)
    return q_data, metadata(μ, scales, LPC)
```

### 2.2 INR 编码器

```
function INR_Encode(q_data, θ, d_z):
    coords ← grid(normalize(x), normalize(coeff_idx))
    train θ on (coords, q_data) with loss MSE + λ||z||²
    z ← argmin_z MSE(INR_θ(coords, z), q_data)
    return z
```

### 2.3 Neural ODE 解码

```
function ODE_Decode(z, T, cfg):
    h₀ ← Map(z)                          // 2-layer MLP, ≤10K params
    h(t) ← ODEIntegrate(f_θ, h₀, t∈[0,1])  // adjoint, O(1) memory
    recon[:,t] ← SignalHead(h(t))
    if PINN_Mode:
        loss += λ · PDE_Residual(recon)    // 弹性波/热扩散
    return recon
```

### 2.4 长时间监测控制

```
function Monitor(recon, state):
    align_time(packet.t_start, packet.clock_offset)
    if energy(recon) > k × baseline:       // k = {{anomaly_factor}}
        trigger ResetSignal / fast Euler mode
    every 10 min: save_snapshot(h(t))
```

### 2.5 {{location_theme}} 应用差异

| 场景 | 激活函数 | PDE 约束 | 劣化阈值策略 |
|------|----------|----------|--------------|
| 海洋 | SIREN | 弹性波方程 | 地震频段 ε=1e-6 |
| 陆地基础设施 | ReLU | 热扩散方程 | 准静态为主 |
| 地质活动监测 | SIREN | 双物理场 | 三级异常触发 |

---

## 三、模块映射

| 文档模块 | 代码文件 |
|----------|----------|
| 物理预处理 | `das_inr_ode/preprocessing.py` |
| INR 编码器 | `das_inr_ode/inr_encoder.py` |
| Neural ODE 解码 | `das_inr_ode/neural_ode_decoder.py` |
| 数据包协议 | `das_inr_ode/packet.py` |
| 监测控制 | `das_inr_ode/monitoring.py` |
| 端到端流水线 | `das_inr_ode/pipeline.py` |

参考代码 `DAS-reconstruction-main` 中 SIREN/RFFN 的网络结构与训练范式已对齐。
