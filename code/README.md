# 基于隐式神经表示与 Neural ODE 的 DAS 数据压缩 — 代码包

实现位于本目录（`code/`）。仓库根目录见上一级 `README.md`。

## 安装与运行

```bash
cd code
pip install -r requirements.txt

# 合成数据冒烟
python scripts/quick_validate.py

# 真实数据（默认 ../data/raw）
python scripts/validate_real_data.py --h5 ../data/raw/11715354_KKFLS.h5
```

输出默认写入仓库根 `../outputs/`。

## 包结构

| 路径 | 说明 |
|------|------|
| `das_inr_ode/` | 预处理、INR、ODE、损失、流水线、路径约定 |
| `config/` | YAML 配置 |
| `scripts/` | 训练 / 压缩 / 验证 / 专利报告 |

## 占位符

`{{location_theme}}`、`{{title_text}}` — 见 `../docs/method/VARIABLES.md`。

## 参考

- 专利与方法文档：`../docs/`
- SIREN/RFFN 精简参考：`../reference/DAS-reconstruction-lite/`
