# AlphaForge 方法与本作业落地说明

## 1. 论文方法要点

AlphaForge 的核心是两阶段框架：

1. **因子挖掘阶段（Factor Mining）**
   - 用生成-预测结构在公式空间中找高质量因子。
   - 目标不仅是高 IC，还要保持多样性、低相关性。
2. **动态组合阶段（Dynamic Combining）**
   - 每个交易日用最近窗口重新评估因子表现（IC、ICIR、RankIC）。
   - 通过阈值与排序动态选因子，再用线性模型动态配权，形成 Mega-Alpha。

论文强调：固定权重在风格切换时会失效，动态选因子与动态权重更稳健。

## 2. 本作业实现策略（AlphaForge-Lite）

受课程作业时限与可复现要求约束，本项目采用 AlphaForge-Lite：

- 不复现深度生成网络（G/P 神经网络），改为**模板化公式因子库**；
- 完整保留 AlphaForge 的关键思想：
  - 因子库质量与低相关控制；
  - 因子时变有效性评估；
  - 动态选因子与动态线性配权。

## 3. 与代码模块的对应关系

### 3.1 因子挖掘（静态阶段）

- 文件：`src/factor_library.py`
- 实现内容：
  - 动量/反转：`mom_w`, `rev_w`
  - 波动：`vol_w`
  - 量价关系：`volume_mom_w`, `volume_to_maw`, `price_volume_corr_w`, `price_volume_cov_w`
  - 衍生项：`price_volume_trend_20`, `price_volume_trend_40`
  - 每日横截面 z-score 标准化

### 3.2 因子评估与筛选

- 文件：`src/evaluation.py`
- 实现内容：
  - 单因子收益与 Sharpe（无交易成本）
  - 每日 IC / RankIC
  - ICIR / RankICIR
  - 相关性矩阵与贪心低相关筛选（最大相关阈值 0.5）

### 3.3 动态组合（时变阶段）

- 文件：`src/evaluation.py`
- 实现函数：`run_dynamic_mega_alpha`
- 机制：
  1. 用历史滚动窗口计算每个因子的 rolling IC、rolling ICIR。
  2. 在交易日 `t`，使用 `t-1` 的指标进行筛选，避免前视偏差。
  3. 使用过去 `train_window` 天样本拟合 ridge 线性权重。
  4. 在当日横截面生成组合信号并计算组合收益。

## 4. 为什么这个版本适合作业提交

1. **满足硬性指标**：相关性约束、平均 Sharpe、可复现代码。
2. **方法有论文映射**：可清晰说明“借鉴 AlphaForge 的两阶段思想”。
3. **解释性强**：动态线性权重可解释，适合作业答辩与报告写作。
4. **工程闭环**：一键执行、自动取数、输出完整报告文件。

## 5. 当前输出文件说明

运行 `run_asg_pipeline.py` 后，`reports/` 中包含：

- `factor_values.csv`：因子面板
- `factor_sharpe.csv`：单因子绩效与信息系数统计
- `factor_corr.csv`：因子相关性矩阵
- `selected_factors.txt`：相关性阈值筛选后的因子
- `dynamic_returns.csv`：动态组合日收益
- `dynamic_weights.csv`：动态组合权重
- `dynamic_ic_icir_panel.csv`：滚动指标面板
- `summary.md`：核心结果摘要

## 6. 风险与后续可优化项

- 本版本未计交易成本，和作业要求一致，但实盘收益会高估。
- 使用的是价格成交量基础字段，若补充基本面/另类数据可进一步提升。
- 可加 walk-forward 年度重训以进一步贴近实盘流程。
