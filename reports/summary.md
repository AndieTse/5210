# 作业结果摘要

- 因子总数：34
- 全因子平均 Sharpe（不含交易成本）：0.218419
- 相关性约束后入选因子数（`corr <= 0.5`）：10
- 入选因子集合最大绝对相关系数：0.4719407534994429
- 静态 Mega-Alpha Sharpe（等权）：0.488276
- 动态 Mega-Alpha Sharpe（滚动 IC/ICIR + Ridge）：0.105498

## 数据来源与口径
- 数据文件：`D:\Users\xiey7\SQL\5210\data\market_data.csv`
- 数据来源：Yahoo Finance，通过 `yfinance` 自动下载
- 访问日期：2026-04-24
- 原始字段：复权收盘价（`auto_adjust=True`）与成交量（日频）
- 标的范围（20只）：
  - `AAPL, AMZN, BAC, CSCO, CVX, DIS, GOOGL, IBM, INTC, JNJ, JPM, KO, MCD, META, MSFT, NVDA, PEP, PG, WMT, XOM`
- 文件覆盖区间：2023-04-24 至 2026-04-23（近三年）
- 总行数：15060

## 回测说明
- 因子层指标（IC/ICIR/Sharpe）用于验证信号有效性。
- 组合层回测用于验证可交易性与风险收益特征。
- 已输出组合层指标：年化收益、年化波动、最大回撤、胜率（见 `backtest_metrics.csv`）。

## 思考与优化
- 已完成动态策略参数敏感性实验（TopN / 窗口 / ridge 正则）。
- 本次样本内最佳动态配置：TopN=10，窗口=60，lambda=1.0，Sharpe=0.10549820319138618。

## AlphaForge-Lite 两阶段
- 阶段1：构建因子库并执行低相关筛选。
- 阶段2：按滚动 IC/ICIR 动态筛选因子并线性配权生成 Mega-Alpha。