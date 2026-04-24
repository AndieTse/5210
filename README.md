# MFE5210 Alpha Factors 作业工程

本项目实现 AlphaForge-Lite 两阶段流程：先构建低相关因子库，再做动态因子选择与线性组合，并输出完整评估报告。

## 目录说明

- `run_asg_pipeline.py`：主入口，一键执行完整流程。
- `data/market_data.csv`：输入数据文件。
- `src/factor_library.py`：因子构建。
- `src/evaluation.py`：IC/ICIR、Sharpe、组合回测、参数敏感性。
- `reports/`：运行产物。
- `references.md`：参考文献。

## 数据来源（明确口径）

本仓库当前 `data/market_data.csv` 的来源与口径如下：

- 数据平台：Yahoo Finance（网页源）。
- 采集接口：Python `yfinance`。
- 接口调用方式：`yf.download(..., interval='1d', auto_adjust=True, group_by='ticker')`。
- 字段映射：
  - `date`：交易日日期
  - `asset`：股票代码
  - `close`：复权收盘价（`auto_adjust=True`）
  - `volume`：成交量
- 标的范围（20只）：
  - `AAPL, AMZN, BAC, CSCO, CVX, DIS, GOOGL, IBM, INTC, JNJ, JPM, KO, MCD, META, MSFT, NVDA, PEP, PG, WMT, XOM`
- 文件当前覆盖区间（以现有文件为准）：
  - 起始日期：`2023-04-24`
  - 结束日期：`2026-04-23`
  - 总行数：`15060`
  - 资产数：`20`
  - 生成日期：`2026-04-24`


## 主要输出（`reports/`）
reports/summary.md
- `summary.md`：总览结果与关键配置
- `factor_sharpe.csv`：单因子 Sharpe/IC/ICIR
- `factor_corr.csv`：因子相关性矩阵
- `selected_factors.txt`：满足相关性阈值后的因子
- `dynamic_returns.csv`：动态组合日收益
- `backtest_metrics.csv`：年化收益/波动/回撤/胜率
- `dynamic_sensitivity.csv`：参数敏感性实验
- `project_reflection.md`：方法思考、优化动作、局限性

## 方法概述

- 阶段1：构建因子库并执行低相关筛选（阈值 `max |corr| <= 0.5`）。
- 阶段2：基于滚动 IC/ICIR 做动态选因子，并用线性 ridge 进行日度组合配权。
