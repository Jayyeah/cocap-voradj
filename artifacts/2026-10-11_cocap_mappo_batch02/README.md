# CoCap MAPPO migration：第一阶段证据

此目录只包含既有Batch01/IQN证据恢复、源码diff、离线参考transition及准备DAG。没有新训练、selection/final、checkpoint promotion或生产E1 adapter。`E1_pass=false`、`independent_validation=false`，等待下一步授权。

- `batch01_audit.json`：六个PROVISIONAL_LONG末点、manifest/source/checkpoint核对，best-observed与last分开，训练首尾窗口。
- `diffs/`：六ARM相对V3r2 candidate的真实源码/config diff；成功IQN与Core token evaluator汇总差异另存。
- `training_curves.csv`、`screen_curves.csv`、两张PNG：既有日志分析，同step重复记录取最后一条用于绘图，原日志/回滚记录保留。
- `iqn_reference_inventory.json`：成功IQN原报告/hash、配置extends链、已存在checkpoint hash、loader AST一致性和评估工具差异。
- `iqn_z05_stage1_success_report.json`：成功Stage1独立报告按字节复制；没有重跑IQN评估。
- `iqn_z05_reference_resolved.json`：hash与成功Stage1原报告一致的resolved合同。
- `reference_sources/`：成功IQN评估源码只读归档，不作为新训练入口。
- `reference_transitions.json`：8个相同物理state的TERL/当前MAPPO/CoCap参考fixture，包含输入、观测、reward、done及next state。
- `dag_before.json`：本轮修改之前的6个Batch02节点，历史依赖保留。
- `validation.json`：交付数据/文件一致性检查，属于Commander工程检查，不能代替独立验证。

在中央worktree重建审计：

```bash
python tools/batch02_migration_audit.py
```

在相同固定源码下重放参考fixture：

```bash
PYTHONPATH=/home/yjq/rl/CoCap1/batch01-t1-long-v3r2-20261010/src:/home/yjq/rl/CoCap1/batch01-t1-long-v3r2-20261010:/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/local/anaconda3/bin/python tools/batch02_reference_transitions.py
```

参考fixture借用已经完成的T1 Stage2原生4P1E1O接口，与成功IQN Stage1同规模；不修改T1训练配置。指定fixture清空current仅是显式诊断输入，不登记新ARM。比较固定P/E命令时APF动作单独记录；native↔现MAPPO parity使用真实native APF。checkpoint权重仍在原服务器路径，Git只归档证据与hash。
