# 全局分层基线开源审计 V4

日期：2026-10-17。**仅静态读源、版本和文件检查；未安装依赖、执行第三方仿真、加载权重或跑其测试。** 四类状态沿用统一 registry：`OFFICIAL_REPRODUCIBLE`、`OFFICIAL_PARTIAL`、`THIRD_PARTY`、`NOT_FOUND`。状态必须带范围；官方求解组件完备不等于完整围捕系统开源。

## 1. 完整方法代码结论

| 对象 | Repo / 可定位版本 | 许可证、入口与依赖 | 缺口与复现判断 |
|---|---|---|---|
| TR-11 JAS 2026 | `NOT_FOUND`；[论文/补充](https://www.ieee-jas.net/article/doi/10.1109/JAS.2025.125684?pageType=en#Supplements)，无可核 commit/tag | 原文 Mosek；代码语言、入口/配置、许可证未知 | 按正文/补充重建 A*、分组/拍卖、C3AVC、原生 LP；无公开测试/权重证据；完整高 |
| TR-13 CEO-DMPC | `NOT_FOUND`；[作者全文](https://april.zju.edu.cn/wp-content/papercite-data/pdf/chen2024adp.pdf)，无可核 commit/tag | 论文 Python3、Quadprog、OR-Tools `pywraplp`；Gazebo；代码许可证未知 | 分组、位置 MILP、预测通信/目标序列及 QP 均需重建；无可核公开测试/权重；分配低至中，完整高至很高 |
| TR-14 AFRA | **`OFFICIAL_PARTIAL`**；[官方 Gitee](https://gitee.com/lqt20/multi-uav)，`72440fdf0aa9516e87fcd52f282ce2a612c2df68`，2026-04-10；仅 master，未见 tag | 根目录未见 LICENSE/COPYING；无 requirements/pyproject/环境锁；依赖从 import 推定 NumPy、Pandas、Matplotlib、Seaborn 等 | 作者关联明确，但主路径与正文关键机制不同，见下文；不可直接作为完整可复现实现 |
| G-CAP / TR-15 | 项目设计，`NOT_FOUND`（完整实现尚未创建，不是负面检索结论） | 拟 Python + SciPy，可选 OR-Tools；完整实现许可/入口待建立 | 已有组件可复用，控制器/接口/释放为项目代码；中至高；无预训练权重需求 |

TR-01、TR-02、TR-06、TR-12 维持 V2 的无完整官方代码/全文缺口状态；本轮没有找到经核实的第三方完整实现。GitHub 精确关键词检索包含 Dynamic Robust Pursuit、CEO-DMPC、MK-WTA、Multi-Stage Multi-Robot、Cooperative Hunting Liao、Self-Organized Multitarget Pursuit、Adaptive Force Ratio，记录在 manifest。`NOT_FOUND` 是有界检索结论；AFRA 在 Gitee 找到，不能用 GitHub 零结果声称全网无代码。

## 2. AFRA 官方仓库：必须修复的差异

固定 commit 链接格式：`https://gitee.com/lqt20/multi-uav/blob/72440fdf0aa9516e87fcd52f282ce2a612c2df68/<path>`。以下是文件行号，代码读取自实际 clone，不以 README 宣传或 GIF 判定能力。

| 文件及行 | 静态证据 | 对复现合同的影响 |
|---|---|---|
| [multi_target_simulation.py](https://gitee.com/lqt20/multi-uav/blob/72440fdf0aa9516e87fcd52f282ce2a612c2df68/multi_target_simulation.py) L9–19 | 三维，地图 2000、半径 100；正文为二维、5000、200 | 不同仿真实例，不能称开箱即复现论文参数 |
| 同文件 L122–127 及其后注释 | ≥4 进入半径便 `return True`；其后注释是三维 bounding-box/轴向分布检查，并非正文二维角间隙公式 | 未执行正文角条件；普通 proximity 被误用时会高估几何围捕；需按正文重建角条件并另适配 CoCap 间隙比 |
| 同文件 L151–173 | RAW_EXPERIENCE 声明但查询未用；`infer_success_rate` 用手写 logistic，阈值/斜率写死 | 未实现正文经验库插值查询；需按 §4.1 重建/验证 |
| 同文件 L175–232 | DP 选人数；L224–230 将剩余预算轮流追加 | 追加可能破坏 `{0,4,…,15}`：按代码逻辑，单目标预算 3 可追加为 3，预算 16 可追加为 16。**这是静态推导反例，未执行** |
| 同文件 L267 | `target_speed=UAV_SPEED*r` | 代码 r 为 vT/vU，论文 λ 为 vU/vT，接经验库前需取倒数；不能直接传同名 ratio |
| 同文件 L245–289、L304–329 | 开始时分组；运行中设置 captured、被捕获组停动 | 有完成标记，但没有组释放后转投剩余目标、在线新目标或每步重新 DP 的完整路径 |
| `task_reliability_analysis.py` L126–143 | 二维配置较接近正文，捕获却用至少 3 + max gap <π | 与正文至少 4 不同；不能将此单目标 MC 结果无核验接入另一套多目标捕获规则 |
| `run_dynamic_reallocation.py` | 未见 `.py` 源文件，仅跟踪的 CPython 3.14 `.pyc` 及 CSV/图 | 有结果资产不代表动态实验可从源重现；不运行来源不透明的字节码补缺 |
| `run_heterogeneous_analysis.py`；`multi_target_simulation.py` L343–345 | 前者只调用再绘图函数；后者函数体是 `pass` | 入口存在仍不等于实验实现完整 |
| `uav_grid_roundup.py` | 另一套随机搜索/记忆/组件共享演示，README 描述的入口偏向此脚本 | 不等于论文 AFRA 多目标 DP 管线；不能混用它的搜索能力为论文补齐生命周期 |
| `test_allocation_fix.py` | 打印几个资源总量结果，未见断言和完整几何捕获测试 | 不足以验证分配合法性或论文成功率；仓库 CSV/GIF 也不是本轮重跑证据 |

仓库夹带 `maddpg-master` 的许可证和 `.pth` 等资产不能证明 AFRA 项目整体许可或它需要 RL；本轮未加载这些权重。没有根许可时，先按论文公式独立重写比直接复制代码更可控。AFRA 的精简 DP 可重建，但完整复现必须补经验库、统一二维动力学/终端、分组/槽位、跨群避碰、动态释放与依赖配置。

**结论：直接拿该仓库跑“论文 AFRA 完整基线”是 NO_GO；以论文为依据重建并标注差异是 CONDITIONAL_GO。**

## 3. 可复用的官方优化/规划组件

| ID / 官方仓库 | 固定 commit / 许可 | 核心文件、入口/测试 | 依赖与真实范围 |
|---|---|---|---|
| V4-C02 [google/or-tools](https://github.com/google/or-tools) | `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`；Apache-2.0 | [`ortools/graph/python/min_cost_flow.cc`](https://github.com/google/or-tools/blob/100f66e6242ab8bf8d32feb8f3bf086db66ae2b5/ortools/graph/python/min_cost_flow.cc) 暴露 SimpleMinCostFlow；`ortools/graph/min_cost_flow.h`；`ortools/graph/min_cost_flow_test.cc`；`ortools/sat/samples/assignment_teams_sat.py` | `ortools` Python wheel/原生库；整数费用/容量，需确定缩放和溢出边界；固定需求流或 CP-SAT，不含围捕。`OFFICIAL_REPRODUCIBLE（组件）` |
| V4-C03 [scipy/scipy](https://github.com/scipy/scipy) | `9c80f0d98ca62a8d9ff96156bf05d84fc4426499`；BSD-3-Clause | [`scipy/optimize/_lsapmodule.c`](https://github.com/scipy/scipy/blob/9c80f0d98ca62a8d9ff96156bf05d84fc4426499/scipy/optimize/_lsapmodule.c)，`rectangular_lsap/rectangular_lsap.cpp`；`tests/test_linear_assignment.py`；入口 `scipy.optimize.linear_sum_assignment` | NumPy/SciPy wheel；modified Jonker–Volgenant；测试覆盖形状、NaN/inf、重复值等，但本轮未跑。不能解决人数激活/等待优先级。`OFFICIAL_REPRODUCIBLE（组件）` |
| V4-C04 [AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) | `cdd0cc888802b584c2d654ca85e3c5460973487d`；MIT | [`PathPlanning/AStar/a_star.py`](https://github.com/AtsushiSakai/PythonRobotics/blob/cdd0cc888802b584c2d654ca85e3c5460973487d/PathPlanning/AStar/a_star.py)，`PathPlanning/PotentialFieldPlanning/potential_field_planning.py`；各文件 `main`；`tests/test_a_star.py` / `test_potential_field_planning.py` | A*/APF 子集主要 NumPy/Matplotlib；完整 requirements 还含 SciPy/CVXPY/ECOS/Pytest 等。单目标规划演示，不含多目标多人围捕。`OFFICIAL_REPRODUCIBLE（演示组件）` |

三者所选功能均不要求 Gurobi、MATLAB 工具箱或 ROS。TR-11 原生 Mosek、TR-13 原生 Gazebo/Quadprog 环境属于完整论文重建的另一条路径，不能因组件不需要而抹除论文依赖。Mosek 替换为别的 LP 求解器也需验证可行性容差和松弛处理；V3 的 SDQP 是 QP 组件，不自动等价原生 LP。

OR-Tools 当前 commit 没有所猜的 `ortools/graph/samples/assignment_min_cost_flow.py` 和 `assignment_teams_min_cost_flow.py`，这两个 URL 返回 404；使用已核核心绑定及[官方文档示例](https://developers.google.com/optimization/flow/assignment_min_cost_flow)。SciPy 实际绑定是 `_lsapmodule.c`，不是 `_lsap.c`。PythonRobotics 依赖路径是 `requirements/requirements.txt`，不是根目录文件。审计记录保留这些失败，避免下轮继续引用不存在的入口。

源码 pin 是审计定位，不等于已验证安装版本。SciPy 该 pin 为当日 HEAD；PythonRobotics requirements 声明 NumPy 2.5.3、SciPy 1.18.1、Matplotlib 3.11.2 等，**本轮未核验这些版本是否可安装**。preflight 应选兼容的已发布 wheel、单独锁版本和测试，不强行安装整个示例库。PythonRobotics 的 model-predictive-speed-and-steer 示例采用汽车自行车转向模型，不能直接当作 CoCap AW9 控制器。

## 4. G-CAP 的复用与自建边界

可直接考虑复用：SciPy 槽位指派、OR-Tools 固定配额流、PythonRobotics A* 结构；V3 Robotarium/SDQP 可参考连续几何或安全接口，但其保证不适用于 AW9 的自动替换。

必须自建并验收：全局任务表与版本同步、配额策略、几何代价/相位、AW9 rollout、跨联盟冲突检查、捕获事件处理、释放重分配、无目标部署、外生到达接口和成本计量。需要的接口是状态/感知快照、`action_index`、normal-capture/remove 事件、评估日志；不需要训练权重。

仓库源文件含真实算法和测试，仍不代表本轮已经重跑或测出性能。完整系统目前没有运行入口或配置，不能预先标成 `OFFICIAL_REPRODUCIBLE`。低层与生命周期是预计工程风险的主体，分配器本身通常较容易验证。
