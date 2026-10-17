# V3 逐层开源资产审计

核查日：2026-10-17。外部仓库均固定 commit，只做只读克隆、README/许可证/核心文件静态检查；没有安装依赖、执行示例、加载权重或启动仿真。分类使用 OFFICIAL_REPRODUCIBLE / OFFICIAL_PARTIAL / THIRD_PARTY / NOT_FOUND；**分类只对写明的组件范围有效**，不将 allocator 或 simulator 记成完整围捕实现。

## 1. 固定版本与可复用范围

| ID | 官方仓库及固定 revision | 分类与许可 | 真实可用范围 |
|---|---|---|---|
| V3-C01 | [inmo-jang/pyGRAPE](https://github.com/inmo-jang/pyGRAPE/tree/7f4b619b188ffb8b4257b51168c460a4957826d7)，`7f4b619b188ffb8b4257b51168c460a4957826d7` | **OFFICIAL_REPRODUCIBLE：基础分配示例**；GPLv3 | 作者 README 明确 Python-equivalent；核心评分、partition、mutex 与图示入口齐；不含已核实的最小人数 §V 分支 |
| V3-C02 | [inmo-jang/GRAPE](https://github.com/inmo-jang/GRAPE/tree/0d3aa265cbf6d97f1b933b9781985563d3643fb5)，`0d3aa265cbf6d97f1b933b9781985563d3643fb5` | **OFFICIAL_PARTIAL：论文部分 Fig.7 代码**；MIT | MATLAB 分配演示与结果 .mat/GIF；不是 Python 版同许可，也不是完整最小人数/动态围捕实现 |
| V3-C03 | [inmo-jang/space-simulator](https://github.com/inmo-jang/space-simulator/tree/b5286bbb1f99f81caf3f0ee50212bc58d4348bea)，`b5286bbb1f99f81caf3f0ee50212bc58d4348bea` | **OFFICIAL_REPRODUCIBLE：原生任务分配模拟**；GPLv3 | main/config/MC、行为树、局部感知、CBBA/GRAPE/greedy、工作量完成与新增任务 |
| V3-C04 | [reso1/LS-MCPP](https://github.com/reso1/LS-MCPP/tree/9dcba0b16a3a7669e7632f1b81a04a029d9ba304)，`9dcba0b16a3a7669e7632f1b81a04a029d9ba304` | **OFFICIAL_REPRODUCIBLE：TRO 已知图覆盖/冲突消解**；GPLv3 | master 指 TRO；aaai24 是前身分支，不能混用；ESTC/local search/PBS/实例/notebooks |
| V3-C05 | [robotarium/robotarium_python_simulator](https://github.com/robotarium/robotarium_python_simulator/tree/2b1b6136a242c164e53081b794ebf10d904cbaae)，`2b1b6136a242c164e53081b794ebf10d904cbaae` | **OFFICIAL_REPRODUCIBLE：通用控制/安全示例**；MIT | 官方 Robotarium simulator、SI/unicycle barrier/control；不是 CoCap controller 或 TR-11 实现 |
| V3-C06 | [sivic/hedac-maze-exploration](https://gitlab.com/sivic/hedac-maze-exploration/-/tree/c7905951141b1683e06731175dfa242731d859f7)，`c7905951141b1683e06731175dfa242731d859f7` | **OFFICIAL_PARTIAL：公开原生探索代码，发布/环境说明仍有缺口**；许可证未找到 | 主算法、求解器、迷宫、批次脚本、pyproject/uv.lock 均存在；README 的 requirements.txt 不存在；不认定已完成干净环境复现 |
| V3-C07 | [OPT4SMART/multirobot-tutorial](https://github.com/OPT4SMART/multirobot-tutorial/tree/ccde5f0d55d69ad97ec40d49ebf83f8af102c37a)，`ccde5f0d55d69ad97ec40d49ebf83f8af102c37a` | **OFFICIAL_PARTIAL：仅 README**；无许可证 | tutorial 论文脚注明确指向此仓库，但当前只有 README；没有可运行算法或实验配置 |

所有“REPRODUCIBLE”均指算法/入口/配置在静态审计中基本齐备，**不代表本地运行已通过**。HEDAC 的 partial 不表示已证实算法缺失，而是安装说明与交付条件尚待解决。公开可见不等同于已核可自由再分发；当前只提交研究笔记，没有复制这些源码。

## 2. GRAPE：理论重建容易的部分与困难接口

V3-C01 核心是 [pyGRAPE.py](https://github.com/inmo-jang/pyGRAPE/blob/7f4b619b188ffb8b4257b51168c460a4957826d7/pyGRAPE.py)：L74 起 scenario；L120–169 `calculate_utility`；L172–230 `distributed_mutex`；L232 起 `grape_allocation`；脚本 main 提供可视化入口。依赖 NumPy、Matplotlib、ImageIO，README 给 `python pyGRAPE.py`。没有精确依赖锁文件、自动化测试合同或 pretrained weights；这是非学习分配，不需要权重。

代码效用有 peaked/logarithm/constant/energy/random 分支，读取完整 `environment['task_locations']`、agent_locations 和当前 allocation。所谓 peaked reward 是总任务收益设计，实际个人收益仍随参与人数减小，不能误当作对三人阈值的实现。全队通信矩阵是模拟网络的参数，不表示有中央决策器；**但默认任务目录/坐标并非未知局部发现合同**。

V3-C02 的 `Fig_7_Visualisation/Get_Util.m`、`Task_Allocation_SC_visual.m`、`Main_visual.m` 与 `GRAPE_instance.m` 可作较宽松许可的公式/组织参考。`Get_Util.m` 未见 §V 的 R_j、辅助效用和 Condition 1。结果 .mat 是实验配置/演示资产，不是训练权重。MATLAB 专用工具箱需求未在该范围文档中完整声明，不擅自写“无工具箱”；若选择 Python 重建数学算法，则不需要执行 MATLAB 演示。

拟移植接口：本机 task registry→偏好计算；partition 快照→实际消息；最低人数→原论文 §V 重建；到位人数→会合/几何控制；完成事件→释放。快照 allocator 难度**低至中**，局部动态多轮整合**中至高**。首要风险是偏好条件、信息传播与移动代价持续变化，不能继承冻结实例的收敛保证。

## 3. SPACE：最接近任务循环的开源框架，不能原样作围捕环境

固定核心文件：

- [modules/behavior_tree.py](https://github.com/inmo-jang/space-simulator/blob/b5286bbb1f99f81caf3f0ee50212bc58d4348bea/modules/behavior_tree.py) L90–98 本地检测/消息；L117–139 到达任务后扣工作量并继续 follow；L143–163 定时随机 waypoint 探索。
- [modules/agent.py](https://github.com/inmo-jang/space-simulator/blob/b5286bbb1f99f81caf3f0ee50212bc58d4348bea/modules/agent.py) L147–154 邻居消息；L263–306 从全局 Agent/Task 容器按半径过滤，半径≤0 时变成全局范围。
- [plugins/grape/grape.py](https://github.com/inmo-jang/space-simulator/blob/b5286bbb1f99f81caf3f0ee50212bc58d4348bea/plugins/grape/grape.py) L17 以所有 `tasks_info` 初始化 partition 的任务 ID；L58–68 直接查看 assigned_task.completed 并清队/重分配；L137–148 效用；L150–165 mutex。
- `plugins/cbba/cbba.py`、`config.yaml`、`config/example/`、`mc_runner.py`、`main.py` 均存在。依赖 requirements 为 pygame、py-trees、PyYAML、ImageIO、pandas、Matplotlib、seaborn；代码另使用 NumPy，不能假设 transitive dependency 始终满足，应在 preflight 中显式冻结。README 的一个 MC 配置文件名与实际列表也需核实。

原生执行入口 `python main.py`，MC 入口 `python mc_runner.py`；已有图片/GIF/结果记录，没有本轮运行的测试或新实验证据。原文明确省略障碍和机间避碰；任务是静态位置工作量消耗，多人参与可加速而非至少三人才能捕获。

特别风险是**直接对象引用**：半径过滤虽然存在，但方法可沿 assigned_task 和全任务目录获取新状态/完成标志；消息中的 partition 也以 Python 对象共享。CoCap 移植应只摘取接口组织思想，输入用不可变 DTO，消息做深拷贝/序列化与带宽统计。不能把原生 simulator 的 `Task.set_done()` 当作 normal capture。

建议复用行为树顺序和插件边界，保留 CoCap 环境/评估器。复用成本**中**；把整套 pygame 模拟器替换成 CoCap 动力学成本更高且没有比较收益。不依赖 Gurobi/MATLAB/ROS；围捕控制和严格局部信息适配由项目新增。

## 4. LS-MCPP：开源覆盖强组件，依赖必须逐项区分

`mcpp/estc.py`、`mcpp/local_search.py`、`mcpp/operator.py` 是覆盖核心；`conflict_solver/high_level_planner.py`、`low_level_planner.py`、`reservation_table.py` 是 PBS/低层规划；`demo.ipynb`、`demo_turn_min_mcpp.ipynb` 是入口；`data/gridmaps` 与实例存在，额外大结果包通过外部链接提供，未下载。

`setup.py` 声明 Matplotlib、NetworkX、`numpy<2`、PyYAML、SciPy、**gurobipy**。本轮检索当前受控 `.py`，仅 setup 中出现 gurobipy，未见主覆盖/冲突消解代码调用其 API。因此应写“原生安装声明该依赖，核心路径是否可剥离需 preflight”，不能写整套算法数学上必须 Gurobi，也不能未经运行就保证无商业 solver 依赖。Jupyter 运行条件需另冻结。

已见 notebook、地图与 demo，不等于所有论文对照/硬件实验都能原样复现；无学习权重需求。CoCap 要将允许信息产生的 map 转成图、定义 footprint/栅格尺度、处理中断/恢复、把路径转成 AW9，并重验 swept 安全。图上的等待和边冲突消解不是实际有惯性转向的安全证明。原生覆盖复现**中**；局部持续围捕系统**高**。优先用途是 H-G 的搜索模块或额外覆盖对照。

## 5. Robotarium：安全组件有价值，输入合同差别很大

`rps/utilities/barrier_certificates.py` L19–33 是 CVXOPT QP helper，L36/122 是 SI barrier，L233/273 是 unicycle wrapper；`transformations.py` L97 起 look-ahead mapping；`controllers.py`、SI/uni barrier 示例与 go-to-goal 示例齐备。`setup.py` 明确 Python≥3.10、NumPy、Matplotlib、CVXOPT。README 的 simulator requirements 与平台可接受的大 `libraries.txt` 应分开，不能把后者所有 GPU/PyTorch 依赖当运行 barrier 必需品。

QP 输入使用整个所传机器人状态矩阵，默认不是符合 CoCap 有限邻接预算的消息协议；约束规模随机器人数增长。它输出连续 SI 速度或 `(v,ω)`，AW9 输入却是加速度/角速度加 drag。映射、截断、0.5 s 保持与预测误差改变了安全条件。

建议只复用约束构造/solver 结构或原生控制参考来做适配实验；不能把 Robotarium 软件叫做“已开源三机围捕”。无 Gurobi/MATLAB/ROS 原生必需；示例和 testbed 能力存在，但本轮没有向真实 Robotarium 提交/运行。通用组件复现**低至中**，AW9 有效安全过滤**中至高**。

## 6. HEDAC：有最新正式论文与算法，仍不是首选局部搜索栈

`maze_exploration.py` 中 `MazeExploration`、SOR/BiCGSTAB/GMRES/DGESV、`calculate_potential` 与 `exploration_step`；`example_50x50_4_agents.py`、`prepare_maze_configurations.py`、`run_small_test.py`、`run_all_test.py`、迷宫布局和初始位置配置均存在。批次脚本是原作者交付资产，**没有在本轮执行**。

存在 `pyproject.toml`、`uv.lock`；前者声明 Python≥3.12，joblib≥1.6.0、Matplotlib≥3.11.2、mazelib≥0.9.16、NumPy≥2.5.3、SciPy≥1.18.1。这只是该 commit 的声明，包可获取性/平台兼容性/锁文件重现未验证。README 指向的 requirements.txt 不存在，须用实际项目元数据设计干净环境。未找到 LICENSE/COPYING，故不把公开代码直接称已核开放许可软件。

论文原生离散正交移动、一步感知、集中势场/顺序占位，发现静止目标就终止。CoCap 需要连续足迹、AW9、年龄重访、追捕中断恢复、邻居地图交换及围捕层。原生复现**中**；改造成 local persistent search **高**，且原论文的覆盖/死锁结论不直接继承。当前只建议参考原式，不复制源码进入主仓库。

## 7. 缺源码方法、第三方边界与采用顺序

TR-11/TR-13 沿用 V2 `NOT_FOUND`，仍然是完整传统强基线第一/第二；未因为没有代码而排除。C*、Multi-CAP、sensor-based coverage、CFG-BAP、TNSE MTR 方法的官方实现本轮均未找到；搜索结果限于论文/作者页面与所列 GitHub repository query，不能证明全网无源码。

本轮没有验真后提升为 THIRD_PARTY 的完整方法。SPACE 中的 CBBA 是 SPACE 作者实现，不能称 CBBA 原论文作者代码；但它作为 SPACE 插件属于官方仓库资产。通用 CBBA 搜索结果、框架仓库、动画和任务分配器均未被升格为完整围捕实现。

建议采用顺序：**独立重建 GRAPE §V 的小型分配器 → 参考 SPACE 行为树/事件接口 → 保持本项目 AW9/捕获/搜索模块透明 → 再考虑 C*/LS-MCPP 搜索升级或 Robotarium 连续参考比较。** 最先优化的应是任务合同和可达性，而不是引入 ROS、Gazebo 或大型 solver 栈。TR-11 的原生公式重建走单独路线，保持论文方法身份。
