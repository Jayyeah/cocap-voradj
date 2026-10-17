# Baseline开源与资产审计（2026-10-17）

本轮为静态代码/资产审计，未安装候选依赖、未执行训练或rollout。检索过程追加至原[search_log](../../artifacts/baselines/search_log.md)，获取状态与hash见[来源manifest](../../artifacts/baselines/focused_review_source_manifest_20261017.json)。**论文开放、实物验证、视频、CSV或作者Software贡献均不等于整套方法代码开源。**

## 1. 四级分类与核查范围

- `OFFICIAL_REPRODUCIBLE`：作者官方仓库，算法、配置和入口基本齐全；并不表示本轮已跑通或复现论文统计。
- `OFFICIAL_PARTIAL`：作者代码存在但关键资产/模块/环境不齐。
- `THIRD_PARTY`：第三方实现，需独立验真。此轮未找到足以进入shortlist的整套传统第三方复现。
- `NOT_FOUND`：在列明的公开检索范围内未找到完整实现；不能推断不存在私有或未索引代码。

检查了精确标题/算法名与作者线索、原论文代码链接、出版社补充和公开GitHub repository search。后者只搜repository元数据，**不是完整GitHub code search**；零结果不构成不存在证明。部分网页全文被403/API401阻断，均留记录，不改写为“无论文”。未向作者发消息。

## 2. 候选代码身份

表中N/A表示没有找到可归属该论文的repo，因此commit、许可与文件不能填写。对TR-05，外部代码分类为NOT_FOUND但性质是project-defined/to-be-implemented；不是论文代码缺失。

| ID | 分类 | Repo URL | 可定位commit/tag | 许可证 | 核心算法文件/入口 |
|---|---|---|---|---|---|
| RL-01 TERL | OFFICIAL_REPRODUCIBLE（静态资产齐备） | https://github.com/ApricityZ/TERL | `143359b2722d49c29b4fecc0ad1fd8d46326e45a` | MIT，根LICENSE | `policy/TERL_model.py`、`policy/agent.py`、`policy/trainer.py`、`robots/pursuer.py`；`train_rl_with_configs.py` |
| RL-02 Role-based MADDPG | NOT_FOUND | N/A | N/A | N/A | N/A；普通MADDPG不实现该论文角色/奖励/场景 |
| TR-01 MK-WTA＋OAPM | NOT_FOUND | N/A | N/A | N/A | N/A；未找到作者软件或可归属算法补充 |
| TR-02 potential-game | NOT_FOUND | N/A | N/A | N/A | N/A |
| TR-03 ACC coalition | NOT_FOUND | N/A | N/A | N/A | N/A |
| TR-04 RMPC | NOT_FOUND | N/A | N/A | N/A | N/A；前身项目页有论文/演示语境，无整套代码链接核实 |
| TR-05 heuristic | NOT_FOUND（外部），项目待实现 | N/A | N/A | 待项目实现时沿用仓库规则 | 本轮没有新增controller文件 |
| TR-06 DMK-WTA＋PPM | NOT_FOUND | N/A | N/A | N/A | N/A |
| TR-07 KS-COAL | NOT_FOUND | N/A | N/A | N/A | N/A；arXiv全文/算法伪代码可得，不等于repo |
| TR-08 distance-based | NOT_FOUND | N/A | N/A | N/A | N/A |
| TR-09 triggered feedback | NOT_FOUND（整套方法） | N/A | N/A | N/A | ChoiRbot/CrazyChoir是底层框架线索，未核得本文完整实现 |
| TR-10 GB-VCE | NOT_FOUND | N/A | N/A | N/A | 附per-seed CSV/统计，未找到MATLAB源程序 |
| TR-11 Dynamic Robust Pursuit | NOT_FOUND（整套方法） | N/A | N/A | N/A | 有正式PDF与补充；补充链接SDQP组件，不能当整套围捕源码 |
| TR-12 Hierarchical PGM | NOT_FOUND | N/A | N/A | N/A | N/A；完整原文未取得 |
| TR-13 CEO-DMPC | NOT_FOUND | N/A | N/A | N/A | 作者机构有完整PDF，无归属本文的算法源文件/配置 |

其余第一轮RL候选不重新扩张检索：OPEN按第一轮缺资产证据归 `OFFICIAL_PARTIAL`；DualCL沿用作者官方代码证据但本轮未重验运行；LIPPO、PRTDA、Ex-MADDPG沿用NOT_FOUND。registry保留各自原核查日期，不能把继承信息冒称本轮新验证。

## 3. 环境、验证与迁移缺口

“未核实”不同于“无需”。没有repo的方法依赖栏记录论文已证实的平台或公式重建所需组件，不冒称其提供requirements/锁文件。

| ID | 依赖/商业工具/平台 | 已有验证或权重 | 缺少的必要功能与CoCap接口 | 完整移植难度 |
|---|---|---|---|---|
| TERL | Python≥3.10、PyTorch≥2.4、NumPy/SciPy/PyYAML/gym/wandb等；Docker基于PyTorch2.4/CUDA11.8；未见必需Gurobi/MATLAB/ROS | 官方`TrainedModels/TERL/network_params_v36.pth` 14,366,626 bytes及配置；未加载/运行验真 | 局部目标mask/selection、AW9/碰撞/normal capture、无目标coverage、释放与到达；权重不能假定跨合同适用 | 中；局部适配中至高 |
| TR-01 | MK-WTA数值网络、OAPM、差速轮机器人；具体语言/工具箱/solver未核 | 出版社报道仿真与差速轮实验；无公开权重/测试程序 | 全控制律/参数、输入矩阵/通信、AW9、capture事件、coverage和再分配 | 高至很高 |
| TR-02 | DBSCAN/博弈迭代/制导；语言、solver与专有依赖未核 | 摘要报道仿真 | 全文公式、人数、目标聚类语义、信息/控制/生命周期 | 很高 |
| TR-03 | coalition/资源/信用/merge-split；工具箱未核 | 作者论文有仿真，未找到可执行测试 | 三机围捕、动态目标控制、coverage、标准事件与通信限制 | 高 |
| TR-04 | 原生RMPC/TMPC、扇区构造；2021前身明确MATLAB；2025 Gazebo移动机器人仿真；商业solver/工具箱未核 | 仿真；未找到预训练权重或测试代码 | 单目标初包围、求解器/参数、AW9、搜索分配和任务循环 | 很高 |
| TR-05 | 拟用CoCap环境、几何/消息/规则；不需要训练；未实现故无真实requirements | 无；设计合同 | 全模块均待实现验证；清单见可行性§5 | 低至中 |
| TR-06 | DMK-WTA、PPM；语言/solver/ROS未核 | 出版社报道多机器人实物；无可执行资产 | 正文/配置、状态/通信、AW9、normal事件和搜索/恢复 | 高至很高 |
| TR-07 | KS-COAL消息与utility；完整Accelerated版本另需HGAN监督训练；工具栈未核 | v3大规模仿真与13机器人实验；无代码/权重核实 | 正式补充、有限侦察通信、三机几何、同质能力映射、AW9、事件/恢复 | 高至很高 |
| TR-08 | 离散双积分器、Kalman、CBAA、伪力；具体工具箱未核 | arXiv UAV/UGV实验；无代码 | 原生两机结构→三机需改方法，局部检测接口、搜索释放 | 很高 |
| TR-09 | ROS2、Webots、ChoiRbot/CrazyChoir架构；稳定低层动力学；专有solver未证实 | 仿真和地空实物；未核整套代码 | 单目标优化→多目标捕获/分配、AW9稳定性、事件与coverage | 很高 |
| TR-10 | 论文MATLAB2023a仿真；额外工具箱未说明；无ROS/硬件证据 | 100 paired seeds结果CSV；不是源代码/权重 | Γ/argmax/TopK协议、目标速度估计、多目标分配、AW9、移除释放 | 很高 |
| TR-11 | A*、分组/auction、C3AVC、CLF–CBF LP；论文用**Mosek**；独立SDQP为C++11/Eigen3 | 仿真7P2E、9P3E；动捕＋host的Crazyflie2.1实验4P2E；补充PDF；无整套测试/权重 | 原生积分器→AW9、路径图/求解参数、三机共同checker、捕获移除/coverage/新波 | 高；恢复理论很高 |
| TR-12 | hierarchical PGM/SPH为公开元数据线索；实际依赖未核 | 摘要报道仿真/实物，未取全文验细节 | 捕获/人数/信息/控制/循环均需正文核实 | 未知，高风险 |
| TR-13 | Python3、Quadprog、OR-Tools `pywraplp`；Ubuntu20.04/Gazebo；MILP backend未说明，不虚填Gurobi依赖 | 数值仿真和Gazebo UAV物理仿真；不是实物；无源码/权重/测试 | 全实时通信与预测融合、MILP参数/QP、AW9、normal移除、coverage/新波 | 高至很高 |

## 4. TERL官方实现的直接证据

固定revision的[README](https://github.com/ApricityZ/TERL/blob/143359b2722d49c29b4fecc0ad1fd8d46326e45a/README.md)自认本文实现；仓库含训练、ablation/baseline入口、`config/`和各模型资产。

[robots/pursuer.py](https://github.com/ApricityZ/TERL/blob/143359b2722d49c29b4fecc0ad1fd8d46326e45a/robots/pursuer.py#L276)中友机先经`check_detection`过滤（L283），障碍同样过滤（L334）；**L307–330目标循环仅跳过deactivated目标，没有距离过滤**。后续排序/token截断可限制目标数量，但不限制距离；所以应精确称“全距离目标输入，受max_evader_num截断”，而非无限容量全目标张量。局部适配必须连同target selection/奖励读取/历史缓存查漏，不能只遮一处坐标。

安装风险：`requirements.txt`写`python >= 3.10`与`pytorch >= 2.4.0`，并非可直接照抄pip安装的完整可靠锁定；Docker显式安装列表也不涵盖所有配置依赖。基本算法/配置/入口齐备满足本分类，但未做环境和checkpoint兼容性运行确认，不声称“一键复现已通过”。

## 5. 可独立复用的组件：SDQP

TR-11补充S2/S4明确链接[ZJU-FAST-Lab/SDQP](https://github.com/ZJU-FAST-Lab/SDQP)。本轮clone审计固定：

- commit：`78e9860508cc7c191870dea79e63b1e7c45e2296`。
- 分类：`OFFICIAL_REPRODUCIBLE`，**范围仅SDQP自身**，不是TR-11整套方法。
- 许可：MIT，位于[`include/sdqp/sdqp.hpp`](https://github.com/ZJU-FAST-Lab/SDQP/blob/78e9860508cc7c191870dea79e63b1e7c45e2296/include/sdqp/sdqp.hpp)文件头；根目录没有独立LICENSE不意味着无许可。
- 核心：上述header；示例`example/sdqp_example.cpp`；构建`CMakeLists.txt`；C++11、Eigen3、CMake，无ROS/MATLAB/Gurobi要求。
- 功能：小维度正定严格凸QP，A x≤b；可用于障碍支持平面QP，需把约束方向/二次项系数映射清楚。示例存在但本轮未build/run。
- **不能直接复用为原控制LP**：其Q要求正定；把线性目标加正则会改变问题，必须另验证。控制LP可先保留论文Mosek，或后续选独立LP solver并验证最优值/可行域；本轮不安装solver。

普通SciPy优化、聚类和几何库可作为重建基础，框架如ChoiRbot可作机器人中间件线索；均不等于已找到完整论文围捕实现。完整论文名称只能用于经过明确公式/合同核对的实现，不能用于拼装的自制controller。
