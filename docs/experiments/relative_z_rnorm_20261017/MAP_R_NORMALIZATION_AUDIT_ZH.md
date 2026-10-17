# 地图R归一化审计

唯一正式新模式normalization.map_scale=side_length_r。R由每个实际环境最终解析的env.width/env.height读取，单位米；严格验证有限正数及方形。非方形环境preflight失败，不能偷偷选width/max/mean或改变地图。R是完整世界边长，不能使用Coverage inner区域边长。旧配置缺省legacy_diagonal，旧Z05回归路径保留。

C_i=n||p_i-c_i||²/R²，Phi_i=-kappa*C_i，r_disp=beta[-C_i(s')-U_i+gamma*Phi_i(s')-Phi_i(s)]。正式beta=10、kappa=1、gamma=.99。旧方形L=sqrt(2)R，新距离sqrt(2)倍、平方CE及其PBRS 2倍，control cost与reward-scale保留；这是有实际奖励尺度影响的改动。动态(1-alpha)乘完整CE及相同权重的phase/terminal correction，最多消项一次；不宣称动态权重下全程固定potential望远镜/策略不变性。

成功阈值来自原YAML，运行时按legacy diagonal/R转换距离阈值。方形世界strict RMS .05→.05sqrt(2)，Max .10→.10sqrt(2)；loose的相应distance阈值同样转换。物理容差为threshold_old*L/sqrt(n)，保持不变；若平方代价作为阈值则应乘2，本线正式success使用RMS/Max距离，没有平方阈值。日志同时保留原物理distance/RMS/Max、R归一化distance/cost、物理阈值和成功判定。边界两侧用例核验旧新判定一致。没有将新的无量纲阈值作为额外实验变量。

所有self的障碍/边界/centroid、friend/target的relative位置与距离、obstacle的relative位置/输入半径/距离使用同一R；速度、朝向、Z、mask、types不按地图缩放。障碍输入半径的表示改为r/R，真实半径不改变。真实捕获半径8、传感半径、障碍半径、速度/加速度/角速度、避障边距、网格分辨率/对角邻接几何保持不变。

调用点详见MAP_R_NORMALIZATION_CALLSITES.csv。_map_diagonal保留为几何历史接口，新正式consumer不调用它；trainer的voradj_v1_to_v2旧权重适配是历史路径，Stage1不加载权重、Stage2/3只允许shape_compatible，不能进入。normalized_center_costs通用参数改名map_scale，接收显式R；正式CE scalar consumer由共享接口读取R。base旧Coverage分支虽非本次VorAdj正式目标，也在启用新模式时读取完整世界R，并转换它原inner-diagonal阈值；缺省旧配置保持旧值。

独立评价使用同一VorAdjEnv及resolved config的native factory，未调用configure_large_case。后者120数值属于原大场景物理配置，不是归一化硬编码，保留并明确不属于新正式native课程调用链。renderer按真实世界坐标画图，图像bounds不作策略/奖励归一化；55/120的历史渲染fallback不当作R。

初次主Agent新/旧定向测试61项通过。最终完整证据与各阶段实际宽高见map_preflight.json和独立报告；未出具的测试不预填PASS。
