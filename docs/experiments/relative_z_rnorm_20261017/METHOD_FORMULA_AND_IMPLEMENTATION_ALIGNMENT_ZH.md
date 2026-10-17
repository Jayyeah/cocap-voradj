# Method公式与实现对齐

设R=W=H>0为环境配置最终解析的世界地图边长，单位米。令N_i^obs(t)仅含动作前实际进入policy observation的有效友机，z_i为self token中的Z。z_{i,max}=max_{j in N_i^obs} z_j，无邻居时为0。

l_i^t=Capture（原直接围捕/既有effective Capture判定成立）；否则若z_{i,max}>z_i（实现浮点同值容差1e-6）则Support；否则Coverage。非Capture的Support资格完全由可观测相对Z决定，绝对Z只控制混合强度。

H_i^t={j in N_i^obs: z_j=z_{i,max}}（实现沿用1e-6近似同值）；j_i*=argmin_{j in H_i}||p_i-p_j||₂。同距时沿用float32可观测几何确定性字典序，几何完全相同时才ID破平；不按token顺序、也不平均位置。alpha_i=z_{j_i*}。

r_i^ally=clip(||p_i^t-p_{j_i*}^t||₂-||p_i^{t+1}-p_{j_i*}^t||₂,-3,3)。友机动作前位置固定为本transition锚点。

C_i(s)=n||p_i-c_i||₂²/R²，Phi_i(s)=-kappa*C_i(s)。U_i是沿用原速度、加速度及角速度各自物理上限归一化后的控制代价，权重schedule不变。r_i^disp=beta[-C_i(s^{t+1})-U_i^t+gamma*Phi_i(s^{t+1})-Phi_i(s^t)]。

r_i^support=alpha_i*r_i^ally+(1-alpha_i)*r_i^disp。正式capture scale=1且ally weight=1，无隐含双乘。Coverage使用完整r_disp，Capture沿用CR-MS/IW候选及原事件。通用安全/终端项仍由原环境统一追加，不将其误写为Support混合内部项。

遇原phase boundary/episode terminal，按本transition实际CE权重w_i（Coverage=1，Support=1-alpha，Capture=0）消去已加入的下一状态potential：delta_reset=-w_i*beta*gamma*Phi_i(s')=w_i*beta*gamma*kappa*C_i(s')，每transition最多一次。若CE未进入奖励或w_i=0则不消项。动态w_i下一步可变化，因此不声称全程固定potential的策略不变性。

Z同步传播、lambda=eta=.5、hard-zero .1不变；失活目标不额外清Z，有真实Recovery时局部历史证据短时衰减，无真实transition时不追加奖励。

与旧模型相比仅改变：(1) Capture邻居Support资格及不必要active_targets门控→动作前可观测Relative-Z；(2) 地图归一化L→R，附带为保持同一物理成功条件的距离阈值L/R转换。方形新距离为旧值sqrt(2)倍，CE平方代价和对应PBRS为2倍；beta与控制代价不补偿缩放，不称影响很小。网络、优化目标、物理动力学、捕获事件、Replay配额、episode/recovery、课程和选模保持原合同。
