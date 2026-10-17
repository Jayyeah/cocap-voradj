# 旧训练安全暂停回执

状态：OPERATOR_SAFE_PAUSED。用户于2026-10-17明确授权替换实验，旧线已停止，禁止自动重启或交接Stage2/3。

- 原分支：`experiment/iqn-z-guided-support-full3stage-20261017`；重新fetch及ls-remote确认暂停前HEAD：`bec482f9a335a30d404969f959dbe58546836576`。
- 归档名称：`legacy-gated-zsupport-rnorm-diagonal`。
- 归档入口：`/home/yjq/rl/CoCap1/legacy-gated-zsupport-rnorm-diagonal`；它引用原runtime `/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime`，未移动或删除任何已有训练结果、恢复状态和评估资产。保留原绝对路径以便历史exact-resume分析。
- 最终环境决策步：**839,140**；真实optimizer updates：**208,922**。
- 当前episode：1408；任务 `voradj`；episode_step=985；current_episode_done=false；本episode transition计数=3940。
- 完整恢复包：`/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime/stages/stage1/training/checkpoints/resume_latest.pt`；5465121491 bytes；SHA256=`8c473c221ad559aef20db6679b2250e57fd1f8462970ae1a4a83d5d24e8acbdb`。
- 最终模型SHA256：`52e4c49dd6e8abde5e729ff390084242c212b69b48897d2fcc3a2fbe212233dc`；算法为参数名排序后依次加入UTF8参数名和CPU contiguous numpy字节。
- resolved YAML文件SHA256：`30b2ce46ba8f255a8b27bd19f4f4a165c4e5231058378b72f689f16aa05e72ec`；原登记resolved-config语义SHA256：`3d47e7edbab2aa5096de6333ad9a6db7768fe058f9ed9a8a20b5980d29288488`；exact-resume合同SHA256：`79c3bf57e3029ad6d630f5e38fb3bcaefd534051604ec2b4cbc25df47ef4e8a8`。这些SHA口径分别记录，不混为同一值。
- 独立torch.load成功核验 `cocap_iqn_full_resume_v1`：online/target模型、optimizer、runtime、四类Replay、两任务env/Z、当前observation/episode、Python/NumPy/Torch/CUDA RNG均存在。核验时间：2026-10-17T23:59:00+0800。

## 暂停过程与重启封锁

先识别监督器3586027、训练器3586100、worker3586095/3586096/3586097及tmux会话，再写原监督器认可的pause_request.json。训练器在下一动作选择前保存完整exact-resume并主动退出。没有向训练器发送终止信号，没有使用kill -9。

旧Supervisor对普通退出会最多重试3次。同步写入scientific_failure.json作为人工运维重启阻断标记，其内容明确说明这是operator hold，不是科学失败结论。发现其60秒wait超时可能中断多GB序列化，临时对已识别Supervisor发送SIGSTOP；训练器完整落盘并退出后，发送SIGTERM并SIGCONT使其正常终止。该保护过程约5分钟，未干预决策或模型状态。

3个旧评估worker当时均idle，正常SIGTERM退出；SQLite共8个selection任务，全部complete，无pending/running，不丢弃已完成资产。pause_request、重启阻断、人工暂停日志与status持久化保留。已核验5个PID不存在，tmux pane_dead=1，GPU上无旧训练PID；cron及用户systemd没有该实验重启条目。没有自动Stage2/3交接。

完整恢复包和轻量JSON回执相互核对。旧权重只允许历史/消融分析，**禁止作为新Stage1正式初始化**。本回执不代表新算法验收或新训练已经启动。
