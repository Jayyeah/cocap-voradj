# Forward-Final MAPPO：P1 transition / return semantics 根因审计

**结论：接手时 FAIL；确认4类可复现缺陷，均完成最小修复与修复后 trace。`P1 TRANSITION/RETURN SEMANTICS: PASS`（修复后的当前4v1生产链路与已列边界）。** 允许继续P2固定策略诊断；P2校准没有因此通过，P3和25k+ PPO仍HOLD。本轮没有正式训练、扫参、Actor/critic拟合或修改已有checkpoint。

审计基线为 `7e6596fc498f2e32bec504e36d0510400de81614`。2026-09-14启动先fetch origin，再核对实验worktree并执行 `git merge --ff-only origin/experiment/small-step-ac-migration-20260828`，结果Already up to date。最近提交为 `7e6596f`（two-head负对照）、`dad17b4`（phase梯度审计）、`259bc32`（context校准HOLD）。已读取root-cause/corrected PPO/parity三份台账、最新two-head的launch/validation及相关测试；历史C3/P2不重新包装为本次结果。

## 1. 证据与范围

- [最终独立验证](../artifacts/2026-09-09_root_cause/transition_audit_20260914/validation.json)、[最终报告](../artifacts/2026-09-09_root_cause/transition_audit_20260914/after_v2/report.json)、[runtime/source/normalizer记录](../artifacts/2026-09-09_root_cause/transition_audit_20260914/after_v2/launch.json)。
- [逐transition可读表](../artifacts/2026-09-09_root_cause/transition_audit_20260914/TRANSITION_TABLE_ZH.md)，包括两次capture前5步、capture、post前20步、两次pure reset前20步、真正终止、timeout与其后reset，以及capture+casualty→inactive snapshot reset。
- 完整586条、每条4 agent的表：[修复前CSV](../artifacts/2026-09-09_root_cause/transition_audit_20260914/before_v2/native/transitions.csv)、[修复后CSV](../artifacts/2026-09-09_root_cause/transition_audit_20260914/after_v2/native/transitions.csv)。完整精度、每agent reward分量、before/after context、obs/central SHA和事件：[before JSON.gz](../artifacts/2026-09-09_root_cause/transition_audit_20260914/before_v2/native/transitions.json.gz)、[after JSON.gz](../artifacts/2026-09-09_root_cause/transition_audit_20260914/after_v2/native/transitions.json.gz)。其余受控probe同目录结构。gzip无有损压缩，解压SHA见报告/manifest。
- 使用实际 `FinalMissionStream`、`collect_transition`、`stack_rollout`、`MAPPOTrainer`，不是另写一个近似collector。只在诊断子类记录step前后状态；不插入reward/flag修正。生产修复与审计代码分开保存。
- frozen BC为现成C3 parent，文件SHA `7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd`；V使用现成geometry MC-100 fitted checkpoint，ValueNorm固定mean **17.028923** / std **47.180908**，不是用零V掩盖bootstrap。Actor/V/normalizer均逐bit不变。
- 复用原heldout seed stream `2027099201` 的episode 0..3，长度176/84/225/101；mixed capture发生在t=57/80；两个pure均为原生capture_snapshot来源，第二次pure复用同一个coverage环境实例。修复前586条的geometry/phase/MC target逐bit等于原bank；修复后geometry/phase仍逐bit相等，50维pre-action context也逐bit等于原context bank。没有增加独立统计样本。
- 运行于当时空闲GPU1、process-local cuda:0，仅冻结推理；数学/回归/最终复核在CPU。受控probe保留真实3000/500 horizon，只将clock置于临界点，不虚称自然运行了3000步。固定AW9 index4的probe明确是干预；自然episode仍为固定RNG的BC sample。

## 2. 时间语义逐环节裁决

| 链路 | 实际消费者与运行证据 | 修复后结论 |
|---|---|---|
| s_t→actor/role/context | `local_and_global`读当前stream observations；env reward使用before_labels。trace对照真实actor self[-1]、effective role、metadata.phase与50维同步context | 同时刻；没有one-step lag |
| a_t→physics→reward | AW9 indices与physical grid逐项相等；reward属于此次动作，逐agent `reward_total = capture + coverage + safety + terminal` | 分量与storage一致 |
| capture→recovery | capture reward留在pre行；event后清K10、重建obs；next context为post。没有其它任务terminal时capture不中断 | PASS；support correction修复见BUG-3 |
| post_started flag | capture后的s_(t+1)已无目标、phase=post，但started=false、post_step=0；第一条recovery action完成后才started=true、step=1 | 计数定义，非phase延迟 |
| next state→V_next | collector在finish/reset前构建central；逐步独立forward与stored next_values逐bit相同；同episode下一行的local/central/context/V_t与上一行after/V_next逐bit相同 | 无reset值替代 |
| truncation→GAE | pure timeout bootstrap物理末态；episode_end截断递归；真实terminal优先于同一步timeout | PASS，旧env边界矛盾已修复 |
| reset→新episode | 原生reset清episode_step/hold/post/stationary/release；允许复制capture位置/active作为显式reset来源 | 位置初始化不是泄漏；inactive Actor路径另有BUG-4 |
| active/episode masks | storage保留pre-action active，包括此次动作导致死亡的agent；其死亡reward有效。后续inactive行在GAE/actor loss排除，不能提前mask掉死亡动作 | PASS，采样现只处理active |
| ValueNorm | stored值是normalized prediction；实际update与critic-only的GAE入参均先按旧统计denormalize；GAE后才更新normalizer | raw reward、raw V、raw return同尺度 |
| flatten/minibatch | GAE先在[time,agent]递归；之后local/value/actions/advantage按同一time-major索引flatten。central attention在joint state内部，不沿episode或time轴注意 | 未发现agent/episode串接 |
| MC target | `full_returns`在每个完整episode独立反向累计，真正terminated处截断。原bank排除truncation，不把截短prefix当MC | MC构造正确；reward本身有BUG-3 |

数学口径为 `δ_t=r_t+γ(1-terminated_t)V_(t+1)-V_t`，`A_t=δ_t+γλ(1-episode_end_t)A_(t+1)`，inactive行置零，`return=A+V`。本轮用独立向前展开TD residual校验全部586条，最大误差 **4.01e−5**（float32）。实际两种update入口在GAE处被诊断异常截停，任何optimizer/normalizer更新之前已验证其入参和return；不是事后拿自写公式替代生产调用。将reset后reward改成1e6，前episode return变化0；单独污染另一agent也不影响本agent。

MC与GAE不要求数值相等：λ=.95使用估计V，完整MC使用实际未来reward；表中另给生产rollout_length=256的GAE。外生timeout/open tail的MC栏为NA；observed discounted prefix单独保存，绝不声称是完整MC。timeout probe若误用reset V会使target最多差 **116.44175**，真实collector未犯此错。

## 3. CONFIRMED BUG

### BUG-1：timeout reward采用terminal势函数清零，而PPO对同一transition bootstrap

**复现：** 相同coverage物理状态/动作，仅episode_step设2998或2999。旧env在done时无条件执行 `apply_ce_pbrs_reset(...,"terminal")`，因此第3000步reward比正常continuation多 `10γC_(t+1)`；collector却将它标成truncated并加入同一物理末态V。静止种子19最小环境probe的agent0：continuation reward=-0.483563739，timeout约0；这不是下一episode混入，而是同一transition把terminal reward和continuing V混用。

**最小修复：** generic CE terminal correction只对真正terminated执行，pure timeout保留末态势函数；capture phase correction仍按原phase合同执行。最终实际stream probe中两种clock下四agent reward严格相同：[-0.6405743,-2.7104278,-0.9399221,-0.7130964]。GAE正确为r+.99V_next；之后reset阻断递归。

**测试：** `test_external_timeout_preserves_continuing_pbrs_reward`。既有 `tests/test_ce_coverage.py` 曾明确断言timeout清零，恰好暴露env/IQN历史边界与PPO外生truncation假设的矛盾；本次前瞻更正该断言，保留历史结果。

### BUG-2：terminal与timeout同一步时，字符串优先级错误允许bootstrap

**复现：** coverage几何已收敛、hold=29、episode_step=2999，下一步CE成功却旧state="too long episode"，四agent被判truncated。post_window=499与3000总horizon重合、too_few、collision造成全局too_few，也复现；collision例中死亡者term=true，活着但任务已经结束的同伴仍错误trunc=true。

**根因：** env合并done，先写timeout字符串；`_split_termination_flags`仅依赖该显示字符串，真实结束原因丢失。

**最小修复：** env先计算mission/death terminated，再令truncated=(timeout或pre-timeout)且非terminated，显式保存两个布尔值及metadata；split优先读取这两个事实，兼容旧env的字符串fallback。done仍是二者OR。全部四种重合probe现为term=true/trunc=false，return=r；例如成功probe的V_next≈[-4.50,-4.29,-4.29,-4.52]仍被计算但完全不进入target。

**测试：** `test_true_terminal_wins_when_time_limit_coincides` 的success/post_window/too_few/collision四参数。

### BUG-3：support的CE boundary correction未乘0.5，并漏记入support CE分量

**复现：** 三机位于目标7m环上，第四机是support，AW9 index4，真实capture触发。普通support CE乘.5，但 `apply_ce_pbrs_reset`旧实现加完整correction。静止evader最小例：应加.088917834，实际加.177835668，support平白获得+.088917834；原 `reward_support_blend_coverage` 还停留在correction之前，不能解释最终reward。

**实际访问信息：** 原heldout episode0/t57/agent1：reward **.641214728→.502105653**；episode2/t80/agent2：**−.016034391→−.034719050**。两次都是native sample捕获，不只是构造状态。修复前MC target与旧bank逐bit一致，证明问题来自存入的reward，而非full_returns索引错误。

**最小修复：** 记录当步实际CE scale，用相同scale乘terminal/phase correction，并同步support CE分量；沿用已有“每transition最多一次correction”的guard。

**测试：** `test_support_capture_correction_uses_same_half_weight_as_ce`，并检查下一步没有重复capture bonus/correction。完整reward顶层分量重构逐步相等。

**因果边界：** 本次只改变两个pre-capture agent-reward及其向前传播的pre MC目标，float32 target最大变化.139109；**post和pure MC return逐bit不变**。因此这两条轨迹中的early-recovery MC欠拟合，不能由该capture bonus超计直接解释。

### BUG-4：capture+casualty snapshot产生inactive reset slot，collector仍对全零Actor观测采样

**复现：** 同三机capture fixture，第4机位于边缘并在同一步触发真实boundary死亡。原生step得到capture event、全局done和snapshot active=[1,1,1,0]；`finish`将其放入原生pool；选择该snapshot后pure reset保留inactive slot。旧collector把padding局部obs传进Actor，all-masked attention产生NaN logits，`Categorical`抛ValueError，第一条pure transition根本无法入库。固定选择pool来源只是让已有.75分支确定触发，没有伪造reset后的观测。

**最小修复：** `collect_transition`只对active local rows调用Actor，V仍读完整joint state；将动作/logp/index散射回原agent位置，inactive用AW9 neutral index4/logp0，实际env仍收到None。不修改原生pool资格、reset分布、reward或min_active=4。

**修复后：** 同一原生capture→pool→reset路径可存入有限数值transition，active mask为[1,1,1,0]；该pure场景仍按原min_active=4在第一步terminal，inactive return=0，随后mixed前20步继续正常。真实存储的active likelihood复算通过。

**测试：** `test_native_capture_casualty_snapshot_reset_samples_only_active_agents`。这是collector健壮性/active语义缺陷；没有证据表明原safe fixed-MC bank曾因此污染，它更可能表现为作业中断。

## 4. SEMANTICS CORRECT：已经排除的主要假设

1. capture大正奖励被写到首个post action、capture后无条件terminal、post role晚一拍：均排除（有同时casualty/任务终止时应terminal，不与capture继续规则矛盾）。
2. normal timeout的V_next来自reset后世界、GAE跨episode递归、pure第二次reset残留上次hold/release/phase：均排除；复制初始位置与active来源是显式reset合同。
3. critic next-input与实际s_(t+1)不一致、rollout保存引用后被原地reset覆盖：逐行SHA/独立forward/下一行衔接排除。
4. active mask提前丢掉死亡当步reward、GAE串agent、flatten错序、MC跨episode累计：实测/poisoning/原bank逐bitparity排除；BUG-4已修复采样侧缺口。
5. boundary处reward与V使用normalized/raw混合尺度：实际两个update入口排除。ValueNorm统计变化可能影响后续拟合，但这不等于本轮发现了时间错位。

## 5. SUSPICIOUS BUT UNPROVEN

- 4类bug是否解释历史低LR PPO效率退化：未证明。timeout/减员probe是受控边界，本次4条自然轨迹均safe、无timeout；无历史发生率或已执行更新的反事实因果证据。
- 已确认的critic state aliasing仍成立：生产geometry V没有完整phase/time/hold/release输入；本次记录50维context用于时间对齐，没有接入生产PPO。**缺失状态与one-step lag是不同问题**；本轮PASS不覆盖“critic输入Markov充分性”或“critic健康”。
- 原40回合的failure/outcome calibration仍不足；本次构造失败仅验证实现，不能作为泛化校准样本。inactive pool短回合可能影响未来训练采样分布，但没有证据授权本轮更改pool政策。
- 角色依赖PBRS是否严格等价于一个全局potential、外生3000步截断是否最符合任务目标，以及ValueNorm/优化/MC随机性的相对贡献，仍属P0/P2问题。本次修复针对已声明的PPO外生truncation合同，不做目标重设计。
- 结论限当前4v1 Forward Final路径；没有宣称8v2/12v3的非最后目标capture、所有历史IQN终止消费者或任意all-inactive场景均已审计。历史main仍需使用原commit复现。

## 6. 修复边界、验证与交付

生产改动限env的termination/PBRS消费者、termination split、Forward Final collector active采样及runtime/checkpoint语义标识；没有改actor/critic结构、γ/λ、ValueNorm算法、学习率、GAE递推、reward权重或正式训练预算。**已改变缺陷边界的实际reward/终止结果**，因此不能继续宣称新源码与canonical main“除collision外逐transition完全相同”。基础CONTRACT ID保留作谱系，新runtime/checkpoint另记录 `terminal-priority-truncation-bootstrap-weighted-ce-v2`；未来匹配实验必须同时核对该字段与源码SHA。preflight中的“only collision changed”是resolved配置及选定getter比较，不再作为历史边界reward逐步等价证明。

[新增9项测试](../test/test_forward_final_transition_semantics_20260914.py)：前三类修复前6 fail/2 pass；第四类另有1 fail。最终相关**57 passed**，仅2项既有protobuf弃用警告；[完整日志](../artifacts/2026-09-09_root_cause/transition_audit_20260914/regression_tests.txt)。审计中两次fixture记录中断明确保存于[startup notes](../artifacts/2026-09-09_root_cause/transition_audit_20260914/startup_notes.json)，不计为算法结果；有效baseline自然轨迹与最终修复后完整probe均已独立验证。原始前后源码快照逐SHA匹配launch，已有bank/checkpoint未覆盖。

复现（需本仓库已有BC/MC checkpoint和bank，SHA在launch内；不启动任何拟合）：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 tools/audit_forward_final_transitions_20260914.py --out /tmp/cocap-transition-audit-fresh --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pytest -q test/test_forward_final_transition_semantics_20260914.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 tools/validate_forward_final_transition_audit_20260914.py --out artifacts/2026-09-09_root_cause/transition_audit_20260914
```

完整stored rollout NPZ保留本地，SHA已记录；Git提交CSV、无损JSON.gz、报告、源码快照及测试。最后一条validator还需本地NPZ；仅查看时间表/每agent数值不需要它。gzip可用Python `json.loads(gzip.decompress(path.read_bytes()))`读取。

**Gate：P1时间对齐修复后PASS；P2允许继续无训练诊断，既有校准HOLD保持；P3/25k+ PPO不允许进入。** 下一步应在明确新旧boundary版本、保留原证据的前提下继续fixed-policy calibration定位；post/pure目标在本次配对中不变，不能因发现bug就宣称early-recovery根因已全部解决。当前无后台训练、无待唤醒任务。
