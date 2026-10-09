# A1-MASTER-02 → Batch01 独立 QA handoff

> **当前执行状态：旧candidate已被正式BASE_QA_BLOCK，旧smoke lease未使用并关闭。下方Master02 handoff仅作历史；不可继续执行其CUDA模板或把窗口沿用于V2。当前等待Core V2，见[新candidate复验合同](BATCH01_CORE_V2_REQA_CONTRACT_20261009_ZH.md)。**

日期：2026-10-09，Asia/Shanghai。由用户直接转交 QA；Master 不创建、调用或等待额外子 Agent。中央唯一写入者仍为 Master。QA 只写自己的 branch/worktree 和新证据目录，交付报告后由用户返回中央。

## 1. 固定验收对象

| 字段 | 值 / 用途 |
|---|---|
| Core branch | `experiment/terl-mappo-batch01-base-20261009` |
| **科学 candidate SHA** | **`40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a`**；QA checkout 和测试必须固定此 SHA |
| Core 交付文档 HEAD | `bdbd502ff2f071b0cac07a0d0d956a2f35015ea1`；只用于读取交付报告，不能填为科学 BASE |
| Candidate lock | `configs/experiments/terl_mappo_batch01_20261009/base_lock.json`，从上述 candidate Git 对象读取 |
| **Canonical lock SHA256** | **`91dc6ea76c0f7b76ffa43441286db40b9e4ea1e9e19885a328df16ab6d024bf3`** |
| 科学 parent / TERL vendor | `bb794ca8435f06b9fa5693c0fed98f567320decf` / `143359b2722d49c29b4fecc0ad1fd8d46326e45a` |
| QA worktree | `/home/yjq/rl/CoCap1/terl-mappo-batch01-qa-20261009`；Master只读观察，未修改 |
| 中央状态 | `BATCH01_BASE_SHA=null`、`QA_SAME_SHA=PENDING`；T1/N1/R1/P1/C0及P1-control均WAITING_BASE |

Canonical 算法严格按 Core：`json.dumps(lock, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()`，默认 `ensure_ascii=True`，对整个 candidate lock 对象 SHA256。其文件字节 SHA256 是 `dff6dc7f8fd37bf7776c9e445ac81ba4aeea164bf5bafaa1361540803a1c0fda`，与canonical digest不同；中央 `batch_contract.lock.json` 的未来冻结digest也是另一个对象，当前仍null。不得混用。

Master已核验candidate可从远程delivery HEAD到达，78个lock源码hash在candidate Git对象、delivery Git对象、Core工作树均一致，T0原源码/配置不变。candidate→delivery仅新增delivery/t0_delta/t0_runtime_manifest和更新交付文档。Core JUnit确为61 passed/0 failure/error/skip、正式256/8/4 CPU/CUDA通过；这些是Core自测，**不能作为独立QA_PASS**。见[Master对象核验](../../artifacts/2026-10-09_terl_mappo_batch01/master02/core_candidate_verification.json)。

读取Core证据时使用固定delivery HEAD下的 `docs/TERL_MAPPO_BATCH01_BASE_CANDIDATE_20261009_ZH.md` 与 `artifacts/2026-10-09_terl_mappo_batch01_base/{delivery.json,validation.json,tests.xml,t0_delta.json,t0_runtime_manifest.json,recovery.json}`；不得测试delivery HEAD后声称测试了candidate。

## 2. 独立验收要求

CPU/静态 QA 可以立即继续，隐藏CUDA、单线程，测试输出使用QA自有新目录。源码与锁保持candidate原样；负测试需要修改源码时，在临时独立fixture中复现并完整登记before/after hash，不修改candidate工作树的科学文件，更不能访问已有训练目录写入。

1. **对象、依赖与锁**：自行核对candidate/delivery关系、canonical digest和78文件；记录Python/PyTorch/NumPy/SciPy/CUDA/cuDNN等实际版本与dependency锁定局限。审查静态/动态/延迟import、插件source/config delta、prepare→collect→update→save→load阶段的fail-closed边界；新增、移除、替换源码或错误pin必须拒绝。一个启动时manifest不能证明后续consumer永远已锁定。
2. **T0科学语义**：以parent/vendor实际代码与数值fixture为参照，核对MarineEnv、AW9索引/物理、dt/substeps/speed、reward逐项与顺序、native19token可见性/ordering/masks、capture判定、active<3/3001 timeout、terminated/truncated和pre-reset V bootstrap。信息可见性与网络结构分别结论；actor不读central状态。
3. **正式网络/PPO有效路径**：用256 hidden/8 heads/4 layers验证前后向、fully masked/no-target、actor/critic独立Adam、GAE/active loss/ValueNorm、dropout eval-mode似然与finite指标。核对请求设置到live consumer；错误optimizer、未生效配置、reward/action/terminal差异应拒绝。不要把缩小网络测试代替正式尺寸。
4. **完整状态resume与775k兼容性**：比较actor/V、双方Adam、ValueNorm、env含PRNG/KDTree几何、Python/NumPy/CPU/CUDA RNG、counters和下一段rollout/update。精确数值比较范围与CPU/CUDA差异必须列出；不声称pickle字节或optimizer标量device布局完全相同。历史selected775k仅只读，hash应为 `590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4`；审查真实anchor加载与strict allowlist。P1中央合同只改target_kl .02→.01；Core通用接口允许更多PPO字段并不授权P1调整LR等参数。未实际验证775k下一步连续性时明确LIMITATION，不能以synthetic resume冒充。
5. **接口与共同代码边界**：验证T1多人数/多敌方/洋流critic的3/1/0/4、4/1/1/6、7/2/2/8有效shape和full任务语义；R1 identity hook是否逐项保持raw/native reward及事件，变更奖励的component ledger是否充分；N1结构插件不扩信息；C0边界/float action consumer、必要概率measure/Jacobian接口；P1 strict resume。Core交付的是共同接口，未实现的arm adapter/shaping/continuous actor不可登记为arm QA PASS；返回后续arm QA清单即可，本轮不实现五线。
6. **评估与运行门禁**：独立复核seed集合、selection顺序、sampling RNG隔离及exception恢复、final不用作选择、full horizon/spawn、两模式50个配对物理seed、per-scene指标及counter单位。审retention是否只输出计划、不删原件；固定hash/lease断言是否真正阻止未经登记的运行。评估中parameter_updates=0。

需要特别追踪的现有问题：

- **评估种子合同差异**：candidate沿用历史T0 screen/selection/final=`2026100800/2036100800/2046101800`；中央新Batch01提案=`2056100900/2066100900/2076100900`、scene偏移100000。分别判断历史复现可行性与新held-out合法性，返回实际集合相交检查和明确的修订建议。中央新域目前是提案，不修改candidate或重算lock假装已冻结。若需要共同eval配置/代码修订，Core提交新candidate+digest并由QA重新针对同SHA验收；Master再评审。
- **QA分支测试可复现性**：Master观察到unmodified CPU suite因`create_lock()`限制Core分支而出现4个failure，使用committed-lock harness的CPU记录为59 passed/2 CUDA skipped。保留两套命令、原始日志、harness差异与测试边界；使用提交的lock验收，不放宽生产branch guard，不隐藏原失败，不把CPU CUDA-skip判为CUDA PASS。
- **延迟导入源码锁逃逸**：QA未提交的`lazy_dependency_repro.json`记录declared reward extension延迟加载未锁定`src/cocap_voradj/models/continuous/box_actor.py`，修改后collect/update/save检查仍通过，显式post-rollout coverage检查才拒绝；并记录raw rewards未保存在batch。该材料仅为Master只读观察、尚未正式验收/自行复现。请交付独立复现、严重性、实际影响、修复需求与回归要求；不要用Core61通过覆盖负测试结果。见[未验收interim回执](../../artifacts/2026-10-09_terl_mappo_batch01/master02/qa_interim_observation.json)。

## 3. QA专用有界CUDA lease

权威机器合同：[qa_smoke_lease.json](../../artifacts/2026-10-09_terl_mappo_batch01/master02/qa_smoke_lease.json)。lease ID `B01-QA-CUDA-40bd91b-20261009T151855`，发放 `2026-10-09T15:18:55.622062+08:00`，到期 **`2026-10-09T17:18:55.622062+08:00`**。允许一次尝试，启动时距到期至少10分钟；过期或资源失败登记WAIT并返回Master，CPU/静态QA照常继续，不自动续约或重试。

A0原件已核验，审计完成、未执行清理/归档。closing home261.60GiB/shared free136.77GiB，个人quota UNKNOWN；5arm74GiB/8run98GiB含30GiB reserve。这不是用户独占quota或正式并发PASS。本轮Master 15:09:02–15:09:12采样两卡util均0、未观察compute PID，GPU0 free48521MiB，RAM available106.14GiB，128CPU load1≈3.31，shared available136.76GiB；资源回执只支持有界smoke，启动仍须fresh preflight。

| 资源 / 工作 | 授权上限 |
|---|---|
| GPU | 物理GPU0，UUID `GPU-fb584eaf-0dbe-5487-d539-695f9124efb0`；`CUDA_VISIBLE_DEVICES=0`，进程内`cuda:0`；GPU1无lease |
| 进程 / CPU / evaluator | 一个QA compute进程、两个节点串行，Torch/OMP/MKL/OpenBLAS/NumExpr线程均1；evaluator workers=0 |
| 时间 / 次数 | 一次尝试；两节点CUDA总墙钟≤300s，含采样与收尾active lease≤600s；不延长、不整套GPU重跑 |
| rollout/update | 总joint environment decisions≤128、rollout PPO calls≤8；预期两节点64 decisions/5 calls，minibatch单列 |
| 内存 | Torch peak allocated≤2048MiB，自身driver VRAM≤4096MiB，RSS≤8GiB |
| 磁盘 | 自有新输出增长≤1GiB；无GIF/正式评估队列、无原checkpoint复制/覆盖、无Storage清理 |

只允许以下candidate已有节点：

```text
test/test_terl_mappo_batch01_base_20261009.py::test_formal_entry_t0_full_update_and_exact_resume[cuda:0]
test/test_terl_native_mappo_20261008.py::test_cuda_finite_update_and_rng_resume
```

QA先自行核对Git HEAD/canonical/source hashes，记录PID、UUID和所用环境。**启动前10×1秒重新采样**：GPU0无外来compute PID、util mean<60%/peak<90%、free VRAM≥12GiB、RAM available≥24GiB、load1<逻辑CPU数×.5、实际output filesystem available≥31GiB（1GiB smoke+30GiB reserve）；使用新建自有输出文件做有界write/fsync检查。quota如可查询需至少1GiB headroom，若未知则明确记录并只准本smoke；写入失败即WAIT/FAIL。该lease不排他占用设备或共享空间。

下面是**QA执行模板，Master未执行**；资源监视器需覆盖整个pytest PID树，每秒记录自身driver VRAM/RSS和磁盘增长，提前越限时只终止自身smoke。timeout仅限制时间，不能代替资源监测。先创建新QA自有输出目录并完成上述preflight，再在固定candidate工作树运行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
timeout --signal=TERM --kill-after=10s 300s python3 -m pytest -q \
  'test/test_terl_mappo_batch01_base_20261009.py::test_formal_entry_t0_full_update_and_exact_resume[cuda:0]' \
  'test/test_terl_native_mappo_20261008.py::test_cuda_finite_update_and_rng_resume' \
  --basetemp="$BATCH01_QA_NEW_OUTPUT/pytest_tmp" \
  --junitxml="$BATCH01_QA_NEW_OUTPUT/cuda_smoke.xml"
```

`BATCH01_QA_NEW_OUTPUT`须由QA设为全新、可写、自有目录；不是旧日志或原实验目录。timeout/nonfinite/OOM/越限/新外部GPU任务出现时，终止自身smoke、保留日志、释放lease，不停止任何其它用户/实验进程。不触碰训练runner/supervisor，不启动T1/N1/R1/P1/C0/P1-control，不跑并发benchmark；测试只可删除自己新生成的fixture，原件和A0清理候选均受保护。

结束后记录实际joint decisions/PPO calls/minibatches、测试结果与skip、peak allocated/reserved/driver VRAM、CPU/RSS、disk before/after、10次post GPU样本、lease释放时间。若CUDA无法执行，返回明确WAIT条件与已完成CPU/静态验收，不把GPU WAIT改成CPU BLOCKED。

## 4. QA交付与Master下一步

返回：`tested_candidate_sha / lock_sha256 / QA_verdict / report_branch / report_head / commands / dependency_versions / before_after_source_manifest / raw logs+JUnit / T0 parity / resume范围与775k限制 / 负测试 / seed reconciliation / open issues / lease_id / physical_gpu+UUID / ownPID / resources before-peak-after / released_at / limitations`。用PASS/FAIL/WAIT分别表达各门禁，科学源码SHA与QA报告commit必须分开；报告commit/push后fetch核验remote HEAD。

QA不能更新中央DAG/state、冻结BASE或授权训练。Master本轮停在等待报告：**只有收到同candidate独立验收、开放问题与seed合同处理结论，并经后续Master明确评审，才可能冻结BASE**。任何修复都走Core新candidate/lock→QA同新SHA→Master，不热换源码，不自行把本candidate或delivery HEAD填入BASE。
