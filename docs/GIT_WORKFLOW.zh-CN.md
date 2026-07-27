# GitHub 上传与迭代工作流

本文档记录 `cocap-voradj` 正式项目的 Git/GitHub 使用方案。核心目标是：代码可追踪、云端可备份、服务器可同步，同时避免把大规模训练输出和临时调试文件误传到仓库。

## 1. 当前仓库边界

建议纳入 GitHub 的内容：

- `src/cocap_voradj/`：核心环境、动力学、APF、IQN、replay、trainer。
- `configs/experiments/`：当前主线 A3 配置和课程配置。
- `configs/smoke/`：短训练 smoke test 配置。
- `tools/`：rollout、并行评估、checkpoint screening、课程监督工具。
- `train.py`、`pyproject.toml`、`requirements.txt`、`Dockerfile`。
- `README.md`、`README.zh-CN.md`、`PROJECT_STRUCTURE.md`、本文件。
- `artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum/best_20rollout10gif/` 中精选的 A3 best checkpoint、对应配置、summary，以及每阶段少量 mix GIF。

默认不纳入 GitHub 的内容：

- `runs/`、`logs/`、`results/`、`wandb/`。
- 临时 rollout、screening、validation 目录。
- 大批量 GIF、episode JSON、jsonl 日志、未筛选 checkpoint。
- 原主项目的核心内部文档；后续如果需要公开，先单独整理脱敏版。

## 2. 首次上传方案

推荐远端仓库名：`cocap-voradj`。

在 GitHub 网页端创建空仓库：

- owner：`Jayyeah`
- repository：`cocap-voradj`
- visibility：按需要选择 private 或 public
- 不勾选自动生成 README、LICENSE、.gitignore，避免和本地文件冲突

本地首次提交：

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj

git status --short --untracked-files=all

git add README.md README.zh-CN.md PROJECT_STRUCTURE.md docs .gitignore LICENSE Dockerfile pyproject.toml requirements.txt train.py src configs tools artifacts

git status --short

git commit -m "Initial cocap-voradj release"
```

连接远端并上传：

```bash
git remote add origin git@github.com:Jayyeah/cocap-voradj.git
# 或使用 HTTPS：git remote add origin https://github.com/Jayyeah/cocap-voradj.git

git push -u origin main
```

如果你选择 private 仓库，后续服务器拉取需要配置 SSH key 或 GitHub token。

## 3. 日常迭代流程

每次改代码前：

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj
git status --short
git pull --ff-only
```

完成一次小改动后，先做轻量检查：

```bash
python3 -m compileall -q train.py src tools
python3 train.py --help
python3 tools/batch_rollouts.py --help
```

涉及训练逻辑时，建议额外跑 smoke test：

```bash
python3 train.py --config configs/smoke/a3_4v1_cpu_smoke.yaml --device cpu
```

确认无误后提交：

```bash
git status --short
git add <changed-files>
git commit -m "Describe the change briefly"
git push
```

## 4. 实验产物更新规则

训练产生的新线默认先留在 `runs/` 或外部 milestone，不直接进入 GitHub。

只有当某条线被确认有保留价值时，再精选同步：

- best checkpoint：只保留最终选定的少量 `.pt`。
- 配置：必须复制 checkpoint 对应的原始 YAML。
- summary：保留 `all_summaries.json` 和关键场景 `batch_summary.json`。
- GIF：默认每阶段只保留 mix 的 3 个代表性 GIF；额外 GIF 需要明确说明原因。
- README：在对应 artifact 目录更新 checkpoint 来源、训练配置、性能摘要。

同步前检查 `.gitignore` 是否放行了需要上传的 checkpoint/GIF：

```bash
git check-ignore -v path/to/file.pt || true
git status --short --untracked-files=all
```

如果文件仍被忽略，优先增加精确 `!` 例外，不要取消全局 `*.pt` 或 `*.gif` 忽略规则。

## 5. 服务器/新机器拉取方案

首次拉取：

```bash
cd /home/yjq/rl/CoCap1
git clone git@github.com:Jayyeah/cocap-voradj.git
cd cocap-voradj
pip install -r requirements.txt
python3 train.py --help
```

日常同步：

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj
git pull --ff-only
```

如果服务器上有未提交本地改动：

```bash
git status --short
```

先判断这些改动是否需要保留。需要保留则提交或 stash；不需要时再丢弃。不要直接 `git reset --hard`，除非明确确认这些本地改动都可以删除。

## 6. 分支建议

短期可以只用 `main`，每次提交保持小而清晰。

当后续开始多条算法线并行时，建议使用短分支：

```bash
git checkout -b exp/a4-cell-center-speed
# 修改、测试、提交
git push -u origin exp/a4-cell-center-speed
```

确认有效后再合入 `main`；无效实验可以保留分支记录，但不把大规模产物合入主分支。

## 7. 推荐提交粒度

推荐按下面类型分开提交：

- `code:` 核心环境、reward、APF、trainer、model 改动。
- `config:` 新实验配置或课程配置。
- `tools:` rollout、screening、supervisor 工具改动。
- `artifact:` 精选 checkpoint/GIF/summary 更新。
- `docs:` README、实验说明、工作流文档。

这样后续回滚或定位问题会轻松很多。
