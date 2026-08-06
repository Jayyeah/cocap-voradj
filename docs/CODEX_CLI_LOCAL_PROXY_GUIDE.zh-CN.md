# Codex CLI 与本地 Mihomo 代理使用说明

本文记录当前服务器上继续使用 Codex CLI 与本地 Clash/Mihomo 代理的手动操作方式。订阅 URL、节点详情和 `config.yaml` 内容属于敏感信息，不写入本文，也不要提交到 GitHub。

## 当前本地约定

- Codex CLI：`/home/yjq/.local/bin/codex`
- Mihomo core：`/home/yjq/.local/bin/mihomo`
- Mihomo 配置目录：`/home/yjq/.config/mihomo`
- 本地代理端口：`127.0.0.1:17892`
- Mihomo 控制端口：`127.0.0.1:19090`
- Mihomo tmux：`mihomo_yjq`
- Mihomo 日志：`/home/yjq/.config/mihomo/mihomo.log`

旧的 `127.0.0.1:17891` 通常来自远程连接端口映射。需要长时间无人值守时，应优先使用服务器本地的 `17892`，避免本地电脑断开后 Codex CLI 失去网络。

## 启动和检查 Mihomo

检查是否已经在跑：

```bash
tmux ls
ss -ltnp | grep -E ':(17892|19090)'
tail -80 /home/yjq/.config/mihomo/mihomo.log
```

如果 `mihomo_yjq` 不存在，手动启动：

```bash
tmux new-session -d -s mihomo_yjq \
  "cd /home/yjq/.config/mihomo && \
   env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
       -u http_proxy -u https_proxy -u all_proxy \
       /home/yjq/.local/bin/mihomo -d /home/yjq/.config/mihomo \
       2>&1 | tee /home/yjq/.config/mihomo/mihomo.log"
```

验证配置文件：

```bash
/home/yjq/.local/bin/mihomo -t -d /home/yjq/.config/mihomo
```

验证不依赖旧端口映射、只走 `17892` 能访问 OpenAI：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    curl -I --connect-timeout 12 \
    --proxy http://127.0.0.1:17892 \
    https://api.openai.com
```

如果配置测试报 `Country.mmdb` 或 GeoIP 下载失败，说明 Mihomo 初始化需要本地 Geo 数据。当前已在 `/home/yjq/.config/mihomo` 放置过所需文件；若迁移新机器，先用可用网络补齐 Geo 数据，再启动本地代理。

## Codex CLI 手动使用

先确认当前命中的 Codex 是新版：

```bash
type -a codex
command -v codex
codex --version
```

期望第一项为：

```text
/home/yjq/.local/bin/codex
```

只读探针：

```bash
cd /home/yjq/rl/CoCap1/cocap-voradj

HTTP_PROXY=http://127.0.0.1:17892 \
HTTPS_PROXY=http://127.0.0.1:17892 \
ALL_PROXY=http://127.0.0.1:17892 \
codex exec \
  -C /home/yjq/rl/CoCap1/cocap-voradj \
  --sandbox read-only \
  -m gpt-5.6-sol \
  -c 'model_reasoning_effort="high"' \
  "只检查当前目录，输出 pwd 和 git status --short 摘要，不要修改文件。"
```

后台任务推荐使用 tmux，并显式指定代理：

```bash
mkdir -p /home/yjq/rl/CoCap1/cocap-voradj/artifacts/codex_background

cat > /tmp/codex_task_prompt.txt <<'EOF'
这里写本次后台任务说明。
要求写清工作目录、可写范围、禁止事项、何时停止、需要更新哪些文档。
EOF

tmux new-session -d -s codex_task_name \
  "cd /home/yjq/rl/CoCap1/cocap-voradj && \
   env HTTP_PROXY=http://127.0.0.1:17892 \
       HTTPS_PROXY=http://127.0.0.1:17892 \
       ALL_PROXY=http://127.0.0.1:17892 \
       http_proxy=http://127.0.0.1:17892 \
       https_proxy=http://127.0.0.1:17892 \
       all_proxy=http://127.0.0.1:17892 \
       codex exec \
         -C /home/yjq/rl/CoCap1/cocap-voradj \
         --add-dir /home/yjq/rl/CoCap1/TERL/docs \
         --sandbox workspace-write \
         -m gpt-5.6-sol \
         -c 'model_reasoning_effort=\"high\"' \
         \"\$(cat /tmp/codex_task_prompt.txt)\" \
       2>&1 | tee /home/yjq/rl/CoCap1/cocap-voradj/artifacts/codex_background/codex_task_name.log"
```

常用查看命令：

```bash
tmux ls
tmux attach -t codex_task_name
tail -160 /home/yjq/rl/CoCap1/cocap-voradj/artifacts/codex_background/codex_task_name.log
pgrep -af 'codex exec|mihomo'
```

退出 tmux 查看但不关闭任务：按 `Ctrl-b`，再按 `d`。

## 权限原则

- 探针、状态检查、文档阅读：优先 `--sandbox read-only`。
- 需要更新项目文档、写日志、保存产物：使用 `--sandbox workspace-write`。
- 需要读取或写入项目外目录：使用 `--add-dir <路径>` 明确加入，例如 `--add-dir /home/yjq/rl/CoCap1/TERL/docs`。
- 不使用 `--yolo` 或无限制沙箱。
- 不在后台提示词中授权 `git push`、删除大目录、重置 Git、停止无关训练，除非本次明确需要。
- Codex CLI `0.146.0` 中 `codex exec --ask-for-approval ...` 会报 unexpected argument；当前后台任务可用默认 `approval: never`。若后续版本需要显式设置，可先用只读探针测试 `-c approval_policy=never`。

## VS Code 插件端 Codex 注意事项

VS Code 插件端的 Codex 不一定自动走 `17892`。插件进程继承的是 VS Code Server 或插件启动时的环境变量，而不是某个终端里后来执行的 `export`。

检查插件端 Codex 进程：

```bash
pgrep -af 'openai.chatgpt|codex'
```

查看某个 PID 是否继承了 `17892`：

```bash
tr '\0' '\n' < /proc/<PID>/environ | grep -i proxy
```

只有看到 `HTTP_PROXY`、`HTTPS_PROXY` 或 `ALL_PROXY` 指向 `http://127.0.0.1:17892`，才能认为插件端也切到了本地代理。

如果插件端仍是 `17891`，可按以下顺序尝试：

1. 确认 `mihomo_yjq` 已经在跑，并且 `17892` 能访问 OpenAI。
2. 在远程 shell 的启动配置中加入 `17892` 代理环境变量，例如 `~/.bashrc` 或 `~/.profile`。
3. 在 VS Code 中执行 `Developer: Reload Window`。
4. 若仍未继承，执行 `Remote-SSH: Kill VS Code Server on Host` 后重新连接。
5. 重新用 `/proc/<PID>/environ` 检查插件端 Codex 进程。

对长时间无人值守任务，仍建议使用 CLI + tmux，并在 tmux 命令里显式写入 `17892` 代理。这样不会依赖 VS Code 插件进程是否正确继承环境。

## 断开远程连接前检查清单

```bash
tmux ls
ss -ltnp | grep 17892
tail -40 /home/yjq/.config/mihomo/mihomo.log
pgrep -af 'codex exec|python3 -m cocap_voradj.training.trainer|evaluate_ce_coverage'
nvidia-smi
```

判断标准：

- `mihomo_yjq` 存在。
- `127.0.0.1:17892` 正在监听。
- Codex 后台命令中显式包含 `HTTP_PROXY=http://127.0.0.1:17892`。
- Mihomo 日志中有 `chatgpt.com` 或 `api.openai.com` 的连接记录。
- 训练进程和评估进程符合预期，没有重复启动同名任务。

## 公开仓库注意

可以提交本文，因为它不包含订阅 URL 或节点信息。不要提交以下内容：

- `/home/yjq/.config/mihomo/config.yaml`
- `/home/yjq/.config/mihomo/config.raw.yaml`
- 订阅 URL
- 节点名称、密码、token、secret
- Codex auth 文件或 OpenAI API key
