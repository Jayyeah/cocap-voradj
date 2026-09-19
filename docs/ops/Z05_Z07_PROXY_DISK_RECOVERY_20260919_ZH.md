# Z05/Z07 Proxy 与磁盘恢复记录

日期：2026-09-19（Asia/Shanghai）

## Proxy

- Root cause：小写 `http_proxy`、`https_proxy`、`all_proxy` 仍指向失效的 `127.0.0.1:17891`。
- Active mihomo mixed-port：`127.0.0.1:17892`，已由 `ss` 确认监听。
- Git config 未发现固定的 `http.proxy`/`https.proxy`。
- 未修改 `~/.bashrc`、global Git config 或系统 proxy；永久配置修改：**NO**。
- `git fetch --all --prune` 已用命令级临时 proxy 成功完成。

临时 Git 命令模板：

```bash
env http_proxy=http://127.0.0.1:17892 https_proxy=http://127.0.0.1:17892 \
  HTTP_PROXY=http://127.0.0.1:17892 HTTPS_PROXY=http://127.0.0.1:17892 \
  all_proxy=socks5://127.0.0.1:17892 ALL_PROXY=socks5://127.0.0.1:17892 \
  git -c http.proxy=http://127.0.0.1:17892 \
      -c https.proxy=http://127.0.0.1:17892 push
```

## Disk

- Before：根盘 `/dev/nvme0n1p2` 915G total，57G free，94% used。
- After：约 90G free，90% used；inode 约 6% used。
- Observed filesystem reclaim：约 33G。
- Logical replay paths deleted：约 60.11 GiB；重复 replay 路径使 logical bytes 与 `df` 实际回收不同。
- 清理内容：历史 legacy artifacts 的 replay、两个已结束历史 worktree 的大 replay；明确可重建 cache 只做审计，未做高风险 VS Code 清理。
- 保留：当前 Z05/Z07 状态、resume/checkpoint、所有 selected/final/milestone checkpoint、configs、docs、tests、reports、manifests。
- `/data/disk2` 本身余量约 4.5T，但 `/data/disk2/home` 为 root 所有，当前用户无法创建既定 `cocap-runs` 路径；未使用 sudo。

## Runtime

- Formal branch：`experiment/iqn-z-unified-decay-dual-curriculum-20260919`。
- Preflight：`PASS`；Z05/Z07 only scientific difference remains `lambda=eta=0.5/0.7`，both `hard_zero_threshold=0.10`。
- Existing supervisor initially blocked at `waiting_for_storage` because the configured `/data/disk2/.../cocap-runs/...` path did not exist and was not writable。
- GPUs：GPU0/GPU1 当前被另一用户的 Isaac Sim PID `3248997` 占用；未停止或干扰该进程。
- Fallback runtime：使用用户可写的 sibling runtime directory on root disk，supervisor `--min-free-gib 50`；这只改变存储位置/门槛，不改变训练或科学合同。

## Watchdog

Supervisor 新增 root filesystem watchdog：warning `<50 GiB`，critical `<20 GiB`。warning 写入 status；critical `failed_closed`，不自动删除 live replay，也不修改 replay capacity、stage budget、LR、epsilon 或其它科学参数。既有 `/data/disk2` storage gate `200 GiB` 保留。

## Git sync

Fetch 使用 17892 临时 proxy 成功。提交后 push 最多尝试两次；若两次均失败，记录 `REMOTE_SYNC_PENDING` 和人工命令，不阻塞本地训练。
