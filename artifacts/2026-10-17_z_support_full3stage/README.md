# 轻量实验登记

`pilot_independent_audit.json`核验19个checkpoint SHA、900对初态及训练状态；`engineering_provenance.json`记录原核心代码SHA、历史selected、配置合同和资源；`gpu_scratch_numeric_smoke.json`只作正式结构GPU数值诊断，不能当作课程结果。

`acceptance_gate.json`是启动签收凭据，未签收不得启动。`training_state_manifest.json`、`checkpoint_registry.json`、`evaluation_queue.json`、`artifact_registry.json`、`failure_retry_registry.json`由监督器和worker从SQLite原始账本导出。`training_curves.csv/png`保留全程进度和更新数。SQLite与大型模型/轨迹/GIF在服务器runtime，不上传GitHub。

验收测试保留JUnit XML及stdout。CPU重评与旧GPU轨迹末值未精确相同的失败诊断也保留；其解释必须依据GPU原路径复现和同输入CPU/CUDA精度检查，不能掩盖为全部通过。
