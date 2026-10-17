"""Chinese reports from persistent, completed evidence; no result imputation."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/2026-10-17_z_support_full3stage'
DOC=ROOT/'docs/experiments/z_support_full3stage_20261017'
HIST=Path('/home/yjq/rl/CoCap1/iqn-z05-independent-20260923/artifacts/2026-09-23_iqn_z05_independent')
HIST_TRAIN=Path('/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages')
STAGES=('stage1','stage2','stage3')
BUDGETS=(2000000,700000,700000)

def read(p,default=None):return json.loads(p.read_text()) if p.is_file() else default

def val(v):
    if v is None:return '待完成'
    if isinstance(v,float):return f'{v:.4f}'
    return str(v)

def metric_row(stage,method,r,step):
    if not r:return f'| {stage} | {method} | {step} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |'
    s=r['summary'];c=s['capture'];m=s['mixed'];ce=s['coverage']
    alive=m.get('alive_agents',{}).get('mean','旧报告未登记')
    collision='/'.join(val(s[x]['collision_rate']) for x in ('coverage','capture','mixed'))
    return '| '+' | '.join(map(str,(stage,method,step,val(c['normal_capture_rate']),val(c['capture_seconds']['mean']),val(ce['strict_ce_rate']),val(m['capture_rate']),val(m['capture_seconds']['mean']),val(m['post_capture_ce_rate']),val(m['recovery_seconds']['mean']),collision,val(alive))))+' |'

def generate(runtime):
    DOC.mkdir(parents=True,exist_ok=True)
    state=read(runtime/'status.json',{'status':'PREPARED'})
    comparison=['# Z-Guided Support 与 Solid IQN-Z05 对照','',f'当前执行状态：`{state.get("status")}`；阶段：`{state.get("stage","待启动")}`；过程：`{state.get("phase","准备")}`。','',
    '主结果冻结为 argmax / fixed-midpoint32；sample 是32个随机 IQN 分位均值 Q 的 argmax，只作独立次要结果。各行使用原生4P1E1O、8P2E2O、12P3E3O合同。旧正式结果引用既有native批次；新最终评价seed族2026102700与selection的2026091900分离。新旧训练seed不同，不能描述为严格matched训练因果实验。队列另用新最终seed对历史正式selected做统一argmax复评；复评完成时表中使用该结果，补全逐局存活人数并验证新旧评价初态配对；此前引用既有报告并明示。','',
    '| 阶段 | 模型 | selected step | Capture normal | Capture完成均时(s) | Coverage CE | Mixed capture | Mixed捕获均时(s) | Post-capture CE | Recovery均时(s) | Collision(CE/Cap/Mix) | Mixed平均存活 |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    stage_status=[]
    for i,stage in enumerate(STAGES,1):
        sd=runtime/'stages'/stage;progress=read(sd/'training/progress.json',{})
        complete=read(sd/'training_complete.json');selection=read(sd/'selection_report.json')
        primary=read(sd/'independent/argmax/report.json');secondary=read(sd/'independent/sample/report.json')
        legacy=read(HIST/('stage1_native','stage2_native','stage3_native_run')[i-1]/'report.json')
        reference=read(sd/'reference/argmax/report.json')
        history=reference or legacy
        old_select=read(HIST_TRAIN/stage/'selection_report.json',{})
        old_step=old_select.get('selected',{}).get('step','未知')
        comparison.append(metric_row(stage,'历史Z05正式selected（统一复评）' if reference else '历史Z05正式selected（既有报告）',history,old_step))
        comparison.append(metric_row(stage,'新Z-Guided Support',primary,selection['selected']['step'] if selection else '待选择'))
        step=(complete or {}).get('environment_decision_steps',progress.get('environment_decision_steps',0))
        updates=(complete or {}).get('optimizer_updates',progress.get('optimizer_updates',0))
        episodes=(complete or {}).get('episodes',progress.get('episodes',0))
        selected=selection['selected'] if selection else None
        status='训练预算完成' if complete else '训练中' if step else ('待验收后scratch启动' if i==1 else '等待课程交接')
        if primary and secondary:status+='；两种模式正式评价完成'
        stage_status.append(dict(stage=stage,status=status,environment_decision_steps=step,optimizer_updates=updates,episodes=episodes,selected=selected))
        lines=[f'# Z-Guided Support Stage{i} 执行结果','',f'状态：**{status}**。未完成任务保留待完成状态，不把预算或运行中状态当作结果。','',
        f'规模：{(4,8,12)[i-1]} Pursuers / {i} Evaders / {i} Obstacles。预算：{BUDGETS[i-1]:,} environment decision steps。',
        f'已持久化进度：{step:,}环境决策；{updates:,}次optimizer更新；{episodes:,}个episode。三者独立计数。','',
        f'初始化：{"scratch随机初始化，禁止历史或pilot权重" if i==1 else "只加载本线前一阶段selected模型；optimizer/replay/env/Z/RNG新建"}。',
        f'配置：`{sd/"resolved.yaml"}`；训练日志：`{sd/"training"}`。','',
        '## 所选模型与交接','','```json\n'+json.dumps(selected,ensure_ascii=False,indent=2)+'\n```' if selected else '待全部100k selection checkpoint评价完成后，按原BalancedFloor及原tie-break顺序选择；没有预定旧Z05最佳步数。','',
        f'交接Gate：`{sd/"handoff_gate.json"}`；下一阶段：{STAGES[i] if i<3 else "最终课程候选"}。',
        f'训练完成凭据：`{sd/"training_complete.json"}`。GIF渲染不作为下一阶段启动前置条件。','',
        '## 正式独立评价与可视化','',
        f'主模式：`{sd/"independent/argmax/report.json"}`；次模式：`{sd/"independent/sample/report.json"}`。',
        '每模式三场景各20固定seed、全部episode纳入统计；60 trajectory、60 event日志；每场景固定偶数index共10 GIF，选取与成功或失败无关。每份GIF登记seed、源轨迹SHA、renderer参数和命令。失败任务最多3次尝试，未成功项目保留在failure_retry_registry和队列中。','']
        if primary or secondary:
            lines+=['| 推理模式 | 场景 | Normal capture | CE success/recovery | Collision | 平均存活 |','|---|---|---:|---:|---:|---:|']
            for mode,result in (('argmax主结果',primary),('sample次结果',secondary)):
                if not result:continue
                for scene in ('coverage','capture','mixed'):
                    ss=result['summary'][scene]
                    ce=ss.get('strict_ce_rate',ss.get('post_capture_ce_rate'))
                    lines.append('| '+' | '.join((mode,scene,val(ss.get('normal_capture_rate',ss.get('capture_rate'))),val(ce),val(ss['collision_rate']),val(ss['alive_agents']['mean'])))+' |')
            lines+=['']
        if primary:
            lines+=['```json',json.dumps(primary['summary'],ensure_ascii=False,indent=2),'```','']
        else:lines+=['正式结果尚未产生；不填写推测性成功率。','']
        lines+=['## 产物登记','',f'服务器原始目录：`{sd}`。代码库轻量registry见`artifacts/2026-10-17_z_support_full3stage/`。','']
        (DOC/f'Z_SUPPORT_STAGE{i}_RESULT_ZH.md').write_text('\n'.join(lines))
    comparison+=['','完成时间仅在达成事件的episode上统计，同时登记成功分母、失败与删失，不能解释为无条件速度优势。旧全量报告没有逐局存活人数，明确记为未登记；不从GIF子样本或碰撞率反推。','',
    'Repeated-arrival适应性：下一步在本线Stage3 selected冻结后，复用既有strict repeated-arrival协议与held-out seed检验。当前完整课程不自动扩展为多波研究；没有数据时不声称已改善。','',
    '限制：当前旧门控使Support常选Z=1友机，Coverage混合权重常为0；多跳Z只影响policy observation，不自动改变旧角色；动态CE权重保留原PBRS项但不具有全程固定potential望远镜保证。单训练seed不能证明方法优越。','',
    f'历史报告：`{HIST}`；所有失败阶段及全曲线保留。']
    (DOC/'Z_SUPPORT_VS_SOLID_Z05_COMPARISON_ZH.md').write_text('\n'.join(comparison)+'\n')
    live=read(runtime/'live_processes.json',{})
    (DOC/'Z_SUPPORT_EXECUTION_SUMMARY_ZH.md').write_text('# 中文执行总结\n\n'+f'当前状态：{state.get("status")}，{state.get("phase","准备")}。\n\n'+'\n'.join(f'- {s["stage"]}：{s["status"]}；{s["environment_decision_steps"]:,}环境决策，{s["optimizer_updates"]:,}次更新，{s["episodes"]:,}局。' for s in stage_status)+'\n\n'+f'监督器PID：{live.get("supervisor_pid","未启动")}；训练PID：{live.get("training_pid","未启动")}；物理GPU：{live.get("gpu","待分配")}。\n'+''.join(f'\n{s}吞吐/ETA：'+str(read(runtime/"stages"/s/"training/progress.json",{}).get("steps_per_second","待测"))+' decisions/s；'+str(read(runtime/"stages"/s/"training/progress.json",{}).get("eta_seconds","待测"))+'秒。' for s in STAGES if (runtime/"stages"/s/"training/progress.json").is_file())+'\n\n下一检查点：当前阶段的下一个100k模型快照；本地每1000决策进度刷新，监督器每20秒导出状态。\n\n验收通过后的课程已获用户授权，监督器自动按序训练和交接。仅合同、数值、资源安全或训练失败进入安全暂停。选模需要的评价仅在阶段边界签收；环境训练循环不运行评价。\n\n'+f'持久化状态：`{runtime/"status.json"}`。大文件留服务器，Git仅同步代码、合同、指标及registry。\n')
