# 参考入口：从仓库根使用 uv run python -m experiments.kuairand.pipeline.run_current_feedback 启动。
# 保留历史实验的严格对齐检查；需要对应数据/schema及历史run，不是下载即跑的独立教程。
"""当前刷4反馈率：单/混合弱双辅助及Fusion，三seed、5epoch完整训练评估。

pace均为当前刷纯时间戳；主Loss回传双塔，辅助合计<=10%。
复用已完成current双辅助无反馈6轨迹；新跑双塔有反馈6、Fusion开/关6。
"""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from rosen.core.train.context import TrainerComponents,TrainerDataLoaders
from experiments.kuairand.pipeline.exit_training import WORKSPACE,EXPERIMENTS,state_hash,joint_parameter_groups
from experiments.kuairand.pipeline.current_feedback import (
    FEEDBACK_FIELDS,CurrentFeedbackExitModel,CurrentFeedbackFusion,load_feedback_batches)
from experiments.kuairand.pipeline.limited_auxiliary_exit import LimitedAuxiliaryExitModel
from experiments.kuairand.pipeline.exit_model import KRExitModel
from experiments.kuairand.pipeline.run_pace_completion import CompletionTrainer
from experiments.kuairand.pipeline.kr_request_schema import request_manager
from experiments.kuairand.pipeline.uid import zero_uid_inputs,uid_tables,tensor_hash
from experiments.kuairand.pipeline.run_epochwise_exit import write_json
from experiments.kuairand.pipeline.build_timestamp_data import sha

MATRIX=[('single_step_pace',True),('mixture_step_pace',True),('fusion_pace',False),('fusion_pace',True)]


def build_model(manager,name,enabled,normalization):
    if name=='fusion_pace':return CurrentFeedbackFusion(manager,**EXPERIMENTS[name],use_current_feedback=enabled)
    return CurrentFeedbackExitModel(manager,**EXPERIMENTS[name],use_current_feedback=enabled,
        left_weight=.1,right_weight=.1,max_aux_ratio=.1,requests_per_session=normalization)


def shared_hash(model):
    import hashlib
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if name.startswith('feedback_projection.'):continue
        digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def smoke(model,manager,name,batch,normalization):
    """同权重真实train batch：初值精确恒等，关闭反馈的共享梯度精确相等。"""
    with torch.random.fork_rng():
        old=(KRExitModel(manager,**EXPERIMENTS[name]) if name=='fusion_pace' else
            LimitedAuxiliaryExitModel(manager,**EXPERIMENTS[name],left_weight=.1,right_weight=.1,
                max_aux_ratio=.1,requests_per_session=normalization))
    old.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('feedback_projection.')},strict=True)
    assert torch.equal(old.predict(batch),model.predict(batch))
    assert torch.equal(old.calculate_loss(batch),model.calculate_loss(batch))
    enabled=model.use_current_feedback;model.use_current_feedback=False
    oldgrads=torch.autograd.grad(old.calculate_loss(batch),list(old.parameters()),allow_unused=True)
    shared=[p for k,p in model.named_parameters() if not k.startswith('feedback_projection.')]
    newgrads=torch.autograd.grad(model.calculate_loss(batch),shared,allow_unused=True)
    for a,b in zip(oldgrads,newgrads):
        if a is None:assert b is None
        else:assert torch.equal(a,b)
    model.use_current_feedback=enabled
    grad=torch.autograd.grad(model.calculate_loss(batch),model.feedback_projection.weight,allow_unused=True)[0]
    assert (grad is not None and float(grad.norm())>0) if enabled else grad is None
    assert all(p.grad is None for p in model.parameters())
    return dict(initial_prediction_loss_exact=True,disabled_shared_gradients_exact=True,
        projection_initial_gradient_norm=None if grad is None else float(grad.norm()))


def run(output):
    torch.set_num_threads(4)
    data=WORKSPACE/'data/v6_current_feedback'
    dm=json.loads((data/'manifest.json').read_text())
    reference=WORKSPACE/'runs/plan22_previous_pace'
    rc=json.loads((reference/'config.json').read_text())
    sources={**dm['source_hashes'],**rc['source_hashes'],**{str(p.resolve()):sha(p) for p in [
        Path(__file__),data/'manifest.json',Path(__file__).with_name('current_feedback.py'),
        Path(__file__).with_name('run_pace_completion.py'),Path(__file__).with_name('pace_completion.py'),
        *[data/f'{s}.parquet' for s in ['train','valid','test']],reference/'summary.json']}}
    assert all(sha(p)==h for p,h in sources.items())
    manager=request_manager(data/'schema',load=True)
    batches=zero_uid_inputs(load_feedback_batches(data,manager))
    coverage=json.loads((reference/'coverage.json').read_text())
    for split,loader in batches.items():
        ids=torch.cat([b['request_id'] for b in loader]).numpy()
        expected=pl.read_parquet(data/f'{split}.parquet',columns=['request_id'])['request_id'].to_numpy()
        assert np.array_equal(ids,expected) and len(ids)==coverage[split]['rows']
        assert all(torch.count_nonzero(b['user_id'])==0 for b in loader)
        assert all(torch.isfinite(b[c]).all() and ((b[c]>=0)&(b[c]<=1)).all() for b in loader for c in FEEDBACK_FIELDS)
    write_json(output/'coverage.json',coverage)
    config={**rc,'data':str(data),'matrix':MATRIX,'source_hashes':sources,'pace_scopes':['current'],
        'current_feedback':FEEDBACK_FIELDS,'feedback_transformation':'current six binary event counts / 6; Float32',
        'feedback_input':'only content/Fusion first hidden preactivation; budget unchanged',
        'projection':'zero-initial 4x64 matrix; no bias; residual to original first affine; exact baseline function',
        'reference_without_feedback':str(reference),'new_trajectories':12,'reused_trajectories':6,
        'training_strategy':'joint BCE to both towers; Ours weak left NLL/right BCE auxiliaries <=10%; Fusion BCE',
        'warm_start':False,'prediction_boundary':'after current screen feedback becomes available'}
    write_json(output/'config.json',config)
    summary={};nps=coverage['train']['requests_per_session']
    for seed in [2026,2027,2028]:
        for name,enabled in MATRIX:
            scope='on' if enabled else 'off';dest=output/f'{seed}/{name}/{scope}';dest.mkdir(parents=True,exist_ok=False)
            torch.manual_seed(seed);np.random.seed(seed)
            model=build_model(manager,name,enabled,nps)
            initial=state_hash(model)
            ref=json.loads((WORKSPACE/f'runs/epochwise5_uid_zero/{seed}/{name}/initialization.json').read_text())
            assert shared_hash(model)==ref['state_sha256']
            groups=joint_parameter_groups(model)
            gmeta=[dict(scope=g['scope'],lr=g['lr'],weight_decay=g['weight_decay'],tensors=len(g['params'])) for g in groups]
            expected=[dict(g) for g in ref['optimizer_groups']]
            for group in expected:
                if group['scope']=='other' and group['weight_decay']==1e-6:group['tensors']+=1
            assert gmeta==expected
            rng=torch.get_rng_state().clone()
            audit=smoke(model,manager,name,batches['train'][0].interaction,nps)
            assert initial==state_hash(model) and torch.equal(rng,torch.get_rng_state())
            uid_initial={n:tensor_hash(t.weight) for n,t in uid_tables(model).items()}
            write_json(dest/'initialization.json',dict(state_sha256=initial,shared_state_sha256=shared_hash(model),
                optimizer_groups=gmeta,uid_initial=uid_initial,**audit))
            cfg=SimpleNamespace(device='cpu',max_epochs=5,ckpt_dir=str(dest),
                experiment_name=f'current_feedback_{seed}_{name}_{scope}',dataset_name='kr_v6_current_feedback',train_log_interval=200)
            trainer=CompletionTrainer(model,torch.optim.AdamW(groups,weight_decay=0),manager,
                TrainerDataLoaders(batches['train'],batches['valid'],batches['test']),TrainerComponents(),cfg,output=dest)
            print('START',seed,name,scope,'audit',audit,flush=True)
            trainer.run()
            assert trainer.global_step==2105 and len(trainer.records)==5
            assert uid_initial=={n:tensor_hash(t.weight) for n,t in uid_tables(model).items()}
            if not enabled:assert torch.count_nonzero(model.feedback_projection.weight)==0
            else:assert model.feedback_projection.weight.norm()>0
            write_json(dest/'complete.json',dict(epochs=5,steps=2105,uid_unchanged=True,restored_best=False,
                feedback_enabled=enabled,feedback_projection_norm=float(model.feedback_projection.weight.norm())))
            summary[f'{seed}/{name}/{scope}']=trainer.records
            write_json(output/'summary.json',summary)
            print('TRAJECTORY_COMPLETE',seed,name,scope,flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    write_json(output/'complete.json',dict(trajectories=len(summary),test_evaluations=60,valid_evaluations=60,
        source_hashes_unchanged=True,shared_initialization_exact=True,disabled_gradient_exact=True))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=WORKSPACE/'runs/plan22_current_feedback')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);(args.output/'logs').mkdir()
    print('训练日志：',args.output/'logs/train.log',flush=True)
    with (args.output/'logs/train.log').open('w') as f,redirect_stdout(f),redirect_stderr(f):run(args.output)
