# 参考入口：从仓库根使用 uv run python -m experiments.movielens.pipeline.train_single_auxiliary 启动。
# 保留历史实验的严格对齐检查；需要对应数据/schema及历史run，不是下载即跑的独立教程。
"""ML1M仅左/右辅助：单/双×左右×0.1/1×三seed，完整单epoch。"""
from contextlib import redirect_stdout,redirect_stderr
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from rosen.core.train.context import TrainerComponents,TrainerDataLoaders
from experiments.movielens.pipeline.prepare_sessions import WORKSPACE,sha
from experiments.movielens.pipeline.current_schema import manager_at
from experiments.movielens.pipeline.single_auxiliary import SingleAuxiliaryModel
from experiments.movielens.pipeline.train_current_epoch1 import load_batches,graph_audit,CurrentTrainer
from experiments.kuairand.pipeline.exit_training import state_hash,joint_parameter_groups,predict_batches
from experiments.kuairand.pipeline.run_epochwise_exit import write_json


def run(output):
    torch.set_num_threads(4)
    reference = WORKSPACE/'runs/plan22_02_current_epoch1'
    old = json.loads((reference/'config.json').read_text())
    data = Path(old['data']);manager = manager_at(data/'schema',load=True)
    batches = load_batches(data,manager)
    for split,loader in batches.items():
        actual = torch.cat([b['request_id'] for b in loader]).numpy()
        expected = pl.read_parquet(data/f'{split}.parquet',columns=['request_id'])['request_id'].to_numpy()
        assert np.array_equal(actual,expected)
        assert all(torch.count_nonzero(b['user_id'])==0 for b in loader)
    sources = {**old['source_hashes'],**{str(p.resolve()):sha(p) for p in [
        Path(__file__),Path(__file__).with_name('single_auxiliary.py')]}}
    assert all(sha(p)==h for p,h in sources.items())
    matrix = [(name,side,w) for name in ['single','mixture'] for side in ['left','right'] for w in [.1,1.]]
    config = {**old,'matrix':matrix,'source_hashes':sources,'reference':str(reference),
        'variants':'only one auxiliary; .1 capped at .1 joint; 1 uncapped; joint to both towers',
        'new_trajectories':24}
    write_json(output/'config.json',config)
    summary = {}
    for seed in old['seeds']:
        for name,side,weight in matrix:
            key = f'{seed}/{name}/{side}_{weight:g}';dest = output/key;dest.mkdir(parents=True,exist_ok=False)
            torch.manual_seed(seed);np.random.seed(seed)
            options = dict(mixture=name=='mixture',side=side,weight=weight,
                requests_per_session=old['coverage']['train']['mean_steps'])
            model = SingleAuxiliaryModel(manager,**options)
            initial = state_hash(model)
            ref = json.loads((reference/f'{seed}/{name}/none/initialization.json').read_text())
            assert initial==ref['state_sha256']
            groups = joint_parameter_groups(model)
            meta = [dict(scope=g['scope'],lr=g['lr'],weight_decay=g['weight_decay'],tensors=len(g['params'])) for g in groups]
            assert meta==ref['optimizer_groups']
            audit = graph_audit(model,batches['train'][0].interaction)
            assert state_hash(model)==initial
            write_json(dest/'initialization.json',dict(state_sha256=initial,original_common_initialization_exact=True,
                within_class_loss_variants_exact=True,optimizer_groups=meta,**audit))
            uid = [m.emb_modules['user_id'].embedding.weight.detach().clone() for m in [model.omni_embedding,model.budget_embedding]]
            cfg = SimpleNamespace(device='cpu',max_epochs=1,ckpt_dir=str(dest),experiment_name=f'ml1m_{seed}_{name}_{side}_{weight}',
                dataset_name='ml1m_current_information_v2',train_log_interval=100)
            trainer = CurrentTrainer(model,torch.optim.AdamW(groups,weight_decay=0),manager,
                TrainerDataLoaders(batches['train'],batches['valid'],batches['test']),TrainerComponents(),cfg,output=dest)
            print('START',key,audit,flush=True);trainer.run()
            assert trainer.global_step==196 and len(trainer.records)==1
            assert all(torch.equal(before,m.emb_modules['user_id'].embedding.weight)
                for before,m in zip(uid,[model.omni_embedding,model.budget_embedding]))
            with torch.random.fork_rng():recovered = SingleAuxiliaryModel(manager,**options)
            recovered.load_state_dict(torch.load(dest/'epoch_01/checkpoint.pt',map_location='cpu',weights_only=False)['model_state_dict'])
            assert predict_batches(recovered,batches['test']).equals(pl.read_parquet(dest/'epoch_01/test_predictions.parquet'))
            write_json(dest/'complete.json',dict(epochs=1,steps=196,uid_unchanged=True,full_test_reload_exact=True))
            summary[key] = trainer.records;write_json(output/'summary.json',summary)
            print('TRAJECTORY_COMPLETE',key,flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    write_json(output/'complete.json',dict(trajectories=24,full_test_evaluations=24,
        all_full_test_checkpoints_reloaded_exact=True,source_hashes_unchanged=True))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    output = WORKSPACE/'runs/plan22_02_single_aux_epoch1'
    output.mkdir(parents=True,exist_ok=False);(output/'logs').mkdir()
    print('日志：',output/'logs/train.log',flush=True)
    with (output/'logs/train.log').open('w') as f,redirect_stdout(f),redirect_stderr(f):run(output)
