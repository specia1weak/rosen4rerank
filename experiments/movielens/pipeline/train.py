# 参考入口：从仓库根使用 uv run python -m experiments.movielens.pipeline.train 启动。
# 保留历史实验的严格对齐检查；需要对应数据/schema及历史run，不是下载即跑的独立教程。
"""ML1M完整训练：Fusion/单/混合，UID0，上一评分pace，三seed五轮。

复用rosen ParquetStreamDataset、原生BaseTrainer训练step及KR逐epoch评估器。
每轮测完整valid/test，不恢复best，不按test选epoch。所有结果独立落盘。
"""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from rosen.core.train.context import TrainerComponents,TrainerDataLoaders
from experiments.movielens.pipeline.prepare_sessions import WORKSPACE,sha
from experiments.movielens.pipeline.schema import manager_at,EXTRAS
from experiments.movielens.pipeline.model import ML1MContinuationModel
from experiments.kuairand.pipeline.run_epochwise_exit import EpochwiseExitTrainer,write_json
from experiments.kuairand.pipeline.exit_training import state_hash,joint_parameter_groups,predict_batches
from experiments.kuairand.pipeline.budget_stop_gradient import gradient_route_audit

CONFIGS = dict(fusion=dict(mode='fusion',mixture=False),single=dict(mode='ours',mixture=False),
               mixture=dict(mode='ours',mixture=True))


def load_batches(data,manager):
    batches={}
    for split in ['train','valid','test']:
        ds=ParquetStreamDataset(pl.scan_parquet(data/f'{split}.parquet'),manager,
            batch_size=4096,shuffle=False,drop_last=False,extra_col_names=EXTRAS)
        loader=list(DataLoader(ds,batch_size=None,num_workers=0))
        for b in loader:
            b['user_id']=torch.zeros_like(b['user_id'])
        batches[split]=loader
    return batches


class ML1MTrainer(EpochwiseExitTrainer):
    def train_epoch(self):
        self.loss_rows=[]
        value=super().train_epoch()
        if self.loss_rows:
            pl.DataFrame(self.loss_rows).write_parquet(self.output/f'loss_steps_epoch_{self.epoch:02d}.parquet')
        return value

    def train_step(self,batch,ctx):
        value=super().train_step(batch,ctx)
        if self.model.mode=='ours':
            row=self.model.last_loss_terms
            assert all(np.isfinite(v) for v in row.values())
            assert row['auxiliary_main_ratio']<=.1+1e-12
            self.loss_rows.append(dict(step=self.global_step+1,rows=len(batch['y_continue']),**row))
        return value


def run(output):
    torch.set_num_threads(4)
    data=WORKSPACE/'data/continuation_v1'
    meta=json.loads((data/'manifest.json').read_text())
    assert all(sha(p)==h for p,h in meta['source_hashes'].items())
    manager=manager_at(data/'schema',load=True)
    batches=load_batches(data,manager)
    for split,loader in batches.items():
        ids=torch.cat([b['request_id'] for b in loader]).numpy()
        expected=pl.read_parquet(data/f'{split}.parquet',columns=['request_id'])['request_id'].to_numpy()
        np.testing.assert_array_equal(ids,expected)
        assert len(ids)==meta['splits'][split]['rows']
    sources=[Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('schema.py'),data/'manifest.json']
    sources += [data/f'{s}.parquet' for s in ['train','valid','test']]
    sources += list((data/'schema').glob('*.json'))
    kr=Path(__file__).resolve().parents[2]/'kuairand/pipeline'
    sources += [kr/n for n in ['exit_model.py','auxiliary_exit_model.py','limited_auxiliary_exit.py',
        'run_epochwise_exit.py','exit_training.py','budget_stop_gradient.py','evaluation.py']]
    hashes={str(p.resolve()):sha(p) for p in sources}
    config=dict(data=str(data),models=CONFIGS,seeds=[2026,2027,2028],epochs=5,
        source_hashes=hashes,full_train=True,full_valid=True,full_test=True,uid_input_zero=True,
        batch_size=4096,budget_lr=.005,other_lr=.001,optimizer='AdamW',weight_decay=1e-6,
        zero_decay='Embedding,bias,1D',clip=10,selection='fixed epoch5; no best restore',
        scheduler=None,ema=False,stop_gradient='joint BCE to budget',auxiliary_max_main_ratio=.1,
        pace='previous rating only',split=meta['protocol'],coverage=meta['splits'])
    write_json(output/'config.json',config)
    summary={}
    ratio=meta['splits']['train']['mean_steps']
    for seed in config['seeds']:
        common=None
        for name,options in CONFIGS.items():
            key=f'{seed}/{name}';dest=output/key;dest.mkdir(parents=True)
            torch.manual_seed(seed);np.random.seed(seed)
            model=ML1MContinuationModel(manager,**options,requests_per_session=ratio)
            shared={n:v.detach().clone() for n,v in model.state_dict().items() if not n.startswith('budget_net.')}
            if common is None:
                common=shared
            else:
                assert shared.keys()==common.keys() and all(torch.equal(v,common[n]) for n,v in shared.items())
            groups=joint_parameter_groups(model)
            write_json(dest/'initialization.json',dict(state_sha256=state_hash(model),
                common_nonbudget_tensors_equal=True,
                optimizer_groups=[dict(scope=g['scope'],lr=g['lr'],weight_decay=g['weight_decay'],tensors=len(g['params'])) for g in groups]))
            smoke=batches['train'][0]
            loss=model.calculate_loss(smoke)
            assert torch.isfinite(loss)
            if model.mode=='ours':
                write_json(dest/'gradient_route.json',gradient_route_audit(model,smoke))
            # 冒烟仅计算梯度，不执行更新，也不改变初始化。
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            model.zero_grad(set_to_none=True)
            cfg=SimpleNamespace(device='cpu',max_epochs=5,ckpt_dir=str(dest),
                experiment_name=f'ml1m_{name}_{seed}',dataset_name='ml1m_continuation_v1',train_log_interval=200)
            trainer=ML1MTrainer(model,torch.optim.AdamW(groups,weight_decay=0),manager,
                TrainerDataLoaders(batches['train'],batches['valid'],batches['test']),TrainerComponents(),cfg,output=dest)
            print('START',key,flush=True)
            trainer.run()
            assert trainer.global_step==5*len(batches['train']) and len(trainer.records)==5
            # 重载最终checkpoint，对完整test再推理，核对所有列。
            saved=torch.load(dest/'epoch_05/checkpoint.pt',map_location='cpu',weights_only=False)
            with torch.random.fork_rng():
                recovered=ML1MContinuationModel(manager,**options,requests_per_session=ratio)
            recovered.load_state_dict(saved['model_state_dict'],strict=True)
            assert predict_batches(recovered,batches['test']).equals(pl.read_parquet(dest/'epoch_05/test_predictions.parquet'))
            write_json(dest/'complete.json',dict(epochs=5,steps=trainer.global_step,full_test_reload_exact=True))
            summary[key]=trainer.records;write_json(output/'summary.json',summary)
            print('TRAJECTORY_COMPLETE',key,flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    write_json(output/'complete.json',dict(trajectories=len(summary),full_test_evaluations=5*len(summary),
        source_hashes_unchanged=True,all_epoch5_full_test_reload_exact=True))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=WORKSPACE/'runs/continuation_v1')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);(args.output/'logs').mkdir()
    print('日志：',args.output/'logs/train.log',flush=True)
    with (args.output/'logs/train.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):run(args.output)
    print('完成：',args.output/'summary.json')
