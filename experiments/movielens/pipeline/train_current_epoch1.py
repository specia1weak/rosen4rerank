# 参考入口：从仓库根使用 uv run python -m experiments.movielens.pipeline.train_current_epoch1 启动。
# 保留历史实验的严格对齐检查；需要对应数据/schema及历史run，不是下载即跑的独立教程。
"""ML1M当前反馈：三主模型，辅助/权重/隔离子类，三seed完整单epoch。"""
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from rosen.core.train.context import TrainerComponents,TrainerDataLoaders
from rosen.data.dataset import ParquetStreamDataset
from experiments.movielens.pipeline.prepare_sessions import WORKSPACE,sha
from experiments.movielens.pipeline.current_schema import manager_at,EXTRA_COLUMNS
from experiments.movielens.pipeline.current_model import CurrentML1MModel,VARIANTS
from experiments.kuairand.pipeline.exit_training import state_hash,joint_parameter_groups,predict_batches
from experiments.kuairand.pipeline.run_epochwise_exit import EpochwiseExitTrainer,write_json

MATRIX = [('fusion','none')]+[(name,v) for name in ['single','mixture'] for v in VARIANTS]


def load_batches(data,manager):
    batches = {}
    for split in ['train','valid','test']:
        dataset = ParquetStreamDataset(pl.scan_parquet(data/f'{split}.parquet'),manager,
            batch_size=4096,shuffle=False,drop_last=False,extra_col_names=EXTRA_COLUMNS)
        batches[split] = list(DataLoader(dataset,batch_size=None,num_workers=0))
        for batch in batches[split]:batch['user_id'] = torch.zeros_like(batch['user_id'])
    return batches


def old_state_hash(model):
    digest = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if name.startswith('rating_projection.'):continue
        digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def graph_audit(model,batch):
    if model.mode == 'fusion':return dict(joint_budget='not applicable')
    terms = model.loss_terms(batch)
    budget = list(model.budget_embedding.parameters())+list(model.budget_net.parameters())
    content = list(model.omni_embedding.parameters())+list(model.content_net.parameters())+list(model.rating_projection.parameters())
    grads = torch.autograd.grad(terms['joint_bce'],budget,allow_unused=True,retain_graph=True)
    detached = model.variant.startswith('detach_')
    assert all(g is None for g in grads) if detached else sum(float(g.norm()) for g in grads if g is not None)>0
    nll_content = torch.autograd.grad(terms['left_session_nll'],content,allow_unused=True)
    assert all(g is None for g in nll_content)
    grad = torch.autograd.grad(model.calculate_loss(batch),model.rating_projection.weight)[0]
    assert grad.norm()>0
    return dict(joint_budget='disconnected' if detached else 'connected',nll_content='disconnected',
        current_rating_projection_gradient=float(grad.norm()))


class CurrentTrainer(EpochwiseExitTrainer):
    def train_epoch(self):
        self.loss_rows = []
        result = super().train_epoch()
        pl.DataFrame(self.loss_rows).write_parquet(self.output/'loss_steps.parquet')
        return result

    def train_step(self,batch,ctx):
        loss = super().train_step(batch,ctx)
        terms = self.model.last_loss_terms
        assert all(np.isfinite(v) for v in terms.values())
        self.loss_rows.append(dict(step=self.global_step+1,rows=len(batch['y_continue']),**terms))
        return loss


def run(output):
    torch.set_num_threads(4)
    data = WORKSPACE/'data/current_information_v2'
    manifest = json.loads((data/'manifest.json').read_text())
    manager = manager_at(data/'schema',load=True)
    batches = load_batches(data,manager)
    for split,loader in batches.items():
        ids = torch.cat([b['request_id'] for b in loader]).numpy()
        expected = pl.read_parquet(data/f'{split}.parquet',columns=['request_id'])['request_id'].to_numpy()
        assert np.array_equal(ids,expected) and len(ids)==manifest['splits'][split]['rows']
        assert all(torch.count_nonzero(b['user_id'])==0 for b in loader)
    sources = {**manifest['source_hashes'],**{str(p.resolve()):sha(p) for p in [
        Path(__file__),Path(__file__).with_name('current_model.py'),Path(__file__).with_name('model.py'),
        data/'manifest.json',*[data/f'{s}.parquet' for s in ['train','valid','test']],
        *list((data/'schema').glob('*.json'))]}}
    kr = Path(__file__).resolve().parents[2]/'kuairand/pipeline'
    sources.update({str((kr/n).resolve()):sha(kr/n) for n in ['exit_model.py','auxiliary_exit_model.py',
        'limited_auxiliary_exit.py','exit_training.py','run_epochwise_exit.py','evaluation.py']})
    assert all(sha(p)==h for p,h in sources.items())
    config = dict(data=str(data),matrix=MATRIX,seeds=[2026,2027,2028],epochs=1,source_hashes=sources,
        full_train=True,full_valid=True,full_test=True,uid_input_zero=True,batch_size=4096,
        budget_lr=.005,other_lr=.001,optimizer='AdamW',weight_decay=1e-6,zero_decay='Embedding,bias,1D',
        clip=10,scheduler=None,ema=False,warm_start=False,selection='fixed epoch1; no best restore',
        prediction='after current rating',default_pace=manifest['protocol']['pace'],
        variants={'none':'joint BCE only','weak':'joint+.1N+.1R capped together at .1joint',
            'strong':'joint+N+R uncapped','detach_0.1':'joint budget detached+.1N+weak R',
            'detach_1':'joint budget detached+1N+weak R'},coverage=manifest['splits'])
    write_json(output/'config.json',config)
    summary = {}
    for seed in config['seeds']:
        initial_by_class = {}
        for name,variant in MATRIX:
            key = f'{seed}/{name}/{variant}';dest = output/key;dest.mkdir(parents=True,exist_ok=False)
            torch.manual_seed(seed);np.random.seed(seed)
            options = dict(mode='fusion' if name=='fusion' else 'ours',mixture=name=='mixture',
                variant=variant,requests_per_session=manifest['splits']['train']['mean_steps'])
            model = CurrentML1MModel(manager,**options)
            initial = state_hash(model)
            old = json.loads((WORKSPACE/f'runs/continuation_v1/{seed}/{name}/initialization.json').read_text())
            assert old_state_hash(model)==old['state_sha256']
            if name in initial_by_class:assert initial==initial_by_class[name]
            initial_by_class[name] = initial
            groups = joint_parameter_groups(model)
            meta = [dict(scope=g['scope'],lr=g['lr'],weight_decay=g['weight_decay'],tensors=len(g['params'])) for g in groups]
            expected = old['optimizer_groups']
            for group in expected:
                if group['scope']=='other' and group['weight_decay']==1e-6:group['tensors']+=1
            assert meta==expected
            rng = torch.get_rng_state().clone()
            audit = graph_audit(model,batches['train'][0].interaction)
            assert initial==state_hash(model) and torch.equal(rng,torch.get_rng_state())
            uid = {n:table.embedding.weight.detach().clone() for n,table in [
                ('budget',model.budget_embedding.emb_modules['user_id']),('content',model.omni_embedding.emb_modules['user_id'])]}
            write_json(dest/'initialization.json',dict(state_sha256=initial,original_common_initialization_exact=True,
                within_class_loss_variants_exact=True,optimizer_groups=meta,**audit))
            cfg = SimpleNamespace(device='cpu',max_epochs=1,ckpt_dir=str(dest),experiment_name=f'ml1m_current_{seed}_{name}_{variant}',
                dataset_name='ml1m_current_information_v2',train_log_interval=100)
            trainer = CurrentTrainer(model,torch.optim.AdamW(groups,weight_decay=0),manager,
                TrainerDataLoaders(batches['train'],batches['valid'],batches['test']),TrainerComponents(),cfg,output=dest)
            print('START',key,audit,flush=True)
            trainer.run()
            assert trainer.global_step==196 and len(trainer.records)==1
            assert torch.equal(uid['budget'],model.budget_embedding.emb_modules['user_id'].embedding.weight)
            assert torch.equal(uid['content'],model.omni_embedding.emb_modules['user_id'].embedding.weight)
            with torch.random.fork_rng():recovered = CurrentML1MModel(manager,**options)
            checkpoint = torch.load(dest/'epoch_01/checkpoint.pt',map_location='cpu',weights_only=False)
            recovered.load_state_dict(checkpoint['model_state_dict'],strict=True)
            assert predict_batches(recovered,batches['test']).equals(pl.read_parquet(dest/'epoch_01/test_predictions.parquet'))
            write_json(dest/'complete.json',dict(epochs=1,steps=196,uid_unchanged=True,full_test_reload_exact=True))
            summary[key] = trainer.records;write_json(output/'summary.json',summary)
            print('TRAJECTORY_COMPLETE',key,flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    write_json(output/'complete.json',dict(trajectories=len(summary),full_test_evaluations=len(summary),
        all_full_test_checkpoints_reloaded_exact=True,source_hashes_unchanged=True))
    print('COMPLETE',flush=True)


if __name__ == '__main__':
    output = WORKSPACE/'runs/plan22_02_current_epoch1'
    output.mkdir(parents=True,exist_ok=False);(output/'logs').mkdir()
    print('日志：',output/'logs/train.log',flush=True)
    with (output/'logs/train.log').open('w') as f,redirect_stdout(f),redirect_stderr(f):run(output)
