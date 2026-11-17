"""当前KR单次运行链路：parquet -> batch -> joint trainer -> selected checkpoint。

只依赖rosen核心和当前schema，不再沿历史实验入口继承Trainer。
保留plan13的AdamW、裁剪、验证节点与1/3epoch预算快照语义。
训练中不回滚best、不重置优化器；结束后才恢复best。test不传给Trainer。
"""
import hashlib
import json
import math
from pathlib import Path
import shutil

import polars as pl
import torch
from torch.utils.data import DataLoader
from rosen.core.trainers import BaseTrainer
from rosen.data.dataset import ParquetStreamDataset
from rosen.utils.optimize import split_model_params_by_decay
from experiments.kuairand.pipeline.kr_request_schema import EXTRA_COLUMNS


WORKSPACE = Path(__file__).resolve().parents[3] / 'workspace/kr_time_budget'
EXPERIMENTS = {
    'single_step': dict(mode='ours', mixture=False, step_only=True, add_pace=False),
    'single_step_pace': dict(mode='ours', mixture=False, step_only=True, add_pace=True),
    'mixture_step': dict(mode='ours', mixture=True, step_only=True, add_pace=False),
    'mixture_step_pace': dict(mode='ours', mixture=True, step_only=True, add_pace=True),
    'fusion': dict(mode='fusion', mixture=False, step_only=False, add_pace=False),
    'fusion_pace': dict(mode='fusion', mixture=False, step_only=False, add_pace=True),
}


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_batches(data, manager, row_limits=None):
    """读取已编码v3；只添加原始分钟别名，不再fit或重复log1p。None为全量。"""
    manifest = json.loads((data/'manifest.json').read_text())
    if manifest['version'] != 3 or manifest['protocol']['clock'] != 'Leader screen end; delta1min':
        raise ValueError('Only the unchanged KR v3 raw-minute contract is supported')
    setting = manager.get_setting('elapsed_min')
    if (setting.min_val, setting.max_val) != (0., 1.):
        raise ValueError('elapsed_min must be the v3 Float32 passthrough, not a scaled clock')
    batches = {}
    for split in ['train', 'valid', 'test']:
        frame = pl.scan_parquet(data/f'{split}.parquet').with_columns(
            pl.col('elapsed_min').alias('elapsed_min_raw'))
        if row_limits is not None and split in row_limits:
            frame = frame.head(row_limits[split])
        dataset = ParquetStreamDataset(frame, manager, batch_size=4096, shuffle=False,
                                       drop_last=False, extra_col_names=EXTRA_COLUMNS+['elapsed_min_raw'])
        batches[split] = list(DataLoader(dataset, batch_size=None, num_workers=0))
    return batches


def joint_parameter_groups(model, budget_lr=.005):
    """一个AdamW，两种LR；原生decay分组，所有参数恰好出现一次。"""
    budget_ids = {id(p) for module in model.budget_modules for p in module.parameters()}
    groups = []
    for decay in split_model_params_by_decay(model, weight_decay=1e-6):
        for scope in ['budget', 'other']:
            params = [p for p in decay['params'] if (id(p) in budget_ids) == (scope == 'budget')]
            if params:
                groups.append(dict(params=params, weight_decay=decay['weight_decay'],
                                   lr=budget_lr if scope == 'budget' else .001, scope=scope))
    ids = [id(p) for group in groups for p in group['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    return groups


class JointExitTrainer(BaseTrainer):
    """Native BaseTrainer + 显式验证选择、首步更新与预算边界留证。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = Path(self.cfg.ckpt_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.best_loss, self.selected, self.eval_step = float('inf'), None, 0
        self.marks = {max(1, math.ceil(len(self.train_loader)*f)) for f in [.25, .5]}

    def calculate_train_loss(self, batch, ctx):
        loss = super().calculate_train_loss(batch, ctx)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        return loss

    def train_step(self, batch, ctx):
        first = self.global_step == 0
        if first:
            if self.model.mode == 'ours':
                watched = {
                    'budget_head': self.model.budget_net[-1].weight,
                    'content_head': self.model.content_net[-1].weight,
                    'budget_user_embedding': self.model.budget_embedding.emb_modules['user_id'].embedding.weight,
                    'content_user_embedding': self.model.omni_embedding.emb_modules['user_id'].embedding.weight,
                }
            else:
                watched = {'fusion_head': self.model.fusion_net[-1].weight,
                           'user_embedding': self.model.omni_embedding.emb_modules['user_id'].embedding.weight}
            before = {name: value.detach().clone() for name, value in watched.items()}
        # BaseTrainer负责：loss.backward -> clip_grad_norm_(10) -> optimizer.step。
        loss = super().train_step(batch, ctx)
        if ctx.batch_idx+1 in self.marks:
            # 此时global_step尚未加1，记录实际完成的更新次数。
            self.eval_step = self.global_step+1
            self.evaluate_epoch()
            self.model.train()
        if first:
            audit = {}
            for name, parameter in watched.items():
                assert parameter.requires_grad and parameter.grad is not None
                grad = parameter.grad.norm().item()
                change = (parameter.detach()-before[name]).abs().max().item()
                assert math.isfinite(grad) and grad > 0 and change > 0, name
                audit[name] = dict(gradient_norm_after_clipping=grad, parameter_max_abs_update=change)
            (self.output/'first_update.json').write_text(json.dumps(audit, indent=2))
            print('FIRST_UPDATE', self.model.mode, json.dumps(audit), flush=True)
        return loss

    def evaluate_epoch(self):
        metrics = super().evaluate_epoch()
        step = max(self.eval_step, self.global_step)
        row = dict(step=step, epoch=self.epoch, stage=self.model.stage, valid=metrics)
        print('VALID', json.dumps(row), flush=True)
        with (self.output/'progress.jsonl').open('a') as log:
            log.write(json.dumps(row)+'\n')
        if metrics['overall']['logloss'] < self.best_loss:
            self.best_loss, self.selected = metrics['overall']['logloss'], row
            torch.save(dict(state_dict=self.model.state_dict(), selection=row), self.output/'best.pt')
        batches = len(self.train_loader)
        if step in {batches, 3*batches}:
            budget = step//batches
            shutil.copy2(self.output/'best.pt', self.output/f'best_within_{budget}ep.pt')
            record = dict(budget_epochs=budget, budget_optimizer_steps=step, selected=self.selected,
                terminal_state_sha256=state_hash(self.model), optimizer_state_entries=len(self.optimizer.state),
                optimizer_state_steps=sorted({int(v['step'].item()) for v in self.optimizer.state.values() if 'step' in v}))
            (self.output/f'budget_{budget}ep.json').write_text(json.dumps(record, indent=2))
        return metrics

    def run(self, *, stop_condition=None):
        # 对所有模式显式包含初始验证；外部不要再重复调用evaluate_epoch。
        self.evaluate_epoch()
        super().run(stop_condition=stop_condition)
        best = torch.load(self.output/'best.pt', map_location=self.cfg.device, weights_only=False)
        self.model.load_state_dict(best['state_dict'])
        (self.output/'selection.json').write_text(json.dumps(
            dict(selected=best['selection'], optimizer_steps_executed=self.global_step), indent=2))
        return best['selection']['valid']


@torch.no_grad()
def predict_batches(model, batches):
    """只读评估；列顺序与plan13一致，支持逐值复核历史预测。"""
    model.eval()
    frames = []
    for batch in batches:
        b = {name: value.to('cpu') for name, value in batch.interaction.items()}
        data = {name: b[name].numpy() for name in ['request_id', 'session_id', 'user_id_raw', 'tab_raw',
                'req_idx_raw', 'y_continue', 'elapsed_min', 'elapsed_min_raw']}
        prediction = model.predict(b)
        assert torch.isfinite(prediction).all()
        data['prediction'] = prediction.numpy()
        if model.mode == 'ours':
            product, gamma, stay, survival, log_s = model.components(b)
            assert torch.equal(product, prediction)
            clocks = model.clock_components(b)
            data.update(gamma=gamma.numpy(), content_stay=stay.numpy(), survival=survival.numpy(),
                log_survival=log_s.numpy(), gamma_time=clocks['time'][0].numpy(), gamma_step=clocks['step'][0].numpy())
            assert torch.equal(gamma, clocks['time'][0]*clocks['step'][0])
        frames.append(pl.DataFrame(data))
    return pl.concat(frames)
