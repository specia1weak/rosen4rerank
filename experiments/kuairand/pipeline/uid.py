"""UID置0与表状态审计，提取纯工具，避免依赖历史训练入口。"""
import hashlib
import torch
from rosen.core.interaction import Interaction

def zero_uid_inputs(batches):
    """仅替换user_id张量，不改原batch；所有其余字段（包括raw UID）原样保留。"""
    return {split: [Interaction(dict(batch.interaction, user_id=torch.zeros_like(batch['user_id'])))
                    for batch in loader] for split, loader in batches.items()}


def uid_tables(model):
    return {'content_or_fusion': model.omni_embedding.emb_modules['user_id'].embedding,
            'budget': model.budget_embedding.emb_modules['user_id'].embedding}


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
