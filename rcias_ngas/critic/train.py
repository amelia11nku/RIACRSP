"""Fixed-epoch grouped OOF training primitives for Q(G,k,D,R)."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import random

import torch

from rcias_ngas.csg.features import tensorize
from rcias_ngas.critic.joint_critic import JointCritic
from rcias_ngas.critic.losses import joint_loss


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_cache(path: Path, manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    if digest(path) != manifest['cache_sha256']:
        raise ValueError('Training cache hash mismatch')
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != manifest['uncompressed_sha256']:
        raise ValueError('Training cache content hash mismatch')
    payload = json.loads(raw)
    if payload['schema'] != 'ngas-a13-training-cache-v1':
        raise ValueError('Training cache schema mismatch')
    records = payload['records']
    if (len(records), sum(len(row['actions']) for row in records)) != (72, 6465):
        raise ValueError('Training cache scope mismatch')
    return records


def model_for(config: dict, seed: int, device: str | torch.device) -> JointCritic:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    critic = config['critic']
    return JointCritic(hidden=int(critic['hidden_dim']), layers=int(critic['message_passing_layers'])).to(device)


def prepare(record: dict, device: str | torch.device) -> tuple[dict, torch.Tensor]:
    batch = tensorize(record['state_features'], record['action_features'], device=device)
    labels = torch.tensor(record['replicate_advantages'], dtype=torch.float32, device=device)
    return batch, labels


def fit(records: list[dict], config: dict, seed: int, device: str | torch.device,
        epoch_callback=None) -> tuple[JointCritic, list[dict]]:
    if len({row['instance_id'] for row in records}) * 4 != len(records):
        raise ValueError('Each training instance must contribute exactly four states')
    model = model_for(config, seed, device)
    training = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training['learning_rate']),
                                  weight_decay=float(training['weight_decay']))
    prepared = {row['state_id']: prepare(row, device) for row in records}
    history = []
    for epoch in range(1, int(training['epochs']) + 1):
        model.train()
        order = list(records)
        random.Random(seed + 1_000_003 * epoch).shuffle(order)
        totals = {'total': 0., 'pairwise': 0., 'listwise': 0., 'regression': 0.,
                  'beats_fallback': 0., 'material_pairs': 0.}
        for record in order:
            batch, labels = prepared[record['state_id']]
            optimizer.zero_grad(set_to_none=True)
            losses = joint_loss(model(batch), labels)
            if not torch.isfinite(losses['total']):
                raise FloatingPointError('Non-finite joint critic loss')
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training['gradient_clip_norm']))
            optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach())
        row = {'epoch': epoch, **{name: value / len(order) for name, value in totals.items()}}
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(row)
    return model, history


def predict(model: JointCritic, records: list[dict], device: str | torch.device) -> list[dict]:
    predictions = []
    model.eval()
    with torch.inference_mode():
        for record in sorted(records, key=lambda row: row['state_id']):
            output = model(tensorize(record['state_features'], record['action_features'], device=device))
            advantage = output['advantage'].detach().cpu().tolist()
            logits = output['beats_fallback_logit'].detach().cpu()
            predictions.append({
                'state_id': record['state_id'],
                'predicted_advantage': advantage,
                'predicted_beats_fallback_logit': logits.tolist(),
                'predicted_beats_fallback_probability': torch.sigmoid(logits).tolist(),
            })
    return predictions
