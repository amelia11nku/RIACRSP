"""Within-state robust value learning with paired-noise-aware ranking."""
import torch
from torch.nn import functional as F


def joint_loss(output, replicate_advantages):
    """One state's [actions, matched CRN replicates]; no cross-state pairs."""
    y = replicate_advantages.mean(1)
    prediction = output['advantage']
    differences = replicate_advantages[:, None, :] - replicate_advantages[None, :, :]
    mean = differences.mean(-1)
    se = differences.std(-1, unbiased=True) / replicate_advantages.shape[1]**.5
    material = torch.triu(mean.abs() > torch.maximum(mean.new_tensor(.001), 2 * se), diagonal=1)
    score_difference = prediction[:, None] - prediction[None, :]
    pair = F.softplus(-mean[material].sign() * score_difference[material] / .01).mean() if material.any() else prediction.sum() * 0
    # Listwise groups use only materially separated comparisons; ties and noisy
    # distinctions receive no artificial unique top-1 ordering.
    directed = material | material.T
    terms = []
    for index in range(len(y)):
        mask = directed[index].clone()
        if mask.any():
            mask[index] = True
            target = torch.softmax(y[mask] / .01, dim=0)
            terms.append(-(target * torch.log_softmax(prediction[mask] / .01, dim=0)).sum())
    listwise = torch.stack(terms).mean() if terms else prediction.sum() * 0
    regression = F.smooth_l1_loss(prediction / .01, y / .01)
    beats = (replicate_advantages > 0).float().mean(1)
    bce = F.binary_cross_entropy_with_logits(output['beats_fallback_logit'], beats)
    total = pair + .5 * listwise + regression + .2 * bce
    return {'total': total, 'pairwise': pair, 'listwise': listwise, 'regression': regression,
            'beats_fallback': bce, 'material_pairs': material.sum()}
