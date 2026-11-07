"""Positive-unlabeled fault learning with soft DTI-like geometry supervision."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def nonnegative_pu(logits, positive, observe, prior=0.015, weights=None):
    """Nonnegative PU risk: pi*L_pos+max(0,L_unlabel_neg-pi*L_pos_neg).

    This is a training surrogate, not the official DTI. Unlabeled includes
    locations that might contain unmapped faults; no true-negative claim.
    """
    logits=logits.squeeze(1).float()
    positive=positive.bool()
    observe=observe.bool()
    unlabeled=observe & ~positive
    weight=torch.ones_like(logits) if weights is None else weights.float()
    wp=positive.float()*weight
    wu=unlabeled.float()*weight
    if wp.sum()<1 or wu.sum()<1:
        return logits.sum()*0
    rp= (wp*F.softplus(-logits)).sum()/wp.sum()
    rpn=(wp*F.softplus(logits)).sum()/wp.sum()
    run=(wu*F.softplus(logits)).sum()/wu.sum()
    return prior*rp+torch.clamp(run-prior*rpn,min=0)


def loss_all(out, truth, supervise, proximity, orient_target, orient_ok, prior_regime,
             pu_prior=0.015, w_tv=0.3, w_prox=0.2, w_orient=0.08):
    probability=out["prob"].squeeze(1)
    logits=torch.logit(probability.clamp(1e-5,1-1e-5)).unsqueeze(1)
    positive=truth>0.5
    base=nonnegative_pu(logits,positive,supervise,pu_prior)
    bglobal=nonnegative_pu(out["global_logits"],positive,supervise,pu_prior)
    soft=(1-proximity/300.).clamp(min=0,max=1)*supervise.float()
    p=probability.float()*supervise.float()
    tp=(p*soft).sum()
    fp=(p*(1-soft)*supervise.float()).sum()
    fn=((1-p)*soft).sum()
    tversky=1-(tp+1e-5)/(tp+0.2*fp+0.8*fn+1e-5)
    # Label-free geophysical soft routing target, not mapped-fault clusters.
    r=prior_regime.float().clamp(0.01,0.99)
    w=out["routes"]
    route_loss=(-r*w[:,0].clamp_min(1e-6).log()-(1-r)*w[:,1].clamp_min(1e-6).log())
    route_loss=(route_loss*supervise.float()).sum()/supervise.sum().clamp_min(1)
    e0=nonnegative_pu(out["experts"][:,0:1],positive,supervise,pu_prior,weights=r)
    e1=nonnegative_pu(out["experts"][:,1:2],positive,supervise,pu_prior,weights=1-r)
    prox_mask=supervise & (proximity<=1000)
    prox_error=F.smooth_l1_loss(out["proximity"].squeeze(1)/300,
                               proximity.clamp(max=300)/300,reduction="none")
    prox_loss=(prox_error*prox_mask.float()).sum()/prox_mask.sum().clamp_min(1)
    orient_mask=supervise & orient_ok & (proximity<=300)
    cos=(out["orientation"]*orient_target).sum(dim=1)
    orient_loss=((1-cos)*orient_mask.float()).sum()/orient_mask.sum().clamp_min(1)
    total=0.55*base+0.45*bglobal+w_tv*tversky+0.05*(e0+e1)+0.04*route_loss+w_prox*prox_loss+w_orient*orient_loss
    return total, {"loss":float(total.detach()),"nnpu":float(base.detach()),
                   "tversky":float(tversky.detach()),"router":float(route_loss.detach()),
                   "proximity":float(prox_loss.detach()),"orientation":float(orient_loss.detach())}
