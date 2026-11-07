"""Shared three-modality GeoStructureNet with two geological specialist heads.

Router receives only label-free geophysical/topographic feature tensors. Geological
specialists share the encoder and decoder (no redundant full U-Nets).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Res(nn.Module):
    def __init__(self, inc, out, dilation=1):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(inc, out, 3, padding=dilation, dilation=dilation),
                               nn.GroupNorm(min(8, out), out), nn.SiLU(inplace=True),
                               nn.Conv2d(out, out, 3, padding=1),
                               nn.GroupNorm(min(8, out), out))
        self.skip = nn.Conv2d(inc, out, 1) if inc != out else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.f(x) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, inc, widths):
        super().__init__()
        self.blocks = nn.ModuleList([Res(inc if i == 0 else widths[i-1], w)
                                     for i,w in enumerate(widths)])

    def forward(self,x):
        out=[]
        for i,block in enumerate(self.blocks):
            if i:
                x=F.avg_pool2d(x, 2)
            x=block(x)
            out.append(x)
        return out


class GeoRegimeNet(nn.Module):
    """Three encoders, four masked-fused levels, regional context, 3 output heads.

    Output score is a mixture of probabilities, not logits, with a global
    fallback. Specialists receive gradient and are trained jointly.
    """
    def __init__(self, nmag, ngrav, nterrain, width=24):
        super().__init__()
        widths=[width,width*2,width*3,width*4]
        self.magnetic=Encoder(nmag,widths)
        self.gravity=Encoder(ngrav,widths)
        self.terrain=Encoder(nterrain,widths)
        self.gates=nn.ModuleList(nn.Conv2d(w*3,3,1) for w in widths)
        self.context=nn.Sequential(nn.Conv2d(3,width*2,3,padding=1),
                                   nn.GroupNorm(8,width*2),nn.SiLU(),
                                   nn.Conv2d(width*2,widths[-1],3,padding=1),
                                   nn.GroupNorm(8,widths[-1]),nn.SiLU())
        self.bottleneck=nn.Sequential(*[Res(widths[-1],widths[-1],dilation=d)
                                        for d in (1,2,4,8)])
        self.decoder=nn.ModuleList([Res(widths[i+1]+widths[i],widths[i])
                                    for i in (2,1,0)])
        self.global_head=nn.Conv2d(widths[0],1,1)
        self.specialists=nn.ModuleList([nn.Conv2d(widths[0],1,1) for _ in range(2)])
        self.router=nn.Conv2d(widths[0],2,1)
        self.specialist_strength=nn.Conv2d(widths[0],1,1)
        nn.init.constant_(self.specialist_strength.bias,-1.4)
        self.proximity=nn.Conv2d(widths[0],1,1)
        self.orientation=nn.Conv2d(widths[0],2,1)

    def forward(self,mag,grav,terrain,context):
        encs=[self.magnetic(mag),self.gravity(grav),self.terrain(terrain)]
        fused=[]
        for level,gate in enumerate(self.gates):
            a,b,c=(features[level] for features in encs)
            weights=gate(torch.cat((a,b,c),dim=1)).softmax(dim=1)
            fused.append(a*weights[:,0:1]+b*weights[:,1:2]+c*weights[:,2:3])
        # Long-range gravity context is computed globally at 5,15,30 km sigma.
        ctx=self.context(context)
        ctx=F.adaptive_avg_pool2d(ctx, (8,8))
        ctx=F.interpolate(ctx,size=fused[-1].shape[-2:],mode='bilinear',align_corners=False)
        x=self.bottleneck(fused[-1]+ctx)
        for layer,skip in zip(self.decoder,reversed(fused[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=layer(torch.cat([x,skip],dim=1))
        gp=self.global_head(x)
        expert_logits=torch.cat([m(x) for m in self.specialists],dim=1)
        routing_logits=self.router(x)
        routes=routing_logits.softmax(dim=1)
        strength=self.specialist_strength(x).sigmoid()
        pglobal=gp.sigmoid()
        pexperts=expert_logits.sigmoid()
        prob=(1-strength)*pglobal+strength*(routes*pexperts).sum(dim=1,keepdim=True)
        return {"prob":prob.clamp(1e-6,1-1e-6),
                "global_logits":gp,"experts":expert_logits,
                "router_logits":routing_logits,"routes":routes,
                "specialist_weight":strength,"proximity":self.proximity(x),
                "orientation":F.normalize(self.orientation(x),dim=1,eps=1e-6)}
