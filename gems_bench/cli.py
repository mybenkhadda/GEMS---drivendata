"""``gems`` command line (entry point declared in pyproject.toml).

    gems data inspect
    gems features build --config configs/features/structural_profiles.yaml [--names p5:terrain:...]
    gems train    --config configs/experiments/b2_nnpu.yaml          (prepare + train)
    gems run      --config configs/experiments/b2_nnpu.yaml          (all four stages)
    gems evaluate --experiment <EXPERIMENT_ID>
    gems predict  --experiment <EXPERIMENT_ID>                       (submission kind: full-region map)
    gems submit   --experiment <EXPERIMENT_ID>                       (submission kind: GeoTIFF export)
    gems status   --experiment <EXPERIMENT_ID>
    gems registry list | promote --model NAME --experiment ID | reject --model NAME --reason TEXT
    gems parity                                                      (synthetic metric-parity cases)
    gems pipeline plan|run|status [--mode RESUME] [--stage NAME] [--pipeline-config my.yaml]

Global options: --root (workspace), --drive (Drive mirror), --code (checkout with configs/), --resume-policy.
"""
from __future__ import annotations

import argparse
import sys


def _paths(a):
    from .paths import ProjectPaths

    return ProjectPaths.discover(a.root, a.code, a.drive)


def _exp(a):
    from .experiment import Experiment

    if getattr(a, "config", None):
        return Experiment.from_yaml(a.config, root=a.root, drive=a.drive, code=a.code, resume_policy=a.resume_policy)
    return Experiment.resume(a.experiment, root=a.root, drive=a.drive, code=a.code, resume_policy=a.resume_policy)


def cmd_data(a):
    from .data import load_native
    from .config import BenchPaths

    p = _paths(a)
    d = load_native(BenchPaths(p.root))
    print(d.summary().to_string(index=False))
    print(d.band_table().to_string(index=False))


def cmd_features(a):
    from .phase5_context import load_context5

    p = _paths(a)
    ctx = load_context5(p.root, paths5=p, profile_cfg=a.config)
    names = a.names or [n for n in ctx.store.p5_names if n.startswith("p5:")]
    print(ctx.store.build(names).to_string(index=False))


def cmd_pipeline(a):
    from .pipeline import Pipeline, PipelineConfig

    paths = {k: v for k, v in (("local_root", a.root), ("drive_root", a.drive), ("code_root", a.code)) if v}
    over = {"paths": paths} if paths else None
    cfg = PipelineConfig.load(PipelineConfig.default_yaml(), overrides=over)
    if a.pipeline_config:
        import yaml

        from .pipeline.config import _deep_merge

        cfg = PipelineConfig(_deep_merge(cfg.raw, yaml.safe_load(open(a.pipeline_config, encoding="utf-8")) or {}),
                             f"{cfg.source} + {a.pipeline_config}")
        cfg.validate()
    pipe = Pipeline(cfg)
    if a.action == "plan":
        print(pipe.plan(a.mode).to_string(index=False))
    elif a.action == "status":
        print(pipe.status().to_string(index=False))
    elif a.stage:
        pipe.run_stage(a.stage)
    else:
        print(pipe.run(a.mode).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gems", description="GEMS fault-discovery research platform")
    ap.add_argument("--root")
    ap.add_argument("--drive")
    ap.add_argument("--code")
    ap.add_argument("--resume-policy", default="strict", choices=["strict", "config", "force"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("action", choices=["inspect"])
    f = sub.add_parser("features")
    f.add_argument("action", choices=["build"])
    f.add_argument("--config")
    f.add_argument("--names", nargs="*")
    for name in ("train", "run"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
    for name in ("evaluate", "predict", "submit", "status"):
        s = sub.add_parser(name)
        g = s.add_mutually_exclusive_group(required=True)
        g.add_argument("--experiment")
        g.add_argument("--config")
    r = sub.add_parser("registry")
    r.add_argument("action", choices=["list", "promote", "reject"])
    r.add_argument("--model")
    r.add_argument("--experiment")
    r.add_argument("--reason", default="")
    sub.add_parser("parity")
    pl = sub.add_parser("pipeline")
    pl.add_argument("action", choices=["plan", "run", "status"])
    pl.add_argument("--mode")
    pl.add_argument("--stage")
    pl.add_argument("--pipeline-config", help="YAML overriding configs/pipeline/master.yaml")
    a = ap.parse_args(argv)

    if a.cmd == "data":
        cmd_data(a)
    elif a.cmd == "features":
        cmd_features(a)
    elif a.cmd == "train":
        e = _exp(a)
        e.prepare()
        e.train()
        print(e.status().to_string(index=False))
    elif a.cmd == "run":
        print(_exp(a).run_all().to_string(index=False))
    elif a.cmd == "evaluate":
        e = _exp(a)
        e.evaluate()
        print(e.status().to_string(index=False))
    elif a.cmd == "predict":
        _exp(a).train()
    elif a.cmd == "submit":
        print(_exp(a).export())
    elif a.cmd == "status":
        e = _exp(a)
        print(e.status().to_string(index=False))
        print(e.units().to_string(index=False))
    elif a.cmd == "registry":
        from .artifacts.registry import ModelRegistry
        from .experiment.promote import promote, reject

        p = _paths(a)
        if a.action == "list":
            print(ModelRegistry(p.registry).table().to_string(index=False))
        elif a.action == "promote":
            print(promote(a.model, a.experiment, p))
        else:
            print(reject(a.model, a.reason or "manual rejection", p))
    elif a.cmd == "pipeline":
        cmd_pipeline(a)
    elif a.cmd == "parity":
        from .evaluation.parity import run_parity

        print(run_parity().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
