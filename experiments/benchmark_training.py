"""Source training and target adaptation with the paper's checkpoint rules."""

from copy import deepcopy
from contextlib import nullcontext
import math
import time

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from experiments.benchmark_common import prepare_batch_inputs
from src.adapters.bnpa_sg import (
    accumulate_bnpa_sg_p_grad,
    bnpa_sg_capture,
    make_bnpa_sg_p_optimizer,
    set_bnpa_sg_projection_frozen,
)
from src.train import freeze_bn_eval


class SGSession:
    """Update P from rematerialized inputs; retain only the reported counters."""

    def __init__(self, model, config):
        set_bnpa_sg_projection_frozen(model)
        self.model = model
        self.capturing = config.p_lr != 0.0
        self.optimizer = make_bnpa_sg_p_optimizer(model, config)
        self.updates = self.sites = 0

    def begin(self):
        set_bnpa_sg_projection_frozen(self.model)
        self.optimizer.zero_grad(set_to_none=True)
        return bnpa_sg_capture(self.model, enabled=self.capturing)

    def accumulate(self, x, force_bn_eval):
        if not self.capturing:
            return None
        self.model.train()
        if force_bn_eval:
            freeze_bn_eval(self.model)
        set_bnpa_sg_projection_frozen(self.model)
        stats = accumulate_bnpa_sg_p_grad(self.model, x)
        if stats.performed:
            self.updates += 1
            self.sites += stats.sites_accumulated
        return stats

    def finish(self, stats):
        if stats is not None and stats.performed:
            self.optimizer.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)
        set_bnpa_sg_projection_frozen(self.model)

    def metrics(self):
        return {
            "bnpa_sg_updates": self.updates,
            "bnpa_sg_sites_accumulated_mean": (
                self.sites / self.updates if self.updates else 0.0
            ),
        }

    def close(self):
        set_bnpa_sg_projection_frozen(self.model)
        self.optimizer.zero_grad(set_to_none=True)


def changing_state(model, *optimizers):
    """Copies of every tensor adaptation can change: optimized parameters, buffers.

    Frozen weights never change, so the best-step checkpoint leaves them out rather
    than copying the whole model. Restore it with load_state_dict(strict=False).
    """
    changing = {id(b) for b in model.buffers()}
    changing |= {id(p) for o in optimizers for g in o.param_groups for p in g["params"]}
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict(keep_vars=True).items()
        if id(tensor) in changing
    }


def fit_steps_best(
    model,
    train_loader,
    val_loader,
    steps,
    lr,
    weight_decay,
    eval_every_steps,
    device,
    backbone,
    force_bn_eval=False,
    sg_config=None,
    time_spec_threshold_macro_f1=None,
    time_spec_start_time=None,
):
    """Select the checkpoint with the lowest target validation loss."""
    sg = SGSession(model, sg_config) if sg_config is not None else None
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    criterion = nn.CrossEntropyLoss()
    optimizers = [optimizer, sg.optimizer] if sg else [optimizer]
    best_state = changing_state(model, *optimizers)
    best_metrics = {"loss": math.inf, "accuracy": 0.0, "macro_f1": 0.0, "step": 0}
    iterator = iter(train_loader)
    time_spec = {
        "time_spec_reached": "",
        "time_spec_reach_step": "",
        "time_spec_reach_time_sec": "",
    }
    for step in range(1, steps + 1):
        model.train()
        if force_bn_eval:
            freeze_bn_eval(model)
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        x = prepare_batch_inputs(x.to(device, non_blocking=True), backbone)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with sg.begin() if sg else nullcontext():
            loss = criterion(model(x), y)
            loss.backward()
        sg_stats = sg.accumulate(x, force_bn_eval) if sg else None
        optimizer.step()
        if sg:
            sg.finish(sg_stats)
        if step % max(1, eval_every_steps) == 0 or step == steps:
            elapsed = (
                adaptation_clock(device) - time_spec_start_time
                if time_spec_start_time is not None
                else None
            )
            metrics = evaluate(model, val_loader, device, backbone)
            metrics["step"] = step
            if (
                time_spec_threshold_macro_f1 is not None
                and elapsed is not None
                and time_spec["time_spec_reached"] is not True
                and metrics["macro_f1"] >= time_spec_threshold_macro_f1
            ):
                time_spec.update(
                    time_spec_reached=True,
                    time_spec_reach_step=step,
                    time_spec_reach_time_sec=elapsed,
                )
            if metrics["loss"] < best_metrics["loss"] - 1e-8:
                best_metrics = metrics
                best_state = changing_state(model, *optimizers)
    if sg:
        best_metrics.update(sg.metrics())
    if time_spec_threshold_macro_f1 is not None:
        best_metrics.update(time_spec)
    model.load_state_dict(best_state, strict=False)
    if sg:
        sg.close()
    return best_metrics


def fit_epochs_best(
    model, train_loader, val_loader, epochs, lr, weight_decay, device, backbone
):
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    criterion = nn.CrossEntropyLoss()
    best_state = deepcopy(model.state_dict())
    best_metrics = {"loss": math.inf, "accuracy": 0.0, "macro_f1": 0.0, "epoch": 0}
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = prepare_batch_inputs(x.to(device, non_blocking=True), backbone)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device, backbone)
        metrics["epoch"] = epoch
        if metrics["loss"] < best_metrics["loss"] - 1e-8:
            best_metrics, best_state = metrics, deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_metrics


@torch.no_grad()
def evaluate(model, loader, device, backbone):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses, y_true, y_pred = [], [], []
    total_examples = 0
    for x, y in loader:
        x = prepare_batch_inputs(x.to(device, non_blocking=True), backbone)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        losses.append(float(criterion(logits, y).item()) * int(y.numel()))
        total_examples += int(y.numel())
        y_true.append(y.detach().cpu())
        y_pred.append(logits.argmax(dim=-1).detach().cpu())
    if not y_true:
        return {"loss": math.inf, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    true_np, pred_np = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
    return {
        "loss": sum(losses) / total_examples,
        "accuracy": float(accuracy_score(true_np, pred_np)),
        "macro_f1": float(f1_score(true_np, pred_np, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(true_np, pred_np, average="weighted", zero_division=0)
        ),
    }


def adaptation_clock(device):
    """Synchronize at measurement boundaries so CUDA work is included."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()
