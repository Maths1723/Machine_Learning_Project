"""Parametric PINNs for families of second-order IVPs.

We learn u(t, p) ~ y(t; p) for all p in [p_min, p_max] at once, where

    y'' = f(t, y, y'; p),   y(t0(p)) = g0,  y'(t0(p)) = g1.

For the 'c_values' configs the parameter IS the initial-condition point,
t0(p) = p = c; otherwise t0 is fixed (X_START_IC_POINT) and p enters f.

This module is independent of the interactive main.py so that experiments
can be scripted (see experiments/).
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from numerics import reference_solution, relative_l2


# --------------------------------------------------------------------------
# Problem description
# --------------------------------------------------------------------------
@dataclass
class Problem:
    ode_func: callable
    g0: float
    g1: float
    t_min: float
    t_max: float
    p_min: float
    p_max: float
    ic_at_param: bool = True      # True: t0 = p (the 'c_values' configs)
    t0_fixed: float = None        # used when ic_at_param is False
    name: str = "problem"

    @classmethod
    def from_module(cls, mod, name=None):
        ic_at_param = mod.PARAM_NAME_IN_MAT_FILE == "c_values"
        return cls(
            ode_func=mod.ode_system,
            g0=float(mod.FIXED_Y0),
            g1=float(mod.FIXED_Y_PRIME0),
            t_min=float(mod.X_EVAL_START_GEN),
            t_max=float(mod.X_EVAL_END_GEN),
            p_min=float(mod.PARAM_MIN_VALUE),
            p_max=float(mod.PARAM_MAX_VALUE),
            ic_at_param=ic_at_param,
            t0_fixed=None if ic_at_param else float(mod.X_START_IC_POINT),
            name=name or getattr(mod, "__name__", "problem"),
        )

    def ic_point(self, p):
        return p if self.ic_at_param else torch.full_like(p, self.t0_fixed)


def load_problem(path):
    """Load one of the config files (data_augmentation_pendulum.py, ...) as a Problem."""
    import importlib.util
    import os

    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return Problem.from_module(mod, name=name)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def mlp(n_in, n_out, hidden, layers, act=nn.Tanh):
    mods = [nn.Linear(n_in, hidden), act()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), act()]
    mods.append(nn.Linear(hidden, n_out))
    return nn.Sequential(*mods)


class ParametricPINN(nn.Module):
    """u(t, p) = N(t_hat, p_hat) with both inputs affinely mapped to [-1, 1].

    normalize=False reproduces the original PINN_MLP (raw t and p as inputs),
    which saturates the tanh units as soon as t is O(10).
    """

    def __init__(self, problem, hidden=64, layers=4, normalize=True):
        super().__init__()
        self.net = mlp(2, 1, hidden, layers)
        if normalize:
            t_mid, t_half = (problem.t_max + problem.t_min) / 2, (problem.t_max - problem.t_min) / 2
            p_mid, p_half = (problem.p_max + problem.p_min) / 2, max((problem.p_max - problem.p_min) / 2, 1e-12)
        else:
            t_mid, t_half, p_mid, p_half = 0.0, 1.0, 0.0, 1.0
        self.register_buffer("shift", torch.tensor([t_mid, p_mid]))
        self.register_buffer("scale", torch.tensor([t_half, p_half]))

    def forward(self, t, p):
        x = (torch.cat([t, p], dim=1) - self.shift) / self.scale
        return self.net(x)


class ShiftPINN(nn.Module):
    """u(t, c) = N(t - c): hard-codes time-translation symmetry.

    If f does not depend on t (autonomous ODE) and the parameter is the IC
    point c, then y(t; c) = Y(t - c) where Y solves the IVP with ICs at 0.
    The whole family is ONE trajectory, shifted, so the network only has to
    learn a function of one variable s = t - c on [t_min - c_max, t_max - c_min].
    """

    def __init__(self, problem, hidden=64, layers=4):
        super().__init__()
        if not problem.ic_at_param:
            raise ValueError("ShiftPINN needs the parameter to be the IC point (c_values configs)")
        self.net = mlp(1, 1, hidden, layers)
        s_lo, s_hi = problem.t_min - problem.p_max, problem.t_max - problem.p_min
        self.register_buffer("s_mid", torch.tensor((s_hi + s_lo) / 2))
        self.register_buffer("s_half", torch.tensor(max((s_hi - s_lo) / 2, 1e-12)))

    def forward(self, t, p):
        return self.net((t - p - self.s_mid) / self.s_half)


class HardIC(nn.Module):
    """Wraps a model so that the initial conditions hold exactly:

        u(t, p) = g0 + g1 (t - t0) + ((t - t0) / T)^2 N(t, p),   t0 = t0(p).

    u(t0) = g0 and u'(t0) = g1 for any N, so the IC penalties (and the alpha,
    beta weights) disappear. In particular u = 0 is no longer admissible
    when g0 != 0, which removes the trivial-solution trap of Van der Pol.
    """

    hard_ic = True

    def __init__(self, base, problem):
        super().__init__()
        self.base = base
        self.problem = problem
        self.T = problem.t_max - problem.t_min

    def forward(self, t, p):
        s = t - self.problem.ic_point(p)
        return self.problem.g0 + self.problem.g1 * s + (s / self.T) ** 2 * self.base(t, p)


def make_model(kind, problem, hidden=64, layers=4):
    """'legacy' | 'soft' | 'hard' | 'shift' | 'shift-hard'."""
    if kind == "legacy":
        return ParametricPINN(problem, hidden, layers, normalize=False)
    if kind == "soft":
        return ParametricPINN(problem, hidden, layers)
    if kind == "hard":
        return HardIC(ParametricPINN(problem, hidden, layers), problem)
    if kind == "shift":
        return ShiftPINN(problem, hidden, layers)
    if kind == "shift-hard":
        return HardIC(ShiftPINN(problem, hidden, layers), problem)
    raise ValueError(f"unknown model kind '{kind}'")


# --------------------------------------------------------------------------
# Residual and losses
# --------------------------------------------------------------------------
def derivatives(model, t, p):
    """u, u_t, u_tt at the points (t, p). t must require grad."""
    u = model(t, p)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, ones, create_graph=True)[0]
    return u, u_t, u_tt


def ode_residual(model, problem, t, p):
    """r = u_tt - f(t, u, u_t; p)."""
    u, u_t, u_tt = derivatives(model, t, p)
    f = problem.ode_func(t, [u, u_t], p)[1]
    return u_tt - f


def ic_loss(model, problem, p):
    t0 = problem.ic_point(p).detach().clone().requires_grad_(True)
    u = model(t0, p)
    u_t = torch.autograd.grad(u, t0, torch.ones_like(u), create_graph=True)[0]
    return torch.mean((u - problem.g0) ** 2), torch.mean((u_t - problem.g1) ** 2)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 3000
    lr: float = 1e-3
    n_t: int = 128            # collocation times per parameter
    n_p: int = 16             # parameters per step
    alpha: float = 10.0       # weight of (u(t0) - g0)^2
    beta: float = 10.0        # weight of (u'(t0) - g1)^2
    resample: bool = True     # draw fresh random collocation points every step
    colloc_range: tuple = None  # override [t_min, t_max]; (0, 1) reproduces the old bug
    causal_eps: float = 0.0   # > 0: two-sided causal weighting around t0 (see causal_pde_loss)
    causal_bins: int = 32
    curriculum: float = 0.0   # > 0: fraction of epochs over which the window around t0 grows to full
    seed: int = 0
    log_every: int = 500


def window_radius(problem, cfg, epoch):
    """Half-width of the admissible |t - t0| window at this epoch (curriculum)."""
    full = problem.t_max - problem.t_min
    if cfg.curriculum <= 0:
        return full
    frac = min(1.0, 0.05 + 0.95 * epoch / (cfg.curriculum * cfg.epochs))
    return frac * full


def sample_points(problem, cfg, step_rng, device, radius=None):
    t_lo, t_hi = cfg.colloc_range or (problem.t_min, problem.t_max)
    if cfg.resample:
        p = problem.p_min + (problem.p_max - problem.p_min) * torch.rand(cfg.n_p, 1, generator=step_rng)
        u = torch.rand(cfg.n_p, cfg.n_t, generator=step_rng)
    else:
        p = torch.linspace(problem.p_min, problem.p_max, cfg.n_p).reshape(-1, 1)
        u = torch.linspace(0, 1, cfg.n_t).expand(cfg.n_p, cfg.n_t)
    if radius is not None:
        # restrict each row to [t0 - radius, t0 + radius] intersected with the domain
        t0 = problem.ic_point(p)
        t_lo = torch.clamp(t0 - radius, min=t_lo)
        t_hi = torch.clamp(t0 + radius, max=t_hi)
    t = t_lo + (t_hi - t_lo) * u
    return t.to(device), p.to(device)


def causal_pde_loss(r2, dist, eps, n_bins):
    """Two-sided causal weighting (after Wang, Sankaran & Perdikaris 2022).

    The IC point t0 = c sits INSIDE the interval, so information propagates
    both forward and backward from it. Points are binned by d = |t - t0|;
    bin k gets weight exp(-eps * sum_{j<k} L_j) (no gradient through the
    weights), so the far field is only fitted once the near field is.
    Returns (loss, min weight).
    """
    d_max = dist.max().clamp_min(1e-12)
    idx = torch.clamp((dist / d_max * n_bins).long(), max=n_bins - 1).ravel()
    sums = torch.zeros(n_bins, device=r2.device).index_add(0, idx, r2.ravel())
    counts = torch.zeros(n_bins, device=r2.device).index_add(0, idx, torch.ones_like(r2.ravel()))
    L = sums / counts.clamp_min(1)
    occupied = counts > 0
    w = torch.exp(-eps * (torch.cumsum(L, 0) - L)).detach()
    loss = torch.sum(w * L * occupied) / occupied.sum()
    return loss, w[occupied].min().item()


def compute_loss(model, problem, cfg, t, p):
    """t: (n_p, n_t) collocation times, p: (n_p, 1). Returns (loss, parts dict)."""
    n_p, n_t = t.shape
    t_flat = t.reshape(-1, 1).requires_grad_(True)
    p_flat = p.expand(n_p, n_t).reshape(-1, 1)
    r = ode_residual(model, problem, t_flat, p_flat)
    if cfg.causal_eps > 0:
        dist = (t_flat - problem.ic_point(p_flat)).abs().detach()
        loss_pde, w_min = causal_pde_loss(r ** 2, dist, cfg.causal_eps, cfg.causal_bins)
    else:
        loss_pde = torch.mean(r ** 2)
    if getattr(model, "hard_ic", False):
        return loss_pde, {"pde": loss_pde.item(), "ic0": 0.0, "ic1": 0.0}
    l0, l1 = ic_loss(model, problem, p)
    loss = loss_pde + cfg.alpha * l0 + cfg.beta * l1
    return loss, {"pde": loss_pde.item(), "ic0": l0.item(), "ic1": l1.item()}


def train_pinn(model, problem, cfg, device="cpu", verbose=True):
    torch.manual_seed(cfg.seed)
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr / 20)
    history = []
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        radius = window_radius(problem, cfg, epoch) if cfg.curriculum > 0 else None
        t, p = sample_points(problem, cfg, rng, device, radius)
        loss, parts = compute_loss(model, problem, cfg, t, p)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            parts.update(epoch=epoch, loss=loss.item(), seconds=time.time() - t_start)
            history.append(parts)
            if verbose:
                print(f"  epoch {epoch:6d}  loss {loss.item():.3e}  pde {parts['pde']:.3e}  "
                      f"ic {parts['ic0']:.1e}/{parts['ic1']:.1e}")
    return history


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, t_eval, p_val, device="cpu"):
    t = torch.as_tensor(np.asarray(t_eval), dtype=torch.float32, device=device).reshape(-1, 1)
    p = torch.full_like(t, float(p_val))
    return model(t, p).cpu().numpy().ravel()


def evaluate(model, problem, n_p=21, n_t=400, device="cpu"):
    """Relative L2 error against the SciPy reference for n_p parameters in range."""
    t_eval = np.linspace(problem.t_min, problem.t_max, n_t)
    p_vals = np.linspace(problem.p_min, problem.p_max, n_p)
    errors = []
    for p in p_vals:
        t0 = p if problem.ic_at_param else problem.t0_fixed
        y_ref = reference_solution(problem.ode_func, problem.g0, problem.g1, t0, t_eval, param=p)[0]
        errors.append(relative_l2(predict(model, t_eval, p, device), y_ref))
    return p_vals, np.array(errors)
