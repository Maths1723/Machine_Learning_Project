"""Data-driven ("hybrid") baseline from the 2025 report, on the same footing as the PINNs.

An MLP maps the parameter p to the whole solution vector on a fixed time grid,
trained on SciPy solutions for n_train parameters (as in SIMPLE_ode_support.ODE_MLP,
ReLU, 3 x 128). The wrapper exposes the PINN interface model(t, p) by linear
interpolation in t, so `pinn_core.evaluate` scores both kinds of model identically.
"""

import time

import numpy as np
import torch
import torch.nn as nn

from numerics import reference_solution


class SolutionMLP(nn.Module):
    def __init__(self, n_out, hidden=128, layers=3):
        super().__init__()
        mods = [nn.Linear(1, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.ReLU()]
        mods.append(nn.Linear(hidden, n_out))
        self.net = nn.Sequential(*mods)

    def forward(self, p_hat):
        return self.net(p_hat)


class SurrogateModel(nn.Module):
    """model(t, p) -> u(t, p) for a column of t at ONE parameter value (as in evaluate/predict)."""

    def __init__(self, problem, t_grid, hidden=128, layers=3):
        super().__init__()
        self.problem = problem
        self.register_buffer("t_grid", torch.as_tensor(t_grid, dtype=torch.float32))
        self.mlp = SolutionMLP(len(t_grid), hidden, layers)
        self.p_mid = (problem.p_max + problem.p_min) / 2
        self.p_half = max((problem.p_max - problem.p_min) / 2, 1e-12)

    def curves(self, p):
        return self.mlp((p.reshape(-1, 1) - self.p_mid) / self.p_half)

    def forward(self, t, p):
        if not torch.all(p == p[0]):
            raise ValueError("SurrogateModel evaluates one parameter value per call")
        curve = self.curves(p[:1])[0]
        tq = t.reshape(-1).clamp(self.t_grid[0], self.t_grid[-1])
        idx = torch.clamp(torch.searchsorted(self.t_grid, tq) - 1, 0, len(self.t_grid) - 2)
        w = (tq - self.t_grid[idx]) / (self.t_grid[idx + 1] - self.t_grid[idx])
        return ((1 - w) * curve[idx] + w * curve[idx + 1]).reshape(-1, 1)


def make_dataset(problem, n_train, t_grid):
    p_train = np.linspace(problem.p_min, problem.p_max, n_train)
    Y = np.stack([
        reference_solution(problem.ode_func, problem.g0, problem.g1,
                           p if problem.ic_at_param else problem.t0_fixed, t_grid, param=p)[0]
        for p in p_train
    ])
    return torch.as_tensor(p_train, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)


def train_surrogate(problem, n_train, epochs=5000, lr=1e-3, n_t=400, seed=0, verbose=False):
    """Returns (model, info). info['data_seconds'] is the cost of the SciPy solves."""
    torch.manual_seed(seed)
    t_grid = np.linspace(problem.t_min, problem.t_max, n_t)
    t0 = time.time()
    p_train, Y = make_dataset(problem, n_train, t_grid)
    data_seconds = time.time() - t0
    model = SurrogateModel(problem, t_grid)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = torch.mean((model.curves(p_train) - Y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and epoch % max(epochs // 5, 1) == 0:
            print(f"  surrogate epoch {epoch:6d}  mse {loss.item():.3e}")
    return model, {"train_mse": loss.item(), "data_seconds": data_seconds,
                   "train_seconds": time.time() - t0}
