import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pinn_core import (  # noqa: E402
    ParametricPINN,
    Problem,
    TrainConfig,
    compute_loss,
    derivatives,
    ic_loss,
    load_problem,
    ode_residual,
)


def harmonic(t, y, p):
    return [y[1], -y[0]]


HARMONIC = Problem(harmonic, g0=1.0, g1=0.0, t_min=0.0, t_max=6.0, p_min=0.0, p_max=2.0)


class Exact(torch.nn.Module):
    """u(t, c) = cos(t - c): exact solution of the harmonic family with y(c)=1, y'(c)=0."""

    def forward(self, t, p):
        return torch.cos(t - p)


def _points(n=50):
    t = torch.linspace(0, 6, n, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    p = torch.full_like(t, 0.7).detach()
    return t, p


def test_derivatives_of_exact_solution():
    t, p = _points()
    u, u_t, u_tt = derivatives(Exact(), t, p)
    assert torch.allclose(u_t, -torch.sin(t - p))
    assert torch.allclose(u_tt, -torch.cos(t - p))


def test_exact_solution_has_zero_loss():
    t, p = _points()
    assert ode_residual(Exact(), HARMONIC, t, p).abs().max() < 1e-12
    l0, l1 = ic_loss(Exact(), HARMONIC, torch.tensor([[0.3], [1.9]], dtype=torch.float64))
    assert l0 < 1e-20 and l1 < 1e-20


def test_compute_loss_shapes_and_positivity():
    torch.manual_seed(0)
    model = ParametricPINN(HARMONIC, hidden=16, layers=2)
    t = torch.rand(4, 8) * 6
    p = torch.rand(4, 1) * 2
    loss, parts = compute_loss(model, HARMONIC, TrainConfig(), t, p)
    assert loss.ndim == 0 and loss.item() > 0
    assert set(parts) == {"pde", "ic0", "ic1"}


def test_input_normalisation_maps_domain_to_unit_box():
    model = ParametricPINN(HARMONIC)
    x = (torch.tensor([[0.0, 0.0], [6.0, 2.0]]) - model.shift) / model.scale
    assert torch.allclose(x, torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))


def test_configs_load_as_problems():
    for name in ("data_augmentation_van_der_pol.py", "data_augmentation_pendulum.py",
                 "data_augmentation_rayleigh.py"):
        pb = load_problem(os.path.join(HERE, name))
        assert pb.ic_at_param and pb.t_max > pb.t_min
        y = [torch.zeros(3, 1), torch.zeros(3, 1)]
        assert pb.ode_func(torch.zeros(3, 1), y, torch.zeros(3, 1))[1].shape == (3, 1)
