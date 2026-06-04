import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numerics import reference_solution, relative_l2  # noqa: E402


def harmonic(t, y, p):
    return [y[1], -y[0]]


def test_harmonic_ic_inside_interval():
    # y'' = -y, y(c) = 1, y'(c) = 0  ->  y = cos(t - c), on both sides of c
    c = 2.3
    t = np.linspace(0, 10, 301)
    y = reference_solution(harmonic, 1.0, 0.0, c, t)
    assert np.allclose(y[0], np.cos(t - c), atol=1e-8)
    assert np.allclose(y[1], -np.sin(t - c), atol=1e-8)


def test_ic_at_left_end_and_outside():
    t = np.linspace(0, 5, 101)
    for c in (0.0, -1.0, 6.0):
        y = reference_solution(harmonic, 0.0, 1.0, c, t)
        assert np.allclose(y[0], np.sin(t - c), atol=1e-8)


def test_relative_l2():
    a = np.ones(10)
    assert relative_l2(a, a) == 0.0
    assert np.isclose(relative_l2(1.1 * a, a), 0.1)


def test_surrogate_interpolates_and_fits_harmonic_family():
    import torch

    from pinn_core import Problem, evaluate
    from surrogate import train_surrogate

    torch.manual_seed(0)
    pb = Problem(harmonic, g0=1.0, g1=0.0, t_min=0.0, t_max=6.0, p_min=0.0, p_max=1.0)
    model, info = train_surrogate(pb, n_train=16, epochs=1500, n_t=120)
    assert info["train_mse"] < 1e-2
    # interface: column of t at one parameter value, same as the PINNs
    t = torch.linspace(0, 6, 7).reshape(-1, 1)
    assert model(t, torch.full_like(t, 0.3)).shape == (7, 1)
    _, err = evaluate(model, pb, n_p=5, n_t=120)
    assert np.median(err) < 0.2
