
import os
import sys
parent = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent)
from JaxAbsLinearization import AbsLinearForm
import jax.numpy as jnp
import jax
import pytest


def fun(x):
    return jnp.sum(jnp.maximum(0, x[:2]-x[2:]))
    #return jnp.max(x[:2]-x[2:])


def test_abslinear_maximum():
    x0 = jnp.zeros(4)
    ALF_fun = AbsLinearForm(fun, x0)
    y0, z0, a, b, Z, L = ALF_fun(x0)
    
    assert y0 == pytest.approx(jnp.array(0.))
    assert z0 == pytest.approx(jnp.array([0., 0.]))
    assert a  == pytest.approx(jnp.array([0.5, 0.5, -0.5, -0.5]))
    assert b  == pytest.approx(jnp.array([0.5, 0.5]))
    assert Z  == pytest.approx(jnp.array([[-1., 0., 1., 0.], [0., -1., 0., 1.]]))
    assert jnp.linalg.norm(L) == pytest.approx(0.)

test_abslinear_maximum()