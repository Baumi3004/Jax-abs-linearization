import os
import sys
parent = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent)
from JaxAbsLinearization import AbsLinearForm
import jax.numpy as jnp
import jax

import pytest
# RN
def rosenbrock_nesterov(x):
    n = x.size
    y = 1/4*jnp.abs(x[0]- 1)
    for i in range(n-1):
        y = y + jnp.abs(x[i+1] - 2*jnp.abs(x[i]) + 1)
    return y

def test_abslinform_rosenbrock_nesterov():

    x0 = jnp.zeros(4, dtype=jnp.float32)
    ALF_rosenbrock = AbsLinearForm(rosenbrock_nesterov, x0)
    y0, z0, a, b, Z, L = ALF_rosenbrock(x0)
    
    assert y0 == pytest.approx(3.25)
    assert z0 == pytest.approx(jnp.array([-1., 0., 1., 0., 1., 0., 1.]))
    assert a  == pytest.approx(jnp.zeros(4))
    assert b  == pytest.approx(jnp.array([0.25, 0., 1., 0., 1., 0., 1.]))
    assert Z  == pytest.approx(jnp.array([[1.,0.,0.,0.], [1.,0.,0.,0.], [0.,1.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]]))
    assert L[2, 1] == pytest.approx(-2.)
    assert L[4, 3] == pytest.approx(-2.)
    assert L[6, 5] == pytest.approx(-2.)
    assert jnp.linalg.norm(L.flatten(), ord=1) == pytest.approx(6.)


# Hill funciton (see Kreimeier Diss Example 3.3)
def relu(x):
    return 0.5*(x+jnp.abs(x))

def hill(x):
    return relu( x[0]- jnp.abs(x[1]))
    
def test_abslinear_hill():

    x0 = jnp.zeros(2)
    ALF_hill = AbsLinearForm(hill, x0)
    y0, z0, a, b, Z, L = ALF_hill(x0)
    assert y0 == pytest.approx(0.)
    assert z0 == pytest.approx(jnp.array([0.,0.]))
    assert a  == pytest.approx(jnp.array([0.5,0]))
    assert b  == pytest.approx(jnp.array([-0.5,0.5]))
    assert Z  == pytest.approx(jnp.array([[0.,1.],[1.,0.]]))
    assert L  == pytest.approx(jnp.array([[0.,0.],[-1.,0.]]))
