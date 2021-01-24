from jax.config import config
config.update('jax_enable_x64', True)
import jax
from jax import jit, jacrev, jacfwd
from jax import numpy as numpy
from jax.scipy.optimize import minimize as jminimize
import scipy.optimize

_lfns={"Gauss" : lambda delta, w: -1*(delta**2/w**2).sum(),
       }

def mlft(Y,
         X=None,
         k=None,
         w=None,
         reg=None,
         lfn="Gauss"):
    """
    Maximum likelihood Fourier Transform estimator
    """
    if X is None: X=numpy.linspace(0, 2*numpy.pi, Y.shape[-1])
    if k is None: k=numpy.arange(0, X.size/2)
    if w is None: w=numpy.ones_like(X)
    cm=numpy.cos(numpy.outer(k, X) )
    sm=numpy.sin(numpy.outer(k, X) )
    mm=numpy.vstack([cm, sm])
    def ll(x):
        delta=Y-numpy.matmul(mm.transpose(), x)
        delta=delta*w
        ll=_lfns[lfn](delta, w)
        return -1*ll
    jac=jit(jacrev(ll))
    r=scipy.optimize.minimize(ll, numpy.zeros_like(X))
    return r
    

