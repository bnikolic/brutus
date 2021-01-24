from jax.config import config
config.update('jax_enable_x64', True)
import jax
from jax import jit, jacrev, jacfwd
from jax import numpy as numpy
from jax.scipy.optimize import minimize as jminimize
import scipy.optimize

_lfns={"Gauss" : lambda delta, w: -1*(delta**2/w**2).sum(),
       "Cauchy": lambda delta, w: numpy.log(w**2/(w**2+delta**2)).sum()
       }

def mkll(Y, mm, w, lfn):
    lfn=_lfns[lfn]
    def ll(x):
        delta=Y-numpy.matmul(mm.transpose(), x)
        ll=lfn(delta, w)
        return -1*ll
    return ll

def mlft(Y,
         X=None,
         k=None,
         w=None,
         reg=None,
         lfn="Gauss",
         check=True):
    """
    Maximum likelihood Fourier Transform estimator
    """
    N=Y.shape[-1]
    if X is None: X=numpy.linspace(0, 2*numpy.pi, N)
    if k is None: k=numpy.arange(0, X.size/2)
    if w is None: w=numpy.ones_like(X)
    cm=numpy.cos(numpy.outer(k, X) )
    sm=numpy.sin(numpy.outer(k, X) )
    mm=numpy.vstack([cm, sm])
    ll=jit(mkll(Y, mm, w, lfn))
    jac=jit(jacrev(ll))
    r=scipy.optimize.minimize(ll,
                              numpy.zeros_like(X),
                              jac=jac,
                              method="BFGS",)

    if check and not r.success:
        print (r)
        raise RuntimeError("Did not converge")
    ff=r.x[0:N//2]+1j*r.x[N//2:]
    return ff
    

