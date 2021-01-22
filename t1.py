import timeit
import numpy
import scipy.optimize

X=numpy.linspace(0, 2*numpy.pi, 256)

Y=numpy.sin(X*3)

Yp=Y+numpy.random.normal(scale=1.0, size=(100, Y.size) )

def mkLL(X):
    k=numpy.arange(0, X.size/2)
    cm=numpy.cos(numpy.outer(k, X) )
    sm=numpy.sin(numpy.outer(k, X) )
    mm=numpy.vstack([cm, sm])
    def ll(x):
        delta=Yp-numpy.matmul(mm.transpose(), x)
        #ll=numpy.log(1/(1+delta**2)).sum()
        ll=-1*(delta**2).sum()
        return -1*ll
    return ll

f=mkLL(X)
if 0:
    r=scipy.optimize.minimize(f, numpy.zeros_like(X))


timeit.timeit("scipy.optimize.minimize(f, numpy.zeros_like(X))", globals=globals() , number=1)
