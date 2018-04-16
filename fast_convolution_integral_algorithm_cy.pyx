import numpy as np

DTYPE = np.float64
cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calcAlpha_cython(double [:,:] phi,double [:] fix_phis):
    cdef Py_ssize_t l,j,x,y
    cdef Py_ssize_t xmax = phi.shape[0]
    cdef Py_ssize_t ymax = phi.shape[1]
    cdef Py_ssize_t ljmax = fix_phis.shape[0]
    alpha_np = np.ones((ljmax,xmax,ymax), dtype=DTYPE)
    cdef double [:,:,:] alpha = alpha_np
    for x in range(xmax):
        for y in range(ymax):
            for l in range(ljmax):
                alpha[l,x,y]*=(2.0*(phi[x,y]*phi[x,y])/((phi[x,y]*phi[x,y])-(fix_phis[l]*fix_phis[l])))
                for j in range(ljmax):
                    if j != l:
                        alpha[l,x,y]*=((phi[x,y]*phi[x,y]) - (fix_phis[j]*fix_phis[j]))/((fix_phis[l]*fix_phis[l]) + (fix_phis[j]*fix_phis[j]))
                        alpha[l,x,y]*=((fix_phis[l]*fix_phis[l]) + (fix_phis[j]*fix_phis[j]))/((phi[x,y]*phi[x,y]) + (fix_phis[j]*fix_phis[j]))
    return alpha

