# positive semidefine matrix
import numpy as np
from scipy import optimize

from ..qmath import square_matrix_dim, dot3, DTYPE


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def lstsq(err_func, T, unit_trace=True, real=True, disp=True, method='CG',
          options={}):

    dim = square_matrix_dim(T)

    # numpy 1.4.0+
    indices_re = np.tril_indices(dim)
    indices_im = np.tril_indices(dim, -1)

    n_re = len(indices_re[0])  # dim*(dim+1)/2

    def gamma(ts):
        G = np.zeros((dim, dim), dtype=DTYPE)
        G[indices_re] = ts[:n_re]
        if not real:
            G[indices_im] += 1j * ts[n_re:]
        return G

    def ungamma(G):
        if real:
            max_imaginary = np.max(G[indices_im].imag)
            # assert max_imaginary < 0.1, max_imaginary
            return G[indices_re].real
        else:
            return np.hstack((G[indices_re].real, G[indices_im].imag))

    def ts_to_mat(ts):
        G = gamma(ts)
        GtG = np.dot(G, G.conj().T)
        if unit_trace:
            return GtG / np.trace(GtG)
        else:
            return GtG

    def mat_to_ts(mat):
        # convert mat into t vector using cholesky decomposition
        # Since it only works if the matrix is positive and hermitian,
        # we diagonalize and fix up the eigenvalues beforehand.
        eigs, vals = np.linalg.eigh(mat)
        eigs = eigs.real
        eigs[eigs < 0.01] = 1E-4
        if unit_trace:
            eigs = eigs / np.sum(eigs)
        mat = dot3(vals, np.diag(eigs), vals.conj().T)
        G = np.linalg.cholesky(mat)
        ts = ungamma(G)
        return ts

    def loss_func(ts):
        mat = ts_to_mat(ts)
        return err_func(mat)

    ts_guess = mat_to_ts(T)

    # find the local minimum of the loss_func function
    # Nelder-Mead simple algorithm
    one_of_ts = np.ones_like(ts_guess)
    bound = optimize.Bounds(-one_of_ts, one_of_ts)
    # BFGS
    ts = optimize.minimize(loss_func, ts_guess, method=method, bounds=bound,
                           options=options)
    if disp:
        print(ts['message'], 'sucess: ', ts['success'])
    ts = ts['x']
    # ts = optimize.fmin_bfgs(loss_func, ts_guess)
    # quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
    # ts = optimize.fmin_bfgs(loss_func, ts_guess)

    # reconstruct the matrix from the parameteried form
    return ts_to_mat(ts)
