# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# |         Based on Matthew Gary Neeley's thesis (2010)               |
# |         A.1 State Tomography (page 144+ )                          |
# ----------------------------------------------------------------------
#
# Measurement results after the jth tomo operator on the kth state is:
#
#               $ P_jk = (ρ_j)_kk = ⟨k|U_j ρ U†_j|k⟩ $
#
# where j is the number of rotations, while k is the number of diagonal
# elements of the density matrix measured. In other word, k is the
# dimension of the Unitary Transform.
# e.g. qubit n = 2, j = 3^n = 9; k = 2^n = 4, i.e. 00, 01, 10, 11
# Total number of the measurement is j*k = 9*4 = 36.
#
# To easily determine the density matrix ρ_mn, we use Roth's lemma to
# convert the superoperator to left-multiplying matrix operator:
#
#           $ vec(UρU†) = (U†.T⊗U)vec(ρ) = (U*⊗U)vec(ρ) $
#           $ vec(⟨k|UρU†|k⟩) = (U*⊗U)_kk_mn vec(ρ_mn) $
#
# Now that we use 3^n·2^n = 6^n input measurements to find 4^n elements
# of the density matrix, so that this inversion is typically done as a
# least-squares optimization to find the closest fit solution.
#
# It is easily seen that the tomo basis blows up exponentially with n.
# e.g. qubit n = 4, m = n = k = 16, (U*⊗U).shape = (256, 256)

import numpy as np
from scipy import optimize

from .. import qmath, qops

DTYPE = qmath.CPLX_TYPE

######################
# qubit basis matrix #
######################

_QST_TRANSFORMS = {}  # initialized tomography protocols


def qst_basis_mat(Us, key=None):
    """
    A set of unitaries (rotation) operations applied before the
    projective measurements

    Args:
        Us: <list> of unitary operations that will be applied to the
        state before projective measurements

        key: <string> as the key of _QST_TRANSFORMS dict under which
        this tomography protocol will be stored so it can be referred
        to without recomputing the transformation matrix

    Returns:
        <array> as the basis matrix that should be passed to qst() along
        with the measurement data to perform state tomography
    """

    Us = np.asarray(Us)
    num_Us = len(Us)  #                    --> j
    n_diag = qmath.square_matrix_dim(Us[0])  #   --> k

    Um = np.array([
        qmath.super2mat(Us[j, k, :], Us[j, k, :].conj()) for j in range(num_Us)
        for k in range(n_diag)
    ], dtype=DTYPE)
    # Note the order of qmath.super2mat does not matter if Us is just a vector

    # save this transform if a key was provided
    if key is not None:
        _QST_TRANSFORMS[key] = (Us, Um)

    return Um


TOMO_BASIS_OPS = {
    'tomo': ['I', 'X/2', 'Y/2'],
    # QST_PMs = ['Pz+', 'Pz-', 'Py+', 'Py-', 'Px-', 'Px+']
    'octomo': ['I', 'X/2', 'Y/2', '-X/2', '-Y/2', 'X'],
    'nonetomo':
    ['I', 'XY', 'YX', '-XY', '-YX', 'XY/2', 'YX/2', '-XY/2', '-YX/2'],
    'smtc_tomo':
    ['I', 'X/2', 'Y/2', 'XY/2', 'YX/2', '-Y/4', 'X/4', 'Y/4', '-X/4'],
    # QST_PMs = ['Pz+', 'Pz-', 'Py+', 'Py-', 'Px-', 'Px+',
    #            'Pyx+', 'Pyx-', 'Pxy-', 'Pxy+', 'Pxz+', 'Pxz-', 'Pyz+', 'Pyz-',
    #            'Pzx+', 'Pzx-', 'Pzy+', 'Pzy-']
    'pm_tomo': ['Pz+', 'Py+', 'Px-'],
    'pm_full': list(qops.PM_DICT.keys())[:9],
    'pm_smtc': list(qops.PM_DICT.keys())
}
for k, v in TOMO_BASIS_OPS.items():
    # 1-qubit
    qst_basis_mat(qops.get_ops(v), k)
    # 2-qubit
    qst_basis_mat(qmath.tensor_combinations(qops.get_ops(v), 2),
                  k + '_2qubits')
    # # 3-qubit
    # qst_basis_mat(qmath.tensor_combinations(qops.get_ops(v), 3),
    #               k + '_3qubits')

# # Qutrit(3-level) tomo

# tomo3_ops = [("I", "I"), ("X/2", "I"), ("Y/2", "I"), ("X", "I"), ("I", "X/2"),
#              ("I", "Y/2"), ("X", "X/2"), ("X", "Y/2"), ("X", "X")]

# octomo3_ops = [("I", "I"), ("X/2", "I"), ("Y/2", "I"), ("-X/2", "I"),
#                ("-Y/2", "I"), ("X/2", "X"), ("Y/2", "X"), ("-X/2", "X"),
#                ("-Y/2", "X"), ("X", "I"), ("I", "X/2"), ("I", "Y/2"),
#                ("I", "-X/2"), ("I", "-Y/2"), ("I", "X"), ("X", "X/2"),
#                ("X", "Y/2"), ("X", "-X/2"), ("X", "-Y/2"), ("X", "X")]

# tomo3_us = gen_tomo3_us(tomo3_ops)
# qst_basis_mat(tomo3_us, 'tomo3lvl')

# octomo3_us = gen_tomo3_us(octomo3_ops)
# qst_basis_mat(octomo3_us, 'octomo3lvl')

############################
# quantum state tomography #
############################


def qst(probs, Um, return_all=False):
    """Fit a density matrix from tomo measurements

    Args:
        probs: <2D array> an array of probabilities after acting the
            qst protocol, which is actually the diagonal elements of
            rotated density matrix. The first index indicates which tomo
            operation was applied before measurement, while the second
            tells which measurement result this is.

        Um: a set of vectorized tomo-rotation operator
        <list of matrix>: tomo basis matrix generated from qst_basis_mat
        <string of a tomo protocol>: name of the same qst protocol as probs

        return_all: boolean, whether to return all from linalg.lstsq()
    """
    if isinstance(Um, str) and Um in _QST_TRANSFORMS:
        Um = _QST_TRANSFORMS[Um][1]

    probs = np.asarray(probs)
    m = probs.shape[1]

    # keep the old version by setting rcond=-1
    # the new rcond default is of machine precision times ``max(M, N)``
    # where M and N are the input matrix dimensions.
    rho_vec, resids, rank, s = np.linalg.lstsq(Um, probs.flatten(), rcond=-1)

    if return_all:
        return rho_vec.reshape((m, m)), resids, rank, s
    else:
        return rho_vec.reshape((m, m))


def qst_mle(probs, Us, F=None, rho0=None, full_output=False, disp=True):
    """State tomography with maximum-likelihood estimation

    rho shall satisfy 3 constraints to be physical meaningful:
    (1) Hermitian (ρ† = ρ);
    (2) positive semidefinite (⟨ψ| ρ |ψ⟩ ≥ 0 for all |ψ⟩
    (3) unit trace (Tr ρ = 1)

    Args:
        probs: <array[i,j]> an array of probabilities after acting the
            qst protocol. Index i indicates which tomo operation was
            applied before measurement, while j tells measurement result.

        Us: a set of tomo-rotation operator
            <list of matrix>: unitary operations applied before measurement
            <string of a tomo protocol>: name of the same qst protocol as probs

        F: Fidelity matrix: P_m = F·P_i, where P_m is the measured probs
            and P_i is the 'intrinsic' probs.
            If fidelity matrix is None, then identity matrix will be used.

        rho0: an initial guess for the density matrix
            If None use the result qst(probs)

    Returns:
        optimized rho that satisfies the above 3 conditions

    Reference:
    Matthew Gary Neeley's thesis (2010)
    A.3.1 Maximum Likelihood Estimation (page 153+):

    It can be shown any ρ which is Hermitian, positive-semidefinite,
    and unit-trace can be defined as ρ = (Γ.T · Γ) / Tr(Γ.T · Γ), where:
        (t0                                         )
        |t1+i*t2    t3                              |
    Γ ≡ |t4+i*t5    t6+i*t7     t8                  |
        |t9+i*t10   t11+i*t12   t13+i*t14   t15  .  |
        (  ...                                   ...)

    We can thus find the maximum-likelihood estimation by optimizing the
    parameters t_i w.r.t the likelihood L(t_0, t_1, ..., t_i; probs),
    where experimental data probs is the parameter of the distribution L
    """
    if isinstance(Us, str) and Us in _QST_TRANSFORMS:
        Us = _QST_TRANSFORMS[Us][0]

    m = qmath.square_matrix_dim(Us[0])
    F = np.eye(m) if F is None else F

    # numpy 1.4.0+
    indices_re = np.tril_indices(m)
    indices_im = np.tril_indices(m, -1)

    n_re = len(indices_re[0])  # m*(m+1)/2

    def gamma(ts):
        G = np.zeros((m, m), dtype=DTYPE)
        G[indices_re] = ts[:n_re]
        G[indices_im] += 1j * ts[n_re:]
        return G

    def ungamma(G):
        return np.hstack((G[indices_re].real, G[indices_im].imag))

    def ts_to_rho(ts):
        G = gamma(ts)
        GtG = np.dot(G, G.conj().T)
        return GtG / np.trace(GtG)

    def rho_to_ts(rho):
        # convert rho into t vector using cholesky decomposition
        # Since it only works if the matrix is positive and hermitian,
        # we diagonalize and fix up the eigenvalues beforehand.
        eigs, vals = np.linalg.eigh(rho)
        eigs = eigs.real
        eigs[eigs < .01] = 0.01
        eigs = eigs / np.sum(eigs)
        rho = qmath.dot3(vals, np.diag(eigs), vals.conj().T)
        G = np.linalg.cholesky(rho)
        ts = ungamma(G)
        return ts

    # Begin of the Algorithm

    # 1. initial guess of params, from the experimental data Probs using
    # tomography and cholesky decomposition
    if rho0 is None:
        F_inv = np.linalg.inv(F)
        probs = np.dot(probs, F_inv)
        rho0 = qst(probs, qst_basis_mat(Us))

    ts_guess = rho_to_ts(rho0)

    # 2. construct the likelihood distribution function
    Us = np.array(Us)
    Uds = np.transpose(Us.conj(), axes=(0, 2, 1))

    def unlikelihood(ts):
        rho = ts_to_rho(ts)
        # using einsum, faster than np.dot(F, np.diag(qmath.dot3(U, rho, Ud)))
        p_int = np.dot(np.einsum('aij,jk,aki->ai', Us, rho, Uds).real, F.T)
        terms = probs * qmath.safe_log(p_int) + (
            1 - probs) * qmath.safe_log(1 - p_int)
        # terms = probs * qmath.safe_log(p_int)
        # I comment the second term, It sum(probs) of each measure is 1,
        # the second term is the same with the first after sum.
        # In this scheme, data should not be corrected, and F should be inputed.
        # Also, 3-level tomo_mle can be supported. --ZZX, 2018.05.12
        return -np.mean(terms.flat)

    # 3. find the local minimum of the unlikelihood function
    # Nelder-Mead simplex algorithm
    ts = optimize.fmin(unlikelihood, ts_guess, full_output=full_output,
                       disp=disp)
    # quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
    # ts = optimize.fmin_bfgs(unlikelihood, ts_guess)

    # 4. reconstruct the physical meaningful rho from parameter ts
    rho = ts_to_rho(ts)

    # End of the Algorithm
    return rho


def probs_to_rhos(probs, tomo_names, qst_name):
    """Get rhos from sets of probs using qst()

    Args:
        probs: <list of <prob set>> It can be grouped to constitute rhos.
            <prob set> is formatted s.t.
            probs[0] = [p0, p1] <= 1 qubit
            probs[0] = [p00, p01, p10, p11] <= 2 qubits

            len(prob group for 1 rho) = num_Us^num_qubits, where num_Us
            is the num of tomo operators.

            If probs are generated from a qpt experiment, then
            len(probs) = len(group of rho)^2
    """
    num_Us = len(tomo_names)
    num_qubits = int(np.log(len(probs)) / np.log(num_Us) / 2)

    ops_per_rho = num_Us**num_qubits

    rhos = []
    while len(probs) > 0:
        curr_probs = [probs.pop(0) for k in range(ops_per_rho)]
        rho = qst(curr_probs, qst_name)
        rhos.append(rho)

    return rhos


###########
## tests ##
###########


def test_qst(n=256):
    """
    Generate a bunch of random states, and check that we recover them
    from state tomography
    """
    def random_rho_1qubit():
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        mag = np.random.uniform(0, 1.0)
        return qmath.polar2rho(theta, phi, mag)

    def rho_to_probs(rho, Us):
        return np.vstack([np.diag(qmath.dot3(U, rho, U.conj().T)) for U in Us])

    def test_qst_protocol(proto, use_mle=False):
        Us = _QST_TRANSFORMS[proto][0]

        if not use_mle:
            # Note this initial rho is not physical.
            # Although it can be used to test qst() it is not suitable for qst_mle
            rho_orig = (np.random.uniform(-1, 1, Us[0].shape) +
                        1j * np.random.uniform(-1, 1, Us[0].shape))
            probs = rho_to_probs(rho_orig, Us)
            rho_calc = qst(probs, proto)
        else:
            n_states = qmath.square_matrix_dim(Us[0])
            num_qubits = int(np.log(n_states) / np.log(2))
            assert 2**num_qubits == n_states, "only generate rho of qubits"
            if num_qubits >= 4:
                return np.NaN
            rho_orig = qmath.tensor(
                [random_rho_1qubit() for _ in range(num_qubits)])
            probs = rho_to_probs(rho_orig, Us)
            rho_calc = qst_mle(probs, proto, full_output=False, disp=False)

        return np.max(np.abs(rho_orig - rho_calc))

    # ---------------------------- standard tests -----------------------------
    for i in range(4):
        # i+1 qubits
        stats = n // 4**i
        et_n = [test_qst_protocol('tomo') for _ in range(stats)]
        eo_n = [test_qst_protocol('octomo') for _ in range(stats)]
        eo_pm_n = [test_qst_protocol('octomo_pm') for _ in range(stats)]
        en_n = [test_qst_protocol('nonetomo') for _ in range(stats)]
        print('{}-qubit error :\n '
              'tomo={:.3g}, octomo={:.3g}, octomo_pm={:.3g}, nonetomo={:.3g}'.
              format(i + 1, max(et_n), max(eo_n), max(eo_pm_n), max(en_n)))

    # ------------------------------- mle tests -------------------------------
    for i in range(4):
        # i+1 qubits
        stats = 2**(3 - i)
        et_n = [test_qst_protocol('tomo', True) for _ in range(stats)]
        eo_n = [test_qst_protocol('octomo', True) for _ in range(stats)]
        eo_pm_n = [test_qst_protocol('octomo_pm') for _ in range(stats)]
        en_n = [test_qst_protocol('nonetomo', True) for _ in range(stats)]
        print('{}-qubit error (maximum-likelihood): \n'
              ' tomo={:.3g}, octomo={:.3g}, octomo_pm={:.3g}, nonetomo={:.3g}'.
              format(i + 1, max(et_n), max(eo_n), max(eo_pm_n), max(en_n)))


if __name__ == '__main__':
    test_qst()
