# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# |         Based on Matthew Gary Neeley's thesis (2010)               |
# |         A.2 Process Tomography (page 148+ )                        |
# ----------------------------------------------------------------------
#
# The quantum operation E is describe as a mapping (E) from an input
# density matrix ρ to an output density matrix ρ':
#
# In the χ-matrix representation, we pick a basis of unitary operators
# {A}, which span the space of all possible unitaries:
#
#               $ ρ' = E(ρ) = SUM_jk{χ_jk A_j·ρ·(A_k)†} $
#
# From the measured input ρ and output ρ', we want to determine χ:
#
#           $ (ρ'_i)_mn = χ_jk·(A_j)_mp·(ρi)_pq·((A_k)†)_qn $
#
# where i is the tomo-operator index labelling the different states that
# we prepared (ρ) and later measured (ρ') after the quantum operation
# which we wish to characterize.
#
# Again, we use a variant of Roth's lemma to define the "pointer-basis"
# χ-matrix in the vectorized space, which can be easily fit by a
# least-squares inversion, as we did in the QST algorithm:
#
#                   $ vec(ρ')_iβ = vec(ρ)_iα · Χ_αβ $
#       $ Χ_αβ = T_αβ_jk · χ_jk = ((A_j)_mp ⊗ (A_k)†_qn) · χ_jk $
#
# where α <- pq and β <- mn are the new indexes after vectorization.
# e.g.
# 2 qubits, m=n=4; α=β=16; Χ.shape=χ.shape=(16,16); T.shape=(256,256)
# 3 qubits, m=n=8; α=β=64; Χ.shape=χ.shape=(64,64); T.shape=(4096,4096)
#
# Another thing worth mentioning is the operator-sum representation,
# which can be found by diagonalizing the χ-matrix:
#
#                  $ ρ' = E(ρ) = SUM_k{E_k·ρ·(E_k)†} $

import itertools
import numpy as np

from .. import qmath, qops
from ..qmath import (sigma_i, sigma_x, sigma_y, sigma_z, sigma_P, sigma_M,
                     expander, super2mat, mat2vec, vec2mat, dot3, rotate,
                     tensor_combinations, square_matrix_dim, tensor, CPLX_TYPE)

CPLX_DTYPE = CPLX_TYPE

##############################
# Quantum Process Tomography #
##############################


def default_basis(rho_in):
    basis = 'pauli-vector'
    if square_matrix_dim(rho_in) != 2:
        basis += '_{}qubits'.format(int(np.log2(square_matrix_dim(rho_in))))
    return basis


def chi_pointer_mat(rhos_in, rhos_out, return_all=False):
    """Calculates the chi-matrix in the pointer-basis

                        ρ' = ρ · χ

    Args:
        rhos_in: <list of array> of input density matrices
        rhos_out: <list of array> of output density matrices

        return_all: boolean, whether to return all from linalg.lstsq()
    """

    # the input and output density matrices can have different
    # dimensions, although this will rarely be the case for us.
    assert np.shape(rhos_in)[0] == np.shape(rhos_out)[0]
    dim_i = rhos_in[0].size
    dim_o = rhos_out[0].size
    n_state = len(rhos_in)

    rhos_i = np.asarray(rhos_in).reshape((n_state, dim_i))
    rhos_o = np.asarray(rhos_out).reshape((n_state, dim_o))
    chi_pointer, resids, rank, s = np.linalg.lstsq(rhos_i, rhos_o, rcond=-1)

    if return_all:
        return chi_pointer, resids, rank, s
    else:
        return chi_pointer


_QPT_TRANSFORMS = {}


def qpt_basis_mat(As, key=None):
    """Vectorized unitary operator basis {As}

    Args:
        As: <list> of unitary operator based on which to compute
            the chi matrix for process tomography. These operators
            should form a 'complete' set to allow the full chi matrix
            to be represented, though this is not enforced.
            e.g. For 2-qubits, a complete set of As has 16 elements,
            each is a 4x4 unitary matrix.

        key: <string> as the key of _QPT_TRANSFORMS dict under which
            this tomography protocol will be stored so it can be
            referred to without recomputing the transformation matrix

    Returns:
        <array> of a basis matrix that should be passed to qpt() along
        with input and output density matrices to perform the process
        tomography
    """

    As = np.asarray(As, dtype=CPLX_DTYPE)

    dim_o, dim_i = As[0].shape
    size_chi = dim_o * dim_i  # 1 qubit = 4; 2 qubit = 16

    # Note: this is different from chi_basis_mat(As)
    # TODO: unify qpt basis in this file and that in qmath
    T = np.array([
        mat2vec(super2mat(A_j,
                          A_k.conj().T, left_dot=False)).flatten()
        for A_j in As for A_k in As
    ]).T
    # transpose here is for right dot with ρ, where jk is the last index

    if key is not None:
        _QPT_TRANSFORMS[key] = (As, T)

    return T


def chi_mat(chi_pointer, T, return_all=False):
    """
    Calculate the chi-matrix from the chi pointer matrix, given the
    chosen basis {A} specified by the unitary basis array As.
    """
    if T in _QPT_TRANSFORMS:
        T = _QPT_TRANSFORMS[T][1]

    _, dim_o = chi_pointer.shape
    chi_vec, resids, rank, s = np.linalg.lstsq(T, chi_pointer.flatten(),
                                               rcond=-1)
    chi = vec2mat(chi_vec)

    if return_all:
        return chi, resids, rank, s
    else:
        return chi


def qpt(rhos_in, rhos_out, T=None, return_all=False):
    """Calculate the χ matrix of a quantum process

    Args:
        rhos_in: <array> of input density matrices
        rhos_out: <array> of output density matrices

        T: <list of matrix> or <string of the basis>
            transformation χ-matrix in the "pointer-basis" so that
            vec(ρ_out) = T @ vec(ρ_in)
    """
    rhos_in = np.array(rhos_in)
    rhos_out = np.array(rhos_out)
    T = default_basis(rhos_in[0]) if T is None else T
    chi_pointer = chi_pointer_mat(rhos_in, rhos_out)
    return chi_mat(chi_pointer, T, return_all)


def gen_ideal_chi_matrix(U, As, rho0=None, zero_th=0, qpt_basis=None,
                         return_states=False, noisy=True):
    """Generate the ideal χ matrix from the input unitary operator U

    Args:
        U: the unitary operator to be analyized
        As: operators to prepare the input states
        rho0: the initial state of the qubits system before preparation
        qpt_basis: (name of) the qpt basis

    Returns:
        χ matrix of the U operator
    """

    U = qops.get_op(U) if isinstance(U, str) else U
    rho0 = np.diag([1] + (len(U) - 1) * [0]) if rho0 is None else rho0
    U_dim = len(U)

    S_dim = len(qops.get_op(As[0]))
    E_dim = U_dim // S_dim

    rhos_i, rhos_o = [], []

    for A_op in As:
        A = qops.get_op(A_op) if isinstance(A_op, (str, tuple)) else A_op
        if S_dim == U_dim:
            in_rho = qmath.dot3(A, rho0, A.conj().T)
            out_rho = qmath.dot3(U, in_rho, U.conj().T)
        else:
            A_se = qmath.tensor((A, qmath.sigma_I(E_dim)))
            # entanglement break
            rho_s_e = A_se @ rho0 @ A_se.conjugate().transpose()
            rho_shape = 2 * [S_dim, E_dim]
            in_rho = np.reshape(rho_s_e, rho_shape).trace(axis1=1, axis2=3)
            if (abs(in_rho) < zero_th).all() and noisy:
                print('tr(ρA)=tr(ρ{}) is small for QPT, discard!'.format(A_op))
                continue
            rho_s_e = U @ rho_s_e @ U.conjugate().transpose()
            out_rho = np.reshape(rho_s_e, rho_shape).trace(axis1=1, axis2=3)
        rhos_i.append(in_rho)
        rhos_o.append(out_rho)

    chi = qpt(rhos_i, rhos_o, qpt_basis)
    if return_states:
        return chi, rhos_i, rhos_o
    else:
        return chi


def cal_process_rho(rho, chi, qpt_basis=None):
    qpt_basis = default_basis(rho) if qpt_basis is None else qpt_basis
    As = _QPT_TRANSFORMS[qpt_basis][0]
    return sum(chi[j, k] * dot3(As[j], rho, As[k].conj().T)
               for j in range(len(As)) for k in range(len(As)))


def cal_process_rhose(rhose, chi, qpt_basis=None):
    """evolute S and E assuming the Identity map on E"""
    qpt_basis = default_basis(rhose) if qpt_basis is None else qpt_basis
    As = _QPT_TRANSFORMS[qpt_basis][0]
    return sum(chi[j, k] * dot3(tensor([As[j], sigma_i]), rhose,
                                tensor([As[k], sigma_i]).conj().T)
               for j in range(len(As)) for k in range(len(As)))


######################
# qubit basis matrix #
######################

# 2-level protocols
pauli_vector_ops = ['I', 'X', 'Y', 'Z']
pauli_vector_basis = [sigma_i, sigma_x, sigma_y, sigma_z]
raise_lower_basis = [sigma_i, sigma_P(2), sigma_M(2), sigma_z]

qpt_basis_mat(pauli_vector_basis, 'pauli-vector')
qpt_basis_mat(raise_lower_basis, 'raise-lower')

# 3-level protocols

sigma3_i02 = expander(sigma_i, dim=3, index=1)
sigma3_x02 = expander(sigma_x, dim=3, index=1)
sigma3_y02 = expander(sigma_y, dim=3, index=1)
sigma3_z02 = expander(sigma_z, dim=3, index=1)

sigma3_x01 = expander(sigma_x, dim=3, index=2)
sigma3_y01 = expander(sigma_y, dim=3, index=2)

sigma3_x12 = expander(sigma_x, dim=3, index=0)
sigma3_y12 = expander(sigma_y, dim=3, index=0)

sigma3_i11 = expander(expander(np.array([[1]]), dim=2, index=0), dim=3)

pauli3_vector_basis = [
    sigma3_i02,
    sigma3_x02,
    sigma3_y02,
    sigma3_z02,
    sigma3_x01,
    sigma3_y01,
    sigma3_x12,
    sigma3_y12,
    sigma3_i11,
]

# TODO: refactor and test
gellmann_basis = [
    # I3
    np.eye(3, dtype=CPLX_DTYPE),
    # sigma3_x01
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=CPLX_DTYPE),
    # sigma3_y01
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=CPLX_DTYPE),
    # sigma3_z01
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=CPLX_DTYPE),
    # sigma3_x02
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=CPLX_DTYPE),
    # sigma3_y01
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=CPLX_DTYPE),
    # sigma3_x12
    np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=CPLX_DTYPE),
    # sigma3_y12
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=CPLX_DTYPE),
    # sigma3_gm8th
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=CPLX_DTYPE) / np.sqrt(3)
]

qpt_basis_mat(pauli3_vector_basis, key='pauli3-vector')
qpt_basis_mat(gellmann_basis, key='gellmann')

###########################
# multiqubit basis matrix #
###########################

qpt_basis_mat(tensor_combinations(pauli_vector_basis, 2),
              'pauli-vector_2qubits')
qpt_basis_mat(tensor_combinations(raise_lower_basis, 2), 'raise-lower_2qubits')

# qpt_basis_mat(tensor_combinations(pauli_vector_basis, 3),
#               'pauli-vector_3qubits')
# qpt_basis_mat(tensor_combinations(raise_lower_basis, 3), 'raise-lower_3qubits')


def test_qpt(n=1):
    """
    Generate a random chi matrix, and check that we recover it from
    process tomography
    """
    def test_qpt_protocol(proto):
        As = _QPT_TRANSFORMS[proto][0]
        num_qubits = int(np.log2(As.shape[1]))
        N = len(As)

        chi_orig = (np.random.uniform(-1, 1, (N, N)) +
                    1j * np.random.uniform(-1, 1, (N, N)))

        # create input density matrices from a bunch of rotations
        qops = [
            sigma_i,
            rotate(sigma_x, np.pi / 2),
            rotate(sigma_y, np.pi / 2),
            rotate(sigma_x, -np.pi / 2),
        ]
        As = tensor_combinations(qops, num_qubits)

        rho0 = np.diag([1] + (2**num_qubits - 1) * [0])
        rhos_in = [dot3(u, rho0, u.conj().T) for u in As]

        # apply operation to all inputs
        rhos_out = [cal_process_rho(rho, chi_orig, proto) for rho in rhos_in]

        # calculate chi matrix and compare to the origin
        chi_calc = qpt(rhos_in, rhos_out, proto)
        return np.max(np.abs(chi_orig - chi_calc))

    # 1 qubit
    errs = [test_qpt_protocol('pauli-vector') for _ in range(n)]
    print('1-qubit pauli-vector max error: {:.5g}'.format(max(errs)))

    errs = [test_qpt_protocol('raise-lower') for _ in range(n)]
    print('1-qubit raise-lower max error: {:.5g}'.format(max(errs)))

    # 2 qubits
    errs = [test_qpt_protocol('pauli-vector_2qubits') for _ in range(n)]
    print('2-qubit pauli-vector max error: {:.5g}'.format(max(errs)))

    errs = [test_qpt_protocol('raise-lower_2qubits') for _ in range(n)]
    print('2-qubit raise-lower max error: {:.5g}'.format(max(errs)))

    # 3 qubits
    ## This will take a lot of time!!!
    # from datetime import datetime

    # start = datetime.now()
    # errs = [test_qpt_protocol('pauli-vector_3qubits') for _ in range(n)]
    # print('3-qubit pauli-vector max error: {:.5g}'.format(max(errs)))
    # print('elapsed:', datetime.now() - start)

    # start = datetime.now()
    # errs = [test_qpt_protocol('raise-lower_3qubits') for _ in range(n)]
    # print('3-qubit raise-lower max error: {:.5g}'.format(max(errs)))
    # print('elapsed:', datetime.now() - start)


if __name__ == '__main__':
    test_qpt()
