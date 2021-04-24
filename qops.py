# -*- coding: utf-8 -*-

import numpy as np

# References
# [1] Kuah, A. M., Modi,... & Sudarshan, E. C. G. (2007).
# How state preparation can affect a quantum experiment:
# Quantum process tomography for open systems.
# Physical Review A, 76(4), 1–12. https://doi.org/10.1103/PhysRevA.76.042113

# [2] Ordered Dict:
# In Python 3.7.0 the insertion-order preservation nature of dict objects has
# been declared to be an official part of the Python language spec.
# Therefore, you can depend on it.

import itertools

from .qdicts import STATE_AXIS_OP_DICT
from .qmath import (rotate, polar2pm, rotvec2su, expander, dot3,
                    square_matrix_dim, tensor, CPLX_TYPE)
from .qmath import sigma_i as I
from .qmath import sigma_x as X
from .qmath import sigma_y as Y
from .qmath import sigma_z as Z

XX_YY = 0.5 * (tensor((X, X)) + tensor((Y, Y)))
SQRT05 = np.sqrt(0.5)
XY = SQRT05 * (X + Y)
YX = SQRT05 * (Y - X)

UC1_DICT = {}
UCS_DICT = {}

#=======================================================================
#                           Single Qubit Tomography
#=======================================================================


def tomo_dagger(tomo_op):
    """the dagger of OP is -OP"""
    if tomo_op[0] == '-':
        return tomo_op[1:]
    else:
        return '-' + tomo_op


TOMO_DICT = {
    "I": I,
    "X/2": rotate(X, np.pi / 2),
    "Y/2": rotate(Y, np.pi / 2),
    "X": rotate(X, np.pi),
    # Mirror operators
    "-X/2": rotate(X, -np.pi / 2),
    "-Y/2": rotate(Y, -np.pi / 2),
    "-X": rotate(X, -np.pi),
    # other special rotations
    "-I": I,
    "Y": rotate(Y, np.pi),
    "X/4": rotate(X, np.pi / 4),
    "Y/4": rotate(Y, np.pi / 4),
    "X3/4": rotate(X, np.pi * 3 / 4),
    "Y3/4": rotate(Y, np.pi * 3 / 4),
    "-Y": rotate(Y, -np.pi),
    "-X/4": rotate(X, -np.pi / 4),
    "-Y/4": rotate(Y, -np.pi / 4),
    "-X3/4": rotate(X, -np.pi * 3 / 4),
    "-Y3/4": rotate(Y, -np.pi * 3 / 4),
    # rotation along the xy-plane
    "XY": rotate(XY, np.pi),
    "YX": rotate(YX, np.pi),
    "XY/2": rotate(XY, np.pi / 2),
    "YX/2": rotate(YX, np.pi / 2),
    "-XY": rotate(XY, -np.pi),
    "-YX": rotate(YX, -np.pi),
    "-XY/2": rotate(XY, -np.pi / 2),
    "-YX/2": rotate(YX, -np.pi / 2)
}
UC1_DICT.update(TOMO_DICT)

# basis of unitary control maps
UC_DICT = {
    "Ui": I,
    "Ux+": rotate(X, np.pi / 2),
    "Uy+": rotate(Y, np.pi / 2),
    "Uz+": rotate(Z, np.pi / 2),
    "Ux-": rotate(X, -np.pi / 2),
    "Uy-": rotate(Y, -np.pi / 2),
    "Uz-": rotate(Z, -np.pi / 2),
    "Uxy+": rotate(SQRT05 * (X + Y), np.pi / 2),
    "Uxz+": rotate(SQRT05 * (X + Z), np.pi / 2),
    "Uyz+": rotate(SQRT05 * (Y + Z), np.pi / 2),
    # Below are not necessary for bilinear process tomography
    "Uxy-": rotate(SQRT05 * (X + Y), -np.pi / 2),
    "Uxz-": rotate(SQRT05 * (X + Z), -np.pi / 2),
    "Uyz-": rotate(SQRT05 * (Y + Z), -np.pi / 2),
    # -------
    "Uyx+": rotate(SQRT05 * (-X + Y), np.pi / 2),
    "Uzx+": rotate(SQRT05 * (-X + Z), np.pi / 2),
    "Uzy+": rotate(SQRT05 * (-Y + Z), np.pi / 2),
    "Uyx-": rotate(SQRT05 * (-X + Y), -np.pi / 2),
    "Uzx-": rotate(SQRT05 * (-X + Z), -np.pi / 2),
    "Uzy-": rotate(SQRT05 * (-Y + Z), -np.pi / 2),
}  # [1]
UC1_DICT.update(UC_DICT)
UCS_DICT.update(UC1_DICT)

# basis of projective measurement maps
PM_DICT = {
    "Px+": 0.5 * (I + X),
    "Px-": 0.5 * (I - X),
    "Py+": 0.5 * (I + Y),
    "Pz+": 0.5 * (I + Z),
    # Below are not nessary for linear process tomography
    "Py-": 0.5 * (I - Y),
    "Pz-": 0.5 * (I - Z),
    "Pxy+": 0.5 * (I + SQRT05 * (X + Y)),
    "Pxz+": 0.5 * (I + SQRT05 * (X + Z)),
    "Pyz+": 0.5 * (I + SQRT05 * (Y + Z)),
    # Below are not necessary for bilinear process tomography
    "Pxy-": 0.5 * (I - SQRT05 * (X + Y)),
    "Pxz-": 0.5 * (I - SQRT05 * (X + Z)),
    "Pyz-": 0.5 * (I - SQRT05 * (Y + Z)),
    # -------
    "Pyx+": 0.5 * (I + SQRT05 * (-X + Y)),
    "Pzx+": 0.5 * (I + SQRT05 * (-X + Z)),
    "Pzy+": 0.5 * (I + SQRT05 * (-Y + Z)),
    "Pyx-": 0.5 * (I - SQRT05 * (-X + Y)),
    "Pzx-": 0.5 * (I - SQRT05 * (-X + Z)),
    "Pzy-": 0.5 * (I - SQRT05 * (-Y + Z)),
}  # [1]

## verify the correctness:
# for key in PM_DICT.keys():
#     u1_op = STATE_AXIS_OP_DICT[key[1:]]
#     u0_op = tomo_dagger(u1_op)
#     P = get_op(u1_op) @ np.diag([1,0]) @ get_op(u0_op)
#     print(P)
#     assert np.allclose(P, get_op(key))

# casual break
# ptf.ref[4] III
# ptf.ref[5] V.A
# Chose the full basis: {Pz+, Px+, Px-, Py+}
CB_FULL_BASIS = ["Pz+", "Px+", "Px-", "Py+"]
PM_OCTOMO_BASIS = ["Pz+", "Pz-", "Px+", "Px-", "Py+", "Py-"]
PM_FULL_BASIS = list(PM_DICT.keys()[:9])
PM_SMTC_BASIS = list(PM_DICT.keys())

def cb_to_mfb_op(pms):
    """P⊗∏ => qmath.super2mat(U_p ⊙ Pz+ ⊙ U_∏)"""
    # move state from the measurement axis to z+ axis
    uc0 = tomo_dagger(STATE_AXIS_OP_DICT[pms[0][1:]])
    # move the state in z+ axis |0> to the axis of pm2
    uc2 = STATE_AXIS_OP_DICT[pms[1][1:]]
    return uc0, 'Pz+', uc2


def mfb_to_u(mfb_op):
    return dot3(TOMO_DICT[mfb_op[2]], PM_DICT[mfb_op[1]], TOMO_DICT[mfb_op[0]])


def cb_to_u(pms):
    return mfb_to_u(cb_to_mfb_op(pms))


def cb_to_lmat(pms):
    AA_c = tensor([PM_DICT[pms[1]], PM_DICT[pms[0]].transpose()])
    return AA_c.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)


# Meas & Projections
CB_FULL_OPS = [ops for ops in itertools.product(CB_FULL_BASIS, repeat=2)]
CB_SMTC_OPS = [ops for ops in itertools.product(PM_OCTOMO_BASIS, repeat=2)]
CB_FULL_DICT = dict([('.'.join(op), cb_to_u(op)) for op in CB_FULL_OPS])
CB_SMTC_DICT = dict([('.'.join(op), cb_to_u(op)) for op in CB_SMTC_OPS])
# OP_CB_LMAT_DICT = dict([('.'.join(op), cb_to_lmat(op)) for op in CB_OPS])
# MFB_OPS = [cb_to_mfb_op(op) for op in CB_OPS]
# OP_MFB_DICT = dict([('.'.join(op), mfb_to_u(op)) for op in MFB_OPS])

#=======================================================================
#                               Two Qubits
#=======================================================================

UC2_DICT = {
    "II": np.diag([1, 1, 1, 1.]),
    # Full Swap
    "Swap": np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1.],
    ]),
    "iSwap": rotate(XX_YY, -np.pi),
    "PiSwap": rotate(XX_YY, -np.pi),
    "-iSwap": rotate(XX_YY, np.pi),
    "NiSwap": rotate(XX_YY, np.pi),
    # Half Swap
    "iSwap/2": rotate(XX_YY, -np.pi / 2),
    "-iSwap/2": rotate(XX_YY, np.pi / 2),
    # Control Gates
    "CZ": np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1.],
    ]),
    "CNOT": np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0.],
    ]),
    "CNOTr": np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0.],
    ])
}
UCS_DICT.update(UC2_DICT)


def get_op(op):
    """Get A unitary operator from the op string
    ops:
        <str> or <list of str>, which is the list of ops of multiqubits
    Returns:
        <u mat of N qubits>
    """
    if np.shape(op) == ():
        # rotate along special axes, e.g. x, y, zy
        if op in UCS_DICT:
            return UCS_DICT[op]
        elif op in PM_DICT:
            return PM_DICT[op]
        elif op in CB_SMTC_DICT:
            return CB_SMTC_DICT[op]
        elif op == 'MIS':
            # contract operation <- break the connection
            return np.array([
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1.],
            ])
        else:
            raise KeyError("{} is not defined".format(op))
    elif len(op) == 3 and isinstance(op[0], (int, float)):
        # rotate along arbitrary rotvec (axis, angle)
        return rotvec2su(rotvec=op)
    elif len(op) == 2 and isinstance(op[0], (int, float)):
        # rotate along axis in the xy plane (theta, axis)
        return polar2pm(theta=op[0], phi=op[1])
    elif len(np.shape(op)) >= 1:
        return tensor([get_op(p) for p in op])
    elif op is None:
        return None
    else:
        raise KeyError


def get_ops(ops):
    """Get unitary operators from a list of op string"""
    return [get_op(op) for op in ops]


def get_init_rho(op, rho0=None):
    """Return the initial rho prepared by the op"""
    u = get_op(op)
    dim = square_matrix_dim(u)
    rho0 = np.diag([1.0] + (dim - 1) * [0]) if rho0 is None else rho0
    return dot3(u, rho0, u.conjugate().transpose())


def get_init_rhos(ops, rho0=None):
    """Return the initial rhos prepared by the ops"""
    return [get_init_rho(op, rho0) for op in ops]
