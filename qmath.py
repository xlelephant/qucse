# -*- coding: utf-8 -*-

import itertools
from functools import reduce
import numpy as np
from scipy import optimize
from scipy.linalg import expm, sqrtm, logm, det, norm
from scipy.stats import unitary_group
from scipy.spatial.transform import Rotation as R

CPLX_TYPE = np.complex128
DTYPE = CPLX_TYPE
sigma_i = np.array([[1, 0], [0, 1]], dtype=DTYPE)
sigma_x = np.array([[0, 1], [1, 0]], dtype=DTYPE)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
sigma_z = np.array([[1, 0], [0, -1]], dtype=DTYPE)
sigmas = [sigma_x, sigma_y, sigma_z]
povm_z_p = np.array([[1, 0], [0, 0]], dtype=DTYPE)
povm_z_n = np.array([[0, 0], [0, 1]], dtype=DTYPE)


def sigma_I(n):
    return np.eye(n)


def sigma_N(n):
    """number operator"""
    return np.diag(range(n))


def sigma_P(n):
    """boson raising operator, AKA sigma plus"""
    return np.diag([np.sqrt(i) for i in range(1, n)], k=-1)


def sigma_M(n):
    """boson lowering operator, AKA sigma minus"""
    return np.diag([np.sqrt(i) for i in range(1, n)], k=1)


#####################
# matrix operations #
#####################


def dot3(A, B, C):
    """Compute the dot product of three matrices"""
    return np.dot(np.dot(A, B), C)


def dots(matrices):
    """Compute the dot product of a list (or array) of matrices"""
    return reduce(np.dot, matrices)


def commutator(A, B):
    """Compute the commutator of two matrices"""
    return np.dot(A, B) - np.dot(B, A)


def lindblad(rho, L, Ld=None):
    """Compute the contribution of one Lindblad term to the master equation"""
    if Ld is None:
        Ld = L.conjugate().T
    return dot3(L, rho, Ld) - 0.5 * dot3(Ld, L, rho) - 0.5 * dot3(rho, Ld, L)


def overlap(A, B):
    return np.trace(np.dot(A, B))


def trace_norm(A):
    """The trace norm ∥ρ∥1 of a matrix ρ is the sum of the singular
    values of ρ.
    The singular values are the roots of the eigenvalues of ρρ†.
    ∥ρ∥1=Tr√(ρρ†)
    ref:
    https://www.quantiki.org/wiki/trace-norm
    """
    return np.trace(np.dot(A, A.conjugate().transpose()))


def unit_trace(A, flat_threshold=1e-5, noisy=True):
    n = square_matrix_dim(A)
    trace_A = np.trace(A)
    if abs(trace_A) < flat_threshold:
        print("Warning: tr(A)={}, set A to zeros".format(trace_A))
        # raise ValueError
        return np.zeros_like(A)
    else:
        return A / trace_A


def unimodularize(A, d=1):
    """Make the matrix unimodular with det(A) = d
    TODO: how about the complex matrix?
    See link below for the relation between unit-trace
    https://math.stackexchange.com/questions/2083410/the-relation-between-trace-and-determinant-of-a-matrix"""
    n = square_matrix_dim(A)
    return d / np.linalg.det(A)**(1.0 / n) * A


def fidelity_rho(rho_in, rho_ideal, normalize=True, imag_atol=1e-6):
    """https://www.quantiki.org/wiki/fidelity"""
    if normalize:
        rho_in = unit_trace(rho_in)
        rho_ideal = unit_trace(rho_ideal)
    sqrt_rho_ideal = sqrtm(rho_ideal)
    fid = np.trace(sqrtm(dot3(sqrt_rho_ideal, rho_in, sqrt_rho_ideal)))
    assert abs(fid.imag) < imag_atol, 'max imag = {}'.format(abs(fid.imag))
    # fid = fid / np.trace(rho_ideal)
    if fid.real > 1.0:
        print("WARNING fid {} > 1.0 and set to 1.0!".format(fid))
        fid = 1.0
    return abs(fid.real)**2


def fidelity_chi(chi_in, chi_ideal, normalize=True, imag_atol=1e-6):
    fid = np.trace(np.dot(chi_in, chi_ideal))
    if normalize:
        fid = fid / np.trace(np.dot(chi_ideal, chi_ideal.T.conj()))
    assert abs(fid.imag) < imag_atol, 'max imag = {}'.format(abs(fid.imag))
    return abs(fid)


def relative_entropy(rho_a, rho_b):
    """Returns the von Neumann relative entropy, which is asymmetric
    V. Vedral, The Role of Relative Entropy in Quantum Information Theory,
    Rev. Mod. Phys. 74, 197 (2002).
    Section II.E
    """
    assert square_matrix_dim(rho_a) == square_matrix_dim(rho_b)
    # return np.trace(np.dot(rho_a, (logm(rho_a) - logm(rho_b))))
    return (np.trace(np.dot(rho_a, logm(rho_a))) -
            np.trace(np.dot(rho_a, logm(rho_b))))


# rotation operators


def rotate(axis, angle):
    """rotate operation on a given axis
    axis can be a basis to form the su(2) group
        e.g. 𝑅𝑜𝑡(𝛼,𝑎⃗ )=cos(𝛼/2)𝟙−isin(𝛼/2)∑𝑖1:3 𝑎𝑖⋅𝜎𝑖, 𝑎⃗ =(x,y,z)/|a|
    Or a matrix from any size of space
        e.g. xx interation: rotvec=π⋅(0.5⋅(σx⊗σx+σy⊗σy)) leads to -iswap"""
    if square_matrix_dim(axis) == 2:
        return np.cos(0.5 * angle) * sigma_i - 1j * np.sin(0.5 * angle) * axis
        # $Rot{(\alpha, \vec{a})}=\cos (\frac{\alpha}{2}) \mathbb{1}-\mathrm{i} \\
        # \sin (\frac{\alpha}{2}) \sum_{i=1}^{3} a_{i} \sigma_{i}$
    else:
        return expm(-0.5j * angle * axis)


def rot_xy(theta, phi):
    axis = sigma_x * np.cos(phi) + sigma_y * np.sin(phi)
    return rotate(axis, theta)


def rotvec2su(rotvec):
    """rotate operation on a given axis
    axis can be a basis to form the su(2) group, [x, y, z] * angle
    """
    assert np.shape(rotvec) == (3, )
    angle = norm(rotvec)
    xyz = rotvec / angle
    axis = sum([v * sigma for v, sigma in zip(xyz, sigmas)])
    return rotate(axis, angle)


def su2rotvec(u):
    PRECISION = 1e-6
    if square_matrix_dim(u) == 2:
        x, y, z = [0.5j * np.trace(np.dot(u, sigma)) for sigma in sigmas]
        assert abs(x.imag) + abs(y.imag) + abs(z.imag) < PRECISION
        x, y, z = x.real, y.real, z.real
        cos_ang_2 = 0.5 * np.trace(u).real
        sin_ang_2 = np.sqrt(x**2 + y**2 + z**2)
        angle = 2 * np.arccos(cos_ang_2)
        vector = np.array([x, y, z]) / sin_ang_2
        return angle * vector
    else:
        rotvec = 2j * logm(u)
        return rotvec


def su2euler(u, degrees=True):
    return R.from_rotvec(su2rotvec(u)).as_euler('xyz', degrees=degrees)


def polar2pm(theta, phi):
    u = rot_xy(theta, phi)
    return dot3(u, povm_z_p, u.conjugate().T)


def pm2polar(p):
    """PM = u⋅Pz+⋅u†
    (a,   b ) = (1, 0) = (a*, -b) == (|a|^2, -ab  )
    (-b*, a*)   (0, 0)   (b*, a )    (-a*b*, |b|^2)
    return: theta, phi
    """
    PRECISION = 1e-6
    assert (p[0, 0] + p[1, 1] - 1).real < PRECISION
    assert abs(p[0, 0].imag) + abs(p[1, 1].imag) < PRECISION
    assert abs(p[1, 0] - p[0, 1].conjugate()) < PRECISION
    a = np.sqrt(p[0, 0].real)  # no z-axis rotation
    b_conj = (-p[0, 1] / a).conjugate()
    return 2 * np.arccos(a), np.angle(-1j * b_conj)


# random operators


def random_theta_phi():
    rand_2 = np.random.rand(2)
    return rand_2[0] * np.pi, rand_2[1] * 2 * np.pi


def random_su(dim):
    """SU - special unitary group: unitary and unimodular (TODO:det(A)=+?)"""
    u = unitary_group.rvs(dim)
    return unimodularize(u)


def random_pm():
    return polar2pm(*random_theta_phi())


# tensor operators


def matrixize(T):
    """convert any tensor to the matrix form"""
    dim = int(np.size(T)**0.5)
    return np.reshape(T, (dim, dim))


def tensor(matrices):
    """Compute the tensor product of a list (or array) of matrices"""
    return reduce(np.kron, matrices)


def tensor_combinations(matrices, repeat):
    """Compute a list of tensor products for the iteration of matrices"""
    return [tensor(ms) for ms in itertools.product(matrices, repeat=repeat)]


def tensor_combinations_phases(matrices, repeat, phases):
    """Compute a list of tensor products for the iteration of matrices,
    with additional rotation along the z-axis
    """
    products = itertools.product(matrices, repeat=repeat)
    tensor_products = []
    for mats in products:
        tensor_products.append(
            tensor([
                dot3(rotate(sigma_z, -phases[i]), mat,
                     rotate(sigma_z, phases[i])) for i, mat in enumerate(mats)
            ]))
    return tensor_products


##############################
# Operator <-> Superoperator #
##############################

# we use c-style order as default, equals to 'ROW' stacking
# alternative is Fortran like, equals to 'COL' in 2 dim matrix
# 'C' => 'ROW'
# 'F' => 'COL'
VEC_ORDER = 'ROW'
VEC_DIR_DICT = {'ROW': 'C', 'COL': 'F'}
RESHAPE_ORDER = VEC_DIR_DICT[VEC_ORDER]


def square_matrix_dim(M):
    """Check if the input is square Matrix and return its first dimension"""
    assert M.shape[0] == M.shape[1] and len(M.shape) == 2
    return M.shape[0]


def mat2vec(M, order=RESHAPE_ORDER):
    """flatten a matrix to vector, 1st row 1st by default
    input: numpy array
    order: default 'C' represents in row form and 'R' in column form
    output: vectorized array, row=n, column=1
    """
    # np.reshape used here is more general than ndarray.flatten
    # by default in row order
    return M.reshape((-1, 1), order=order)


def vec2mat(V, order=RESHAPE_ORDER, shape=None):
    """reshape a vector to a square matrix, whos number of rows equals cols

    input: any iterable object with size of N**2
    """
    if shape is None:
        dim = np.sqrt(len(V))
        assert dim.is_integer()
        shape = (int(dim), int(dim))
    return np.reshape(V, shape, order=order)


def super2mat(A, C=None, order=VEC_ORDER, left_dot=True):
    """convert the super operator to a left-multiplying matrix operator
    by vectorizing its state space

    C = A† if C is None

    We use the Roth's lemma to make this conversion:
    vec(ABC) = (A⊗C.T)vec(B), for row stacking
    vec(ABC) = (C.T⊗A)vec(B), for column stacking
    where ⊗ is the tensor product operator

    Inspired by Rigetti Computing's excellent Forest-Benchmarking document:
    https://forest-benchmarking.readthedocs.io/en/latest/superoperator_representations.html
    Note that Forest-Benchmarking uses column stacking by default
    Another rule worth mentioning is: (A⊗B)(C⊗D)=AC⊗BD.
    """
    C = A.T.conjugate() if C is None else C
    if order == 'ROW':
        return np.kron(A, C.T) if left_dot else np.kron(A.T, C)
    elif order == 'COL':
        return np.kron(C.T, A) if left_dot else np.kron(C, A.T)
    else:
        raise NotImplementedError()


def liouville_H(H):
    """Convert Hamiltonian in Hilbert space to liouville space
    The origin H is on a 2 dim basis, then the results should be 4 dim

    The function seems not relative to the order of vectorization, which
    is interesting
    """
    dim = square_matrix_dim(H)
    I = np.eye(dim, dtype=DTYPE)
    return -1j * (np.kron(H, I) - np.kron(I, H.T))


def liouville_decoherence_channel(A):
    """returns the liouville matrix representation of decoherence
    channels, where the system-environment coupling term is A
    """
    dim = square_matrix_dim(A)
    Adag_A = np.dot(A.conjugate().T, A)
    return super2mat(
        A,
        A.conjugate().T) - 0.5 * (super2mat(sigma_I(dim), Adag_A) +
                                  super2mat(Adag_A, sigma_I(dim)))


def trace_mat(M, idx, dims=None):
    """It reduces matrix of the indexed subsystem by tracing over all
    other subsystems.

    Args:
        M: matrix of the global system.
            if dims is None, M should be pre-shaped
            e.g. For a system containing a 2-level qubit, a 3-level
            qutrit and 4-level resonator the shape of M should be
            (2,3,4, 2,3,4)
            if dims is given, M is reshaped to dims
            e.g. for a system consist of A(m1,n1) and B(m2,n2),
            the dims should be the form [(m1,n1),(m2,n2)]
            or (m1,m2,...,n1,n2,...)
        idx: <int> or <list of int> denotes the index to keep
    Returns:
        the reduced M of the subsystem under indexation
        Note: The result is different for idx=(1,2) and (2,1)
    """

    if dims is None:
        num_sub_sys = len(np.array(M).shape) / 2
        assert num_sub_sys.is_integer()
        num_sub_sys = int(num_sub_sys)
    elif len(np.shape(dims)) == 1:
        M = M.reshape(dims)
        num_sub_sys = int(len(dims) // 2)
    elif len(np.shape(dims)) == 2:
        num_sub_sys = len(dims)
        dim_row = [dims[i][0] for i in range(num_sub_sys)]
        dim_col = [dims[i][1] for i in range(num_sub_sys)]
        M = M.reshape(dim_row + dim_col)

    idx_keep = np.array(idx).flatten()
    idx_trace = [d for d in range(num_sub_sys) if d not in idx_keep]

    for i in reversed(idx_trace):
        M = np.trace(M, axis1=i, axis2=num_sub_sys + i)
        num_sub_sys -= 1

    if not all(idx_keep[i] <= idx_keep[i + 1]
               for i in range(len(idx_keep) - 1)):
        sort_idx = np.argsort(idx_keep)
        trans_idx = [i for i in sort_idx] + \
                    [len(idx_keep) + i for i in sort_idx]
        M = np.transpose(M, axes=trans_idx)

    return M


#######################################################################
# conversion between ket, mat(rho)-vec(rhos)-diag(rho), bloch & polar #
#######################################################################

# ket  |0>=(1, 0), |1>=(0, 1), etc...


def ket2rho(ket, normalize=True):
    """Convert a state (ket) to a density matrix (rho)."""
    rho = np.outer(ket, np.conjugate(ket))
    if normalize:
        return unit_trace(rho)
    else:
        return rho


def ket2polar(ket):
    ket_norm = np.array(ket, dtype=DTYPE) / np.linalg.norm(ket)
    theta = 2 * np.arccos(abs(ket[0]))
    phi = np.angle(ket[1] / ket[0])
    return theta, phi, 1.0


# rho projection and expansion


def projector(rho, state=[0, 1]):
    """project the system density matrix to qubit subspace"""
    state = list([state]) if not np.iterable(state) else list(state)
    N = square_matrix_dim(rho)
    M = len(state)
    P = np.zeros((M, N))
    for i, s in enumerate(state):
        P[i, s] = 1.0
    rho = dot3(P, rho, P.T)
    return rho.squeeze()


def expander(rho, dim=3, index=2):
    """expand the density matrix to shape of (dim, dim)
    It inserts 0s to third row and column by default.
    """
    N = square_matrix_dim(rho)
    for _ in range(dim - N):
        rho = np.insert(rho, index, 0, axis=0)
        rho = np.insert(rho, index, 0, axis=1)
    return rho


def disentanglement(rho, separated=False, dim=None):
    dims = np.shape(rho) if dim is None else list(dim) + list(dim)
    indices = range(np.size(dims) // 2)
    rhos = [trace_mat(np.array(rho), i, dims=dims) for i in indices]
    if separated:
        return rhos
    else:
        return tensor(rhos)


# rho ([[𝜌11, 𝜌12], [𝜌11, 𝜌12]])


def vecs_to_rhos(vecs, d=None, m=None, real_vec=True):
    """Convert vecs to rhos
    vecs.shape = (d^2, m), where
    m is the number of snapshots
    l is the num of elements of X if real_vec=False
    2l is the num of elements of X if real_vec=True
    """
    if d is None and m is None:
        l, m = np.shape(vecs)
        d = int(np.sqrt(l))
    else:
        l = d**2
    if real_vec:
        vecs_cplx = vecs[:l] + 1j * vecs[l:]
        return np.transpose(vecs_cplx).reshape(m, d, d)
    else:
        return np.transpose(vecs).reshape(m, d, d)


def rhos_to_vecs(rhos, l=None, m=None, real_vec=True):
    """Convert rhos to vecs
    vecs.shape = (l, m), where
    m is the number of snapshots
    l is the num of elements of X if real_vec=False
    2l is the num of elements of X if real_vec=True
    """
    if l is None and m is None:
        m, dr, dc = np.shape(rhos)
        l = dr * dc
    if real_vec:
        return np.concatenate(
            (np.real(rhos).reshape(m, l).T, np.imag(rhos).reshape(m, l).T))
    else:
        return np.reshape(rhos, (m, l)).T


def rho2bloch(rho):
    # Note: all rhos in shoud be 2x2
    return np.array([np.trace(np.dot(rho, sigma)) for sigma in sigmas])


def rhos2bloch(rhos):
    """returns xs, ys, zs"""
    xyzs = np.array([rho2bloch(rho) for rho in rhos])
    return xyzs.T


def rho2polar(rho):
    # the imaginary parts are typically of magnitude of ~1e-16
    return bloch2polar(rho2bloch(rho).real)


def rhos2polar(rhos):
    """returns thetas, phis, mags"""
    return blochs2polar(rhos2bloch(rhos).real)


# bloch (x, y, z)


def bloch2rho(xyz):
    """ρ = 0.5 * (I + x⋅σ_x + y⋅σ_y + z⋅σ_z)"""
    # Note: all rhos in shoud be 2x2
    return 0.5 * np.sum(
        [sigma_i] + [k * sigma for k, sigma in zip(xyz, sigmas)], axis=0)


def blochs2rho(xyzs):
    """input shape is xs, ys, zs"""
    m = len(xyzs[0])
    return 0.5 * np.sum(
        [np.kron(sigma_i, np.ones(m)).reshape(2, 2, m).transpose(2, 0, 1)] + [
            np.kron(sigma, k).reshape(2, 2, m).transpose(2, 0, 1)
            for k, sigma in zip(xyzs, sigmas)
        ],
        axis=0)


def bloch2polar(xyz):
    x, y, z = xyz
    mag = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / mag)
    phi = np.angle(x + 1j * y)
    return theta, phi, mag


def blochs2polar(xyzs):
    """input xs, ys, zs"""
    return bloch2polar(xyzs)


# polar (𝜃, 𝜙, r)


def polar2ket(theta, phi):
    # mag = 1.0
    return np.array(
        [np.cos(0.5 * theta),
         np.sin(0.5 * theta) * np.exp(1j * phi)],
        dtype=DTYPE)


def polar2rho(theta, phi, mag=1.0):
    return bloch2rho(polar2bloch(theta, phi, mag))


def polars2rho(polars):
    return blochs2rho(polars2bloch(polars))


def polar2bloch(theta, phi, mag=1.0):
    x = mag * np.sin(theta) * np.cos(phi)
    y = mag * np.sin(theta) * np.sin(phi)
    z = mag * np.cos(theta)
    return x, y, z


def polars2bloch(polars):
    """input thetas, phis, mags"""
    return polar2bloch(*polars)


###################
# Other Utilities #
###################


def safe_log(x):
    """Safe version of log that returns -Inf when x < 0, rather than NaN"""
    return np.log(np.maximum(x.real, 1e-100))
