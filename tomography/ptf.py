# -*- coding: utf-8 -*-

# Tomography of the Process Tensor:
# "After all, this is the objective of tomography—by performing
# measurements on a small number of select input states, a complete
# description of a process can be obtained and predict the output state
# for any arbitrary input state." [1]

# The map Λ is reconstructed experimentally by determining the output states:
#                               ρ′=Λ[ρ]

# In Choi representation, a.k.a Sudarshan B form, the map Λ acting on ρ is
# written as:
#                       Λ̂ [ρ]=tr_in[(𝟙_out⊗ρT)Λ]

# One-step process tensor is also called superchannels.

# Generalization of Λ to multiple steps:
#       𝑇̂ 𝑁:0[𝐴̂ 𝜇0,…,𝐴̂ 𝜇𝑁−1]=𝑝(𝐴̂ 𝜇0,…,𝐴̂ 𝜇𝑁−1)⋅ρ(𝐴̂ 𝜇0,…,𝐴̂ 𝜇𝑁−1)

# In the process tensor formalism, it is the action of process tensor 𝑇̂ {𝑁:0}
# on a general sequence 𝐴_{𝑁−1:0}:
#       𝑇̂ 𝑁:0[𝐴̂ 𝑁−1:0]=tr𝑖𝑛[(𝟙_out⊗𝐴̂ T𝑁−1:0)𝑇^{𝑁:0}]:=ρ′(𝐴𝑁−1:0)
#
# where {𝐴𝜇⃗} := {𝐴_𝑁−1:0} forms a basis of the space of N sequences of CP
# operations:
#               𝐴𝜇⃗ =⊗_𝑘:0~(𝑁−1) 𝐴_𝜇𝑘 at times 𝑡_0,…,𝑡_{𝑁−1}

import numpy as np
import scipy
import itertools

from . import qst, qpt, psd
from .. import qmath, qops

global tmp_i
tmp_i = 0

ZERO_RHO_TH = 1E-6

QST_BASIS = 'tomo'
QST_PMs = ['Pz+', 'Pz-', 'Py+', 'Py-', 'Px-', 'Px+']
# should be more than enough for QST!
# QST_BASIS = 'smtc_tomo'
# QST_PMs = [
#     'Pz+', 'Pz-', 'Py+', 'Py-', 'Px-', 'Px+', 'Pyx+', 'Pyx-', 'Pxy-', 'Pxy+',
#     'Pxz+', 'Pxz-', 'Pyz+', 'Pyz-', 'Pzx+', 'Pzx-', 'Pzy+', 'Pzy-'
# ]

QPT_PMs = qst.TOMO_BASIS_OPS['pm_octomo']


def basis_ops(group='uc', steps=1, complement=False):
    if group == 'uc':
        op_set = list(qops.UC_DICT.keys())
        fit_set = op_set[0:10]
    elif group == 'pm':
        op_set = list(qops.PM_DICT.keys())
        fit_set = op_set[0:9]
    elif group == 'cb':
        op_set = list(qops.CB_SMTC_DICT.keys())
        fit_set = list(qops.CB_FULL_DICT.keys())
    else:
        raise KeyError('invalid group {}'.format(group))

    if complement is False:
        return [n for n in itertools.product(fit_set, repeat=steps)]
    elif complement is True:
        ops_fit = [n for n in itertools.product(fit_set, repeat=steps)]
        ops_cpl = [n for n in itertools.product(op_set, repeat=steps)]
        all_fit = 0
        for op in ops_fit:
            ops_cpl.remove(op)
            all_fit += 1
            print('-', end='')
        print(all_fit, group + 's were removed.')
        return ops_cpl
    elif complement is None:
        return [n for n in itertools.product(op_set, repeat=steps)]


def pretty_rotvec_op(ops, join_str='*',
                     format_str='(Rx:{:.1f}Ry:{:.1f}Rz:{:.1f})'):
    """ops: <list of Rx, Ry, Rx>"""
    return join_str.join(
        [op if isinstance(op, str) else format_str.format(*op) for op in ops])


def pretty_povm_ops(ops, join_str='*', format_str='(r:{:.2f}p:{:.2f})'):
    """ops: <list of theta, phi>"""
    return join_str.join(
        [op if isinstance(op, str) else format_str.format(*op) for op in ops])


def ops2str(ops):
    return [
        pretty_rotvec_op(op, '\n', '({:.1f}:{:.1f}:{:.1f})') if len(op) == 3
        else pretty_povm_ops(op, '\n', r'$\theta$:{:.2f}$\phi$:{:.2f}')
        for op in ops
    ]


class ProcessTensor(object):
    """full process tensor"""
    def __init__(self, T_choi=None, T_mat=None):
        self.D_S = 2  # dimension of system
        self.D_E = 2  # dimension of environment
        self.T_choi = T_choi
        self.T_mat = T_mat
        self.basis = 'cb'
        if self.T_choi is not None:
            step_els = self.steps_of_tensor(T_choi)
            self.N = step_els
        elif self.T_mat is not None:
            step_els = self.steps_of_tensor(T_mat)
            self.N = step_els
        else:
            self.N = None

    @staticmethod
    def choi_order(step_els, reverse=False):
        """The transpose order of Matrix-formed (A-form) tensor
        to its Choi state (B-form) representation.
        Matrix-form is of computational convenience,
        while the Choi representation is more operational meaningful.
        According to the index definition in ref[4]:

        # A form:
        ρ_se(r0 ϵ0:s0 γ0)
        A(r0's0':r0 s0)
        u(r1 ϵ1 :r0'ϵ0)
        ---
        U(r1 ϵ1 ;s1 γ1 :r0'ϵ0 s0'γ0)
        A(r1's1' r0's0': r1 s1 r0 s0: 𝟙 𝟙)
        M(r1's1' r0's0' ;r1 s1 r0 s0 :r2 s2)

        # B form
        A_choi(𝟙  r1'r1 r0'r0 :𝟙  s1's1 s0's0)
        T_choi(r2 r1'r1 r0'r0 :s2 s1's1 s0's0)

        ⟨As⟩⋅⎹ρ⟩⟩: (r1's1' r0's0'× r1 s1 r0 s0)(r1 s1 r0 s0)
        <==>
        <As_prod>⋅I⊗I <----> tr(⎹00⟩+⎹11⟩×⟨00⎸+⟨11⎸ <As_choi>)

        e.g. Num of steps = 2:
        A⋅M = ρ_s <As_prod>.transpose(0,4,2,6, 1,5,3,7)
        """
        if not reverse:
            # rk
            T_order = [[4 * step_els]]
            # ri',ri
            T_order += [[2 * i, 2 * (step_els + i)] for i in range(step_els)]
            # sk
            T_order += [[4 * step_els + 1]]
            # si', si
            T_order += [[2 * i + 1, 2 * (step_els + i) + 1]
                        for i in range(step_els)]
        else:
            # ri',si'
            T_order = [[1 + 2 * i, 2 + 2 * i + 2 * step_els]
                       for i in range(step_els)]
            # ri ,si
            T_order += [[2 + 2 * i, 3 + 2 * i + 2 * step_els]
                        for i in range(step_els)]
            # rk ,sk
            T_order += [[0, 1 + 2 * step_els]]
        return sum(T_order, [])

    def steps_of_tensor(self, T):
        """calculate the step of the 1-qubit (n=D_S) process tensor"""
        D_S, D_E = self.D_S, self.D_E
        return int(np.log2(np.size(T) - D_S * D_S) / np.log2(D_S**4))

    def trace_env(self, rho):
        return rho.reshape(self.D_S, self.D_E, self.D_S,
                           self.D_E).trace(axis1=1, axis2=3)

    def A_to_AA(self, A):
        """convert local operation (Bilinear) to LeftMatrix form"""
        if not isinstance(A, np.ndarray):
            A = qops.get_op(A)
        elif np.size(A) != self.D_S**2:
            A = qops.get_op(A)
        return qmath.super2mat(A)

    def As_to_choi(self, As):
        D_S, _ = self.D_S, self.D_E
        step_els = len(As)
        AsAs = 1
        for A in As:
            AA = self.A_to_AA(A)
            AsAs = qmath.tensor((AA, AsAs))
        As_vec = AsAs.reshape(-1)
        As_out_vec = np.einsum(As_vec, [0], np.eye(D_S) / D_S, [1, 2])
        A_shape = 2 * step_els * [D_S, D_S] + [D_S, D_S]
        A_tensor = np.reshape(As_out_vec, A_shape)
        A_choi = np.transpose(A_tensor, self.choi_order(step_els))
        return A_choi

    def Us_to_choi(self, Us):
        D_S, D_E = self.D_S, self.D_E
        dim_U_choi = (D_S * D_E)**2
        Us_choi = 1
        for op in Us:
            u = qops.get_op(op) if isinstance(op, str) else op
            U = qmath.super2mat(u)
            U_tensor = np.reshape(U, 4 * [D_S, D_E])
            U_choi = np.transpose(U_tensor, [0, 1, 4, 5] + [2, 3, 6, 7])
            U_choi = np.reshape(U_choi, (dim_U_choi, dim_U_choi))
            Us_choi = qmath.tensor((U_choi, Us_choi))
        return Us_choi

    # Derive the process tensor

    def cal(self, rho_se, Us, return_format='choi'):
        """The U of us has the index:
        super2mat(u, u†) = U(r1 ϵ1,s1 γ1:r0'ϵ0,s0'γ0)
        => U_Choi(r1 ϵ1 r0'ϵ0 :s1 γ1 s0'γ0)
        => Us_Choi(r2 ϵ2 r1'ϵ1 ,r1 ϵ1 r0'ϵ0 : s2 γ2 s1'γ1, s1 γ1 s0'γ0)
        T_se_tensor = Us_Choi_k ⊗... Us_Choi_0 ⊗ rho_se
        => (r2 ϵ2 r1'ϵ1; r1 ϵ1 r0'ϵ0; r0 ϵ0 :: s2 γ2 s1'γ1; s1 γ1 s0'γ0; s0 γ0)
        =>     |     |______|     |______|        |     |______|     |______|
               |__________________________________|
        => tr_se(T) = ∑_ϵγ ρ_se ⋅ (Π_k-1:0 U_k) ⋅ 𝛿ϵγ
        =>T_Choi(r2 r1'r1 r0'r0 :: s2 s1's1 s0's0)
        """
        self.N = len(Us) if self.N is None else self.N
        D_S, D_E = self.D_S, self.D_E
        dim_U_choi = (D_S * D_E)**2
        Us_choi = 1
        for op in Us:
            u = qops.get_op(op) if isinstance(op, str) else op
            U = qmath.super2mat(u)
            U_tensor = np.reshape(U, 4 * [D_S, D_E])
            U_choi = np.transpose(U_tensor, [0, 1, 4, 5] + [2, 3, 6, 7])
            U_choi = np.reshape(U_choi, (dim_U_choi, dim_U_choi))
            Us_choi = qmath.tensor((U_choi, Us_choi))
        T_se_shape = 2 * (2 * self.N + 1) * [D_S, D_E]
        T_se_tensor = np.reshape(qmath.tensor((Us_choi, rho_se)), T_se_shape)
        tensor_order = len(T_se_shape)
        trace_eg_idx = np.arange(tensor_order)
        for i in range(self.N):
            implicit_e_idx = -(4 * i + 2 + 1 + tensor_order // 2)
            trace_eg_idx[implicit_e_idx + 2] = trace_eg_idx[implicit_e_idx]
            implicit_g_idx = -(4 * i + 2 + 1)
            trace_eg_idx[implicit_g_idx + 2] = trace_eg_idx[implicit_g_idx]
        explicit_e_idx = 1
        explicit_g_idx = tensor_order // 2 + 1
        trace_eg_idx[explicit_g_idx] = trace_eg_idx[explicit_e_idx]
        T_choi = np.einsum(T_se_tensor, trace_eg_idx)
        if return_format == 'choi':
            return T_choi
        elif return_format == 'matrix':
            T_mat = self.choi_to_matrix(self.T_choi)
            return T_mat
        else:
            raise KeyError

    def least_square_fit(self, Bss, rhos, disp=False):
        """Fit the Matrix formed process tensor
        e.g. for Bss.shape==(100,4^N)
        A(100,16^N) ⋅ M(16^n,4) = ρ(100,4)
        Args:
            Bss: a list of N-step operators
            rhos: the output density matrix
        """
        self.N = len(Bss[0]) if self.N is None else self.N
        D_S = int(np.size(Bss[0][0])**0.5)
        rho_vec = np.reshape(rhos, (len(rhos), -1))
        As_vec = []
        for As in Bss:
            As = qops.get_ops(As)
            AsAs = qmath.tensor([qmath.super2mat(A) for A in As[::-1]])
            As_vec.append(AsAs.reshape(-1))
        M, resids, rank, s = np.linalg.lstsq(As_vec, rho_vec, rcond=None)
        if disp:
            print('\nresids is: ', resids, '\nrank is: ', rank, '\ns is: ', s)
            dim = int(np.size(M)**0.5)
            print('M is \n', M.reshape(dim, dim))
        T_mat = M
        return T_mat

    def least_square_psd_fit(self, Bss, rhos, disp=True, options=None):
        self.N = len(Bss[0]) if self.N is None else self.N
        D_S = int(np.size(Bss[0][0])**0.5)
        rho_vec = np.reshape(rhos, (len(rhos), -1))
        As_vec = []
        for As in Bss:
            As = qops.get_ops(As)
            AsAs = qmath.tensor([qmath.super2mat(A) for A in As[::-1]])
            As_vec.append(AsAs.reshape(-1))
        M, resids, rank, s = np.linalg.lstsq(As_vec, rho_vec, rcond=None)
        T_choi_mat = qmath.matrixize(self.matrix_to_choi(M))

        def err_func(T):
            M = self.choi_to_matrix(T)
            epison = np.linalg.norm(rho_vec - np.dot(As_vec, M))
            global tmp_i
            tmp_i += 1
            if not (tmp_i % 1000):
                print(tmp_i, 'current loss is ', epison)
            return epison

        global tmp_i
        tmp_i = 0
        T_choi_mat = psd.lstsq(err_func, T_choi_mat, unit_trace=False,
                               real=options['real'], disp=False, method=None,
                               options={
                                   'gtol': 1E-4,
                                   'maxiter': 10
                               })
        tmp_i = 0
        T_choi_mat = psd.lstsq(err_func, T_choi_mat, unit_trace=False,
                               real=options['real'], disp=False,
                               method=options['method'], options=options)
        tmp_i = 0
        T_mat = self.choi_to_matrix(T_choi_mat)
        return T_mat

    def fit(self, Bss, rhos, real=False, disp=False, options=None):
        if options is None:
            return self.least_square_fit(Bss, rhos, disp=False)
        else:
            return self.least_square_psd_fit(Bss, rhos, disp=disp,
                                             options=options)

    def rhose_out_ideal(self, rho0, As, Us):
        """Simulate the dynamics in the process tensor framework"""
        rho_se = rho0
        D_S, D_E = self.D_S, self.D_E
        for A, U in zip(As, Us):
            A = qops.get_op(A) if isinstance(A, (str, tuple)) else A
            U = qops.get_op(U) if isinstance(U, str) else U
            A_se = qmath.tensor((A, qmath.sigma_I(self.D_E)))
            rho_se = A_se @ rho_se @ A_se.conjugate().transpose()
            rho_se = U @ rho_se @ U.conjugate().transpose()
        return rho_se

    def sim(self, rho0, Bss, Us, return_format='matrix', options=None):
        """Simulate the closed two-qubit evolution process and fit the
        process tensor of 1-qubit open quantum evolution"""
        assert len(Bss[0]) == len(Us), '# Ctrl Ops should be paired with U_se'
        D_S, D_E = self.D_S, self.D_E
        rhos = []
        for As in Bss:
            rho_se = self.rhose_out_ideal(rho0, As, Us)
            rho_se = np.reshape(rho_se, (D_S, D_E, D_S, D_E))
            rho_m = np.trace(rho_se, axis1=1, axis2=3)
            rhos.append(rho_m)
        T_mat = self.fit(Bss, rhos, options=options)
        # , real=True
        if return_format == 'matrix':
            T_mat = T_mat
            return T_mat
        elif return_format == 'choi':
            T_choi = self.matrix_to_choi(T_mat)
            return T_choi
        else:
            raise KeyError

    def predict(self, As, T=None, rho_in=None, method='matrix'):
        """method could be matrix or choi or chi"""
        D_S, D_E = self.D_S, self.D_E
        num_steps = self.N if T is None else self.steps_of_tensor(T)
        if method == 'matrix':
            T = self.T_mat if T is None else T
            T = self.choi_to_matrix(self.T_choi) if T is None else T
            AsAs = qmath.tensor([self.A_to_AA(A) for A in As[::-1]])
            rho_m = np.reshape(np.dot(AsAs.reshape((1, -1)), T), (D_S, D_S))

        elif method == 'choi':
            As_choi = self.As_to_choi(As)
            T = self.T_choi if T is None else T
            T = self.matrix_to_choi(self.T_mat) if T is None else T
            T_sum_idx = np.arange(len(np.shape(T)))
            T_sum_idx[0] = 4 * num_steps + 2
            T_sum_idx[2 * num_steps + 1] = 4 * num_steps + 3
            A_sum_idx = np.arange(len(np.shape(As_choi)))
            A_sum_idx[0] = 0
            A_sum_idx[2 * num_steps + 1] = 0
            rho_m = np.einsum(T, T_sum_idx, As_choi, A_sum_idx)

        elif method == 'chi':
            Chis = self.Chis if T is None else T
            chi_dim = qmath.square_matrix_dim(Chis[0])
            if chi_dim == D_S**2:
                rho_dim = qmath.square_matrix_dim(rho_in)
                rho_in = self.trace_env(rho_in) if rho_dim != D_S else rho_in
                for A, Chi in zip(As, Chis):
                    if (isinstance(A, np.ndarray)
                            and qmath.square_matrix_dim(A) == D_S**2):
                        A_chi = A
                        rho_in = qpt.cal_process_rho(rho_in, A_chi)
                    else:
                        A = qops.get_op(A)
                        rho_in = A @ rho_in @ A.conjugate().transpose()
                    rho_out = qpt.cal_process_rho(rho_in, Chi)
                    rho_in = rho_out
                rho_m = rho_in
            elif chi_dim == (D_S * D_E)**2:
                rho_in = rho_in
                for A, Chi in zip(As, Chis):
                    if (isinstance(A, np.ndarray)
                            and qmath.square_matrix_dim(A) == D_S**2):
                        rho_in = qpt.cal_process_rhose(rho_in, A)
                    else:
                        A = qops.get_op(A)
                        A_se = qmath.tensor([A, qmath.sigma_I(D_E)])
                        rho_in = A_se @ rho_in @ A_se.conjugate().transpose()
                    rho_out = qpt.cal_process_rho(rho_in, Chi)
                    rho_in = rho_out
                rho_m = self.trace_env(rho_in)
            else:
                raise NotImplementedError
        return rho_m

    # Transformations
    def trace(self, As, T_choi=None, out_idx=-1):
        """
        T_Choi(r2 r1'r1 r0'r0 :: s2 s1's1 s0's0)
        A_Choi( 𝟙 r1'r1 r0'r0 ::  𝟙 s1's1 s0's0)
        MES/IΨ(|r1's1'><r1 s1 |⊗|r0's0'><r0 s0 |)
        tensor(r1's1'r0's0' : r1 s1 r0 s0 : 𝟙 𝟙)
        Ψ+Choi( 𝟙 r1'r1 r0'r0 ::  𝟙 s1's1 s0's0)
        Args:
            As, <list of A>, for example of 2 step process:
            [None, None] gives the original process tensor.
            [None, 'MIS'], out_idx=1: gives the contracted process tensor of
                the first step, ignoring the last step. (Maximally Mixed State)
            [I, None] gives the contracted process tensor of the last step,
                conditioned on the I operation on the fist step
            [A0, None] gives the averaged system state after the 1st step
                A0-U1:0 conditioned on operation A1 at the second step.
            ['MIS', 'MIS'], out_idx=0: gives the initial averaged system state.
            [I, I] gives the final averaged system state.
        ref [4] IV C
        ref [5] Appendix A3"""
        if T_choi is None:
            T_choi = self.T_choi
            step_els = self.N
        else:
            step_els = self.steps_of_tensor(T_choi)
        D_S, _ = self.D_S, self.D_E

        A_sum_ix = []
        trace_els = 0
        AsAs = 1
        out_idx = ((step_els + 1) + out_idx) % (step_els + 1)
        A_o_step = out_idx
        for i in range(step_els):
            if As[i] is None:
                assert out_idx > i, "Output must go beyond the contracted tensor!"
                A_o_step -= 1
                continue
            else:
                # (Λ⊗𝟙)[Ψ+] is just the choi-ordered AA
                AA = self.A_to_AA(As[i])
                trace_els += 1
            AsAs = qmath.tensor((AA, AsAs))
            A_sum_ix.insert(0, 2 * (step_els - i))
            A_sum_ix.insert(0, 2 * (step_els - i) - 1)
        # return origin tensor
        if trace_els == 0:
            return T_choi
        As_vec = AsAs.reshape(-1)
        As_out_vec = np.einsum(As_vec, [0], np.eye(D_S) / D_S, [1, 2])
        A_shape = 2 * trace_els * [D_S, D_S] + [D_S, D_S]
        A_tensor = np.reshape(As_out_vec, A_shape)
        A_choi = np.transpose(A_tensor, self.choi_order(trace_els))
        A_sum_ix = np.array([0] + A_sum_ix)
        A_sum_idx = np.hstack([A_sum_ix, A_sum_ix + 2 * step_els + 1])
        T_sum_idx = np.arange(len(np.shape(T_choi)))
        # T_Choi(r2 r1'r1 r0'r0 :: s2 s1's1 s0's0)
        # A_Choi( 𝟙 no'no r0'r0 ::  𝟙 no'no s0's0)
        # a:      tr                tr           -> last step
        #     (MIS)             (MIS)
        # b:           tr                tr      -> intermediate step
        #           MIS               MIS
        TA_o_trace_p = 2 * (2 * step_els + 1)
        A_o_idx_r = 2 * (trace_els - A_o_step)
        A_o_idx_s = A_o_idx_r + 2 * trace_els + 1
        A_sum_idx[A_o_idx_r] = TA_o_trace_p
        A_sum_idx[A_o_idx_s] = TA_o_trace_p
        # print('trace index are  ', T_sum_idx, A_sum_idx)
        return np.einsum(T_choi, T_sum_idx, A_choi, A_sum_idx)

    def matrix_to_choi(self, T_mat=None):
        T_mat = self.T_mat if T_mat is None else T_mat
        D_S, _ = self.D_S, self.D_E
        step_els = self.steps_of_tensor(T_mat)
        T_shape = 2 * step_els * [D_S, D_S] + [D_S, D_S]
        T_mtensor = np.reshape(T_mat, T_shape)
        return np.transpose(T_mtensor, self.choi_order(step_els))

    def choi_to_matrix(self, T_choi=None):
        T_choi = self.T_choi if T_choi is None else T_choi
        D_S, _ = self.D_S, self.D_E
        step_els = self.steps_of_tensor(T_choi)
        rvs_choi_order = self.choi_order(step_els, reverse=True)
        choi_shape = len(rvs_choi_order) * [D_S]
        if T_choi.shape != choi_shape:
            T_choi = np.reshape(T_choi, choi_shape)
        M_tensor = np.transpose(T_choi, rvs_choi_order)
        return np.reshape(M_tensor, ((D_S**4)**step_els, D_S**2))

    def lam_to_chi(self, lam=None, pms=None):
        """Convert the process tensor to Chi Matrix, using op=PMs
        If the output is not as predicted by the linear map (QPT), then the
        process is bilinear, indicating the exsitence of SE correlation. [6]
        """
        rhos_in = []
        rhos_out = []
        pms = QPT_PMs if pms is None else pms
        for pm_op in pms:
            pm = qops.get_op(pm_op)
            rho_m = self.predict([pm], T=lam, method='choi')
            # see the normalize factor Γ_n - [4](III) [5](Appendix.B)
            # rho_m = qmath.unit_trace(rho_m)  # rho_out should be normalized too
            if (abs(rho_m) < ZERO_RHO_TH).all():
                print('Prob of As {} is small for QPT, discard!'.format(pm_op))
                continue
            rhos_in.append(pm * np.trace(rho_m))
            rhos_out.append(rho_m)
        return qpt.qpt(rhos_in, rhos_out)

    def choi_1_to_lam(self, T_choi):
        # lam_mat(r1;s1:r0's0')
        # lamchoi(r1;r0':s1;s0')
        D_S, _ = self.D_S, self.D_E
        rhos_in = []
        rhos_out = []
        for pm_op in QPT_PMs:
            pm = qops.get_op(pm_op)
            rho_m = self.predict([pm], T=T_choi, method='choi')
            if (abs(rho_m) < ZERO_RHO_TH).all():
                print('Prob of As {} is small for QPT, discard!'.format(pm_op))
                continue
            rhos_in.append(pm * np.trace(rho_m))
            rhos_out.append(rho_m)
        lam_mat = qpt.chi_pointer_mat(rhos_in, rhos_out)
        return np.transpose(np.reshape(lam_mat, 4 * [D_S]), [0, 2, 1, 3])

    def choi_to_product_state(self, T_choi=None, options=None):
        D_S, _ = self.D_S, self.D_E
        T_choi = self.T_choi if T_choi is None else T_choi
        A_s = ['I'] * self.N
        prod_state = 1
        rho0_avg = self.trace(A_s, T_choi, out_idx=0)
        if self.N == 1:
            lam = self.choi_1_to_lam(T_choi)
            prod_state = qmath.tensor([lam, rho0_avg])
        else:
            for i in range(self.N):
                A_s = ['I'] * self.N
                A_s[i] = None
                if options is None:
                    T1_choi = self.trace(A_s, T_choi, out_idx=i + 1)
                else:
                    T1_choi = self.trace(A_s, T_choi, out_idx=i + 1,
                                         options=options)
                lam = self.choi_1_to_lam(T1_choi)
                prod_state = qmath.tensor([lam, prod_state])
            prod_state = qmath.tensor([prod_state, rho0_avg])
        return np.reshape(prod_state, [D_S, D_S] * (2 * self.N + 1))

    def non_markovianity(self):
        basis = self.basis
        assert self.N == 1
        A_s_rho0 = ['I'] * self.N
        rho_avg = self.trace(A_s_rho0, out_idx=0)
        lam_tr = np.trace(self.T_choi, axis1=2, axis2=5).reshape(2, 2, 2, 2)
        lam_qpt = qmath.matrixize(self.choi_1_to_lam(self.T_choi))
        resize_factor = np.trace(lam_qpt) / np.trace(qmath.matrixize(lam_tr))
        print('resize factor is ', resize_factor)
        lam_tr = qmath.matrixize(lam_tr) * resize_factor
        assert np.allclose(lam_tr, lam_qpt), '{} \n {}'.format(lam_tr, lam_qpt)
        T_markov = qmath.tensor(
            [qmath.matrixize(lam_qpt) / resize_factor, rho_avg / resize_factor])

        choi_state_corlat = qmath.matrixize(self.T_choi)
        choi_state_markov = qmath.matrixize(T_markov)
        D_relative_entropy = qmath.relative_entropy(choi_state_corlat,
                                                    choi_state_markov)
        return D_relative_entropy


class PTensorUC(ProcessTensor):
    def __init__(self, T_choi=None, T_mat=None):
        super().__init__(T_choi=T_choi, T_mat=T_mat)
        self.basis = 'uc'

    def trace(self, ops, out_idx=-1):
        raise NotImplementedError('Trace function is not supported!')


class PTensorPM(ProcessTensor):
    def __init__(self, T_choi=None, T_mat=None):
        super().__init__(T_choi=T_choi, T_mat=T_mat)
        self.basis = 'pm'

    def trace(self, As, T_choi=None, out_idx=-1, int_ops=['Pz+', 'Pz-'],
              options=None):
        """
        T_Choi(r2 r1'r1 r0'r0 :: s2 s1's1 s0's0)
        A_Choi( 𝟙 r1'r1 r0'r0 ::  𝟙 s1's1 s0's0)
        Args:
            As, <list of A>, for example of 2 step process:
            [None, None] gives the original process tensor.
            [None, 'I'], out_idx=1: gives the contracted process tensor of
                the first step, ignoring the last step.
                We use [Pz+, Pz-] as "Do nothing" operator. -[5](APPENDIX B)
        Returns:
            if None in As: The contracted pm tensor T_k:j
            else: <rho_avg, herald_rate> the average rho at out_idx
        ref: [5] Appendix B
        """
        if T_choi is None:
            T_choi = self.T_choi
            step_els = self.N
        else:
            step_els = self.steps_of_tensor(T_choi)
        D_S, _ = self.D_S, self.D_E
        assert len(As) == step_els

        # TRACE_IM_MAX = 1E-6
        TRACE_IM_MAX = 1E1
        out_step_mrk = np.array([1 if A is None else 0 for A in As])
        traced_steps = sum(out_step_mrk)
        out_idx = ((step_els + 1) + out_idx) % (step_els + 1)

        def get_init_steps(As):
            int_steps = []
            for i, A in enumerate(As):
                if isinstance(A, np.ndarray) and np.allclose(A, np.eye(D_S)):
                    int_steps.append(i)
                if isinstance(A, str) and A != 'pms' and np.allclose(
                        qops.get_op(A), np.eye(D_S)):
                    int_steps.append(i)
            return int_steps

        # return a intermediate state
        if not traced_steps:
            # return the final step, this should be the same as trace_tensor()
            if out_idx == step_els:
                A_s = list(As)  # this is fully refreshed in every loop
                int_idx = get_init_steps(A_s)
                int_pms = list(itertools.product(int_ops, repeat=len(int_idx)))
                rho_avg = 0
                for pms in int_pms:
                    for i, pm in zip(int_idx, pms):
                        A_s[i] = pm
                    r_p = self.predict(A_s, T_choi, method='choi')
                    rho_avg += r_p
            # return the intermediate state
            else:
                A_s = list(As)  # this is fully refreshed in every loop
                A_s[out_idx] = 'pms'
                int_idx = get_init_steps(A_s)
                int_pms = list(itertools.product(int_ops, repeat=len(int_idx)))
                B_probs = []
                for qst_pm in QST_PMs:
                    p_k_int = 0
                    for pms in int_pms:  # sum over different pms
                        for i, pm in zip(int_idx, pms):
                            A_s[i] = pm
                        A_s[out_idx] = qst_pm
                        r_p = self.predict(A_s, T_choi, method='choi')
                        p_k_int += np.trace(r_p)
                    assert p_k_int.imag < TRACE_IM_MAX, p_k_int.imag
                    assert abs(p_k_int) <= 1.0 + 0.1, 'P={}'.format(p_k_int)
                    B_probs.append(p_k_int)
                rho_avg = qst.qst(
                    np.array(B_probs).reshape(len(QST_PMs) // 2, 2), QST_BASIS)
            return rho_avg
        else:
            # return a contracted process tensor
            # loop over fit basis (B_k:j)
            Bss_ops = basis_ops('pm', traced_steps, complement=False)
            rhos_out = []
            for Bss_op in Bss_ops:
                A_s = [A for A in As]
                # put Bs fit to None of As
                Bs_els = 0
                for i in range(step_els):
                    if A_s[i] is None:
                        A_s[i] = Bss_op[Bs_els]
                        Bs_els += 1
                # Now calcuate the rho * p at out_idx
                rho_k = self.trace(A_s, T_choi, out_idx=out_idx)
                rhos_out.append(rho_k)
            # tomography.plotTrajectory(rhos_out)
            T_mat = self.fit(Bss_ops, rhos_out, options=options)
            T_choi = self.matrix_to_choi(T_mat)
            return T_choi

    def choi_to_qmaps(self, T_choi=None, options=None):
        """𝚼(choi state of T) = (⨂_1~k Λ_k:k-1)⊗ρ0"""
        A_s_rho0 = ['I'] * self.N
        Rho0s = []
        Lambdas = []
        Chis = []
        for i in range(self.N):
            Rho0s.append(self.trace(A_s_rho0, T_choi, out_idx=i))
            A_s_lam = ['I'] * self.N
            A_s_lam[i] = None
            T_lam = self.trace(A_s_lam, T_choi, out_idx=i + 1, options=options)
            Chis.append(self.lam_to_chi(T_lam))
            Lambdas.append(T_lam)
        self.Rho0s = Rho0s
        self.Chis = Chis
        self.Lambdas = Lambdas
        return self.Rho0s, self.Chis, self.Lambdas

    def qmap_prod_tensor(self, rho0, Chis, options=None):
        """Convert the quantum process to process tensor (Markovian)"""
        num_steps = len(Chis)
        Bss_ops = basis_ops(self.basis, num_steps, complement=False)
        rhos_out = []
        for Bss_op in Bss_ops:
            rhos_out.append(self.predict(Bss_op, Chis, rho0, method='chi'))
        T_mat = self.fit(Bss_ops, rhos_out, options=options)
        self.T_choi_prod = self.matrix_to_choi(T_mat)
        return self.T_choi_prod

    def non_markovianity(self, T_choi_ref=None, options=None):
        """the reference full tensor"""
        basis = self.basis
        assert self.N == 1
        A_s_rho0 = ['I'] * self.N
        rho_avg = self.trace(A_s_rho0, out_idx=0)
        lam_qpt = qmath.matrixize(self.choi_1_to_lam(T_choi=self.T_choi))

        # reference full process tensor
        ref_tensor = self.T_choi if T_choi_ref is None else T_choi_ref
        lam_ref = np.trace(ref_tensor, axis1=2, axis2=5).reshape(2, 2, 2, 2)
        lam_ref = qmath.matrixize(lam_ref)
        resize_factor = np.trace(lam_qpt) / np.trace(qmath.matrixize(lam_ref))
        print('resize factor is ', resize_factor)
        lam_qpt_resize = lam_qpt / resize_factor
        # ideal case
        # assert np.allclose(lam_ref, lam_qpt_resize), '{} \n {}'.format(
        #     lam_ref, lam_qpt_resize)
        T_markov = qmath.tensor([lam_qpt_resize, rho_avg / resize_factor])

        T_correl = qmath.matrixize(self.T_choi)
        # return a contracted process tensor
        # loop over fit basis (B_k:j)
        As_vec = []
        rho_vec = []
        for As in basis_ops('pm', self.N, complement=False):
            # put Bs fit to None of As
            rho_m = self.predict(As, method='choi')
            rho_vec.append(rho_m)
            As = qops.get_ops(As)
            AsAs = qmath.tensor([qmath.super2mat(A) for A in As[::-1]])
            As_vec.append(AsAs.reshape(-1))
        rho_vec = np.reshape(rho_vec, (len(rho_vec), -1))

        tmp_i = 0

        def err_func(T):
            M = self.choi_to_matrix(T)
            epison = np.linalg.norm(rho_vec - np.dot(As_vec, M))
            epison += qmath.relative_entropy(T, T_markov).real * 1E-2
            global tmp_i
            tmp_i += 1
            if not (tmp_i % 500):
                print(tmp_i, 'current loss is ', epison)
            return epison

        tmp_i = 0
        T_correl = psd.lstsq(err_func, T_correl, unit_trace=False,
                             real=options['real'], disp=False, method=None,
                             options={
                                 'gtol': 1E-4,
                                 'maxiter': 10
                             })
        tmp_i = 0
        T_correl = psd.lstsq(err_func, T_correl, unit_trace=False,
                             real=options['real'], disp=False,
                             method=options['method'], options=options)

        D_relative_entropy = qmath.relative_entropy(T_correl, T_markov)
        return D_relative_entropy


# REFERENCES:
# [1] Kuah, A. M., Modi,... & Sudarshan, E. C. G. (2007).
# How state preparation can affect a quantum experiment:
# Quantum process tomography for open systems.
# Physical Review A, 76(4), 1–12. https://doi.org/10.1103/PhysRevA.76.042113

# [2] Modi, K. (2012). Operational approach to open dynamics and quantifying
# initial correlations. Scientific Reports, 2.
# https://doi.org/10.1038/srep00581

# [3] Milz, S., Pollock, F. A., & Modi, K. (2017).
# An Introduction to Operational Quantum Dynamics.
# Open Systems and Information Dynamics, 24(4), 1–35.
# https://doi.org/10.1142/S1230161217400169

# [4] Pollock, F. A., Rodríguez-Rosario, C., Frauenheim, T., Paternostro, M.,
# & Modi, K. (2018). Non-Markovian quantum processes:
# Complete framework and efficient characterization.
# Physical Review A, 97(1), 1–13. https://doi.org/10.1103/PhysRevA.97.012127

# [5] Milz, S., Pollock, F. A., & Modi, K. (2018).
# Reconstructing non-Markovian quantum dynamics with limited control.
# Physical Review A, 98(1), 1–14. https://doi.org/10.1103/PhysRevA.98.012108
