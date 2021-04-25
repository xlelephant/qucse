import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=1000)

from . import qst, qpt, ptf
from .. import qmath, qops
from ..show.matrix import show_chi


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


num_steps = 2
Uss_ops = [
    # ["II", "iSwap"],
    # ["iSwap/2", "iSwap/2"],
    # ["CZ", "iSwap", "CZ"],
    # ["iSwap", "iSwap", "CZ"],
    # ["II", "II", "II"],
    ["CZ", "CNOT", "iSwap"],
    ["CNOT", "CZ", "iSwap"],
]

# Test projective measurements
basis = 'pm'
Bss_ops = ptf.basis_ops(basis, num_steps, False)
B_els = len(Bss_ops)
Ass_ops = ptf.basis_ops(basis, num_steps, True)
rho0se = qops.get_init_rho(('I', 'I'))
A_els = len(Ass_ops)

for Uops in Uss_ops:
    if basis == 'pm':
        process_tensor = ptf.PTensorPM()
    elif basis == 'uc':
        process_tensor = ptf.PTensorUC()
    elif basis == 'cb':
        process_tensor = ptf.ProcessTensor()

    Us_ops = Uops[:num_steps]

    # T of this process

    process_tensor.sim(rho0se, Bss_ops, Us_ops, return_format='matrix')
    T_choi_fit = process_tensor.matrix_to_choi()
    T_choi_cal = process_tensor.cal(rho0se, Us_ops, return_format='choi')

    T_choi_fit_square = qmath.matrixize(T_choi_fit)
    T_choi_cal_square = qmath.matrixize(T_choi_cal)

    # offset the choi state
    # vals, vecs = np.linalg.eig(T_choi_fit_square)
    # vals = vals - min(vals)
    # T_choi_fit_square = vecs @ np.diag(vals) @ np.linalg.inv(vecs)
    # T_choi_fit = np.reshape(T_choi_fit_square, 5 * [2, 2])
    # ax = plt.figure().add_subplot()
    # ax.plot(vals.real)
    # ax.plot(vals.imag)
    if not np.allclose(T_choi_fit_square, T_choi_cal_square):
        print('Fit T\n{} \nCal T\n{}'.format(T_choi_fit_square,
                                             T_choi_cal_square))
    else:
        print('T cal == T fit: \n', T_choi_cal_square)

    # χ-matrix of every step

    PM_ops = ptf.QPT_PMs  # this is for bilinear QPT
    ZERO_RHO_TH = 1E-6
    Chis = [
        qpt.gen_ideal_chi_matrix(U, PM_ops, rho0=rho0se, zero_th=ZERO_RHO_TH)
        for U in Us_ops
    ]

    # ========================== Reconstruction Test ==========================

    Tests_ops = Bss_ops
    Bs_sticks, As_sticks = [], []
    rs_ideal, rs_t_fit, rs_t_cal, rs_qpt_s = [], [], [], []
    fids_fit, fids_cal, fids_qpt = [], [], []
    for i, As_op in enumerate(Tests_ops):
        rho_ideal = process_tensor.rhose_out_ideal(rho0se, As_op, Us_ops)
        rho_ideal = process_tensor.trace_env(rho_ideal)
        if (abs(rho_ideal) > ZERO_RHO_TH).any():
            if i < B_els:
                Bs_sticks.append(As_op)
            else:
                As_sticks.append(As_op)
        else:
            print('Prob of As {} is too small, discard!'.format(As_op))
            continue
        rs_ideal.append(rho_ideal)

        rho_fit = process_tensor.reconstruct(As_op, T_choi_fit, method='choi')
        rs_t_fit.append(rho_fit)
        fids_fit.append(qmath.fidelity_rho(rho_fit, rho_ideal, imag_atol=0.5))

        rho_cal = process_tensor.reconstruct(As_op, T_choi_cal, method='choi')
        rs_t_cal.append(rho_cal)
        fids_cal.append(qmath.fidelity_rho(rho_cal, rho_ideal, imag_atol=0.5))

        rho_qpt = process_tensor.reconstruct(As_op, Chis, rho0se, method='chi')
        rs_qpt_s.append(rho_qpt)
        fids_qpt.append(qmath.fidelity_rho(rho_qpt, rho_ideal, imag_atol=0.5))

    # show
    bar_width = 0.3
    ax = plt.figure(tight_layout=True).add_subplot()
    ax.set_ylim([0, 1.05])
    xs = range(len(Bs_sticks))
    ax.set_xticks(xs)
    ax.set_xticklabels(ops2str(Bs_sticks))
    line = ax.plot(xs, fids_qpt, 'k', ds='steps-mid', lw=2, label='qpt')
    line = ax.plot(xs, fids_fit, '.r', ds='steps-mid', lw=2, label='fit')
    line = ax.plot(xs, fids_cal, '-.b', ds='steps-mid', lw=2, label='cal')
    ax.legend()

    # =========================== Containment Test ===========================
    rand_stats = 3
    if basis == 'pm' or basis == 'cb':
        Tests_ops = [[qmath.random_theta_phi() for _ in range(num_steps)]
                     for _ in range(rand_stats)]
    # elif basis == 'uc':
    #     Tests_ops = [[
    #         qmath.su2euler(qmath.random_su(2)) for _ in range(num_steps)
    #     ] for _ in range(rand_stats)]
    else:
        raise NotImplementedError

    # Compare rhos
    out_idxes = [2, 1, 0]
    for As_ops in Tests_ops:
        for out_idx in out_idxes:
            A_s = As_ops[:out_idx]
            U_s = Us_ops[:out_idx]

            # # get rhos at step out_idx conditioned on As on later steps
            # # should not pass if we did not herald the state on later steps
            # print('out index: ', out_idx, As_ops)
            # r_cal = process_tensor.trace(As_ops, T_choi_cal, out_idx)
            # r_fit = process_tensor.trace(As_ops, T_choi_fit, out_idx)
            # assert np.allclose(r_sim, r_cal), '\n{}\n{}'.format(r_sim, r_cal)
            # assert np.allclose(r_cal, r_fit), '\n{}\n{}'.format(r_cal, r_fit)
            # print('conditional rhos at step {} allclose to \n{}'.format(
            #     out_idx, r_sim))

            # average rho at step out_idx without considering the later steps
            rse_sim = process_tensor.rhose_out_ideal(rho0se, A_s, U_s)
            r_sim = process_tensor.trace_env(rse_sim)
            A_I_s = process_tensor.N * ['I']
            A_I_s[:out_idx] = As_ops[:out_idx]
            r_cal = process_tensor.trace(A_I_s, T_choi_cal, out_idx)
            r_fit = process_tensor.trace(A_I_s, T_choi_fit, out_idx)
            assert np.allclose(r_sim, r_cal), '\n{}\n{}'.format(r_sim, r_cal)
            assert np.allclose(r_fit, r_cal), '\n{}\n{}'.format(r_fit, r_cal)
            print('average rhos at step {} allclose to \n{}'.format(
                out_idx, r_sim))

    # Compare CPTP maps
    step_index = [1]
    As_ops = Tests_ops[0]
    As_ops = ['Pxz-', 'I']
    for idx in [0, 1]:
        A_s = As_ops[0:idx]
        U_s = Us_ops[0:idx]
        rhoi_se = process_tensor.rhose_out_ideal(rho0se, A_s, U_s)
        chi_sim = qpt.gen_ideal_chi_matrix(Us_ops[idx], PM_ops, rho0=rhoi_se,
                                           zero_th=ZERO_RHO_TH)
        A_I_s = process_tensor.N * ['I']
        A_I_s[:idx] = As_ops[:idx]
        A_I_s[idx] = None
        PT_lam_cal = ptf.ProcessTensor(
            T_choi=process_tensor.trace(A_I_s, T_choi_cal, idx + 1))
        PT_lam_fit = ptf.ProcessTensor(
            T_choi=process_tensor.trace(A_I_s, T_choi_fit, idx + 1))
        chi_cal = PT_lam_cal.lam_to_chi()
        chi_fit = PT_lam_fit.lam_to_chi()

        labels = qpt.pauli_vector_ops
        title = 'QPT step No. {} of U:{}'.format(idx, Us_ops)
        show_sim_chi = True
        figsize = (4, 4)
        if show_sim_chi:
            fig, ax = None, None
            fig, ax = show_chi(chi_sim, labels, alpha=0.5, figsize=figsize,
                               title=title + '(sim)', fig=fig, ax=ax)
        if not np.allclose(chi_sim, chi_cal):
            print('\n{}\n{}'.format(chi_sim, chi_cal))
            # fig, ax = None, None
            fig, ax = show_chi(chi_cal, labels, title=title + '(sim != cal)',
                               fig=fig, ax=ax, figsize=figsize)
            show_sim_chi = False

        if not np.allclose(chi_cal, chi_fit):
            print('\n{}\n{}'.format(chi_cal, chi_fit))
            # fig, ax = None, None
            fig, ax = show_chi(chi_fit, labels, title=title + '(cal != fit)',
                               fig=fig, ax=ax, figsize=figsize)
            show_sim_chi = False

plt.show()
