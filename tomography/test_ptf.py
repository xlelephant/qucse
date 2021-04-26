import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=1000)

from . import qst, qpt, ptf, psd
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

for Uops in Uss_ops[1:]:

    Us_ops = Uops[:num_steps]

    # T of this process
    PT_cal = ptf.ProcessTensor()
    T_choi_cal = PT_cal.cal(rho0se, Us_ops, return_format='choi')
    T_choi_cal_square = qmath.matrixize(T_choi_cal)

    PT_fit = ptf.PTensorPM()
    T_choi_fit = PT_fit.sim(rho0se, Bss_ops, Us_ops, return_format='choi')
    T_choi_fit_square = qmath.matrixize(T_choi_fit)

    process_tensor = ptf.ProcessTensor(T_choi=PT_cal.T_choi)

    # Check the eigen value and positive semidefine
    ax = plt.figure().add_subplot()
    vals, vecs = np.linalg.eig(T_choi_cal_square)
    vals = np.array(sorted(vals, key=lambda s: abs(s)))
    ax.plot(vals.real, label='cal_real')
    ax.plot(vals.imag, label='cal_imag')
    vals, vecs = np.linalg.eig(T_choi_fit_square)
    vals = np.array(sorted(vals, key=lambda s: abs(s)))
    ax.plot(vals.real, label='fit_real')
    ax.plot(vals.imag, label='fit_imag')
    plt.legend()

    # print('bound of cal T.real is ', np.min(T_choi_cal_square.real),
    #       np.max(T_choi_cal_square.real))
    # print('bound of cal T.imag is ', np.min(T_choi_cal_square.imag),
    #       np.max(T_choi_cal_square.imag))
    # print('bound of cal T.real is ', np.min(T_choi_fit_square.real),
    #       np.max(T_choi_fit_square.real))
    # print('bound of cal T.imag is ', np.min(T_choi_fit_square.imag),
    #       np.max(T_choi_fit_square.imag))

    if not np.allclose(T_choi_fit_square, T_choi_cal_square):
        print('Fit T\n{} \nCal T\n{}'.format(T_choi_fit_square,
                                             T_choi_cal_square))
    else:
        print('T cal == T fit: \n', T_choi_cal_square)

    PM_ops = ptf.QPT_PMs  # this is for bilinear QPT
    ZERO_RHO_TH = 1E-6
    Chis = [
        qpt.gen_ideal_chi_matrix(U, PM_ops, rho0=rho0se, zero_th=ZERO_RHO_TH)
        for U in Us_ops
    ]

    Tests_ops = Bss_ops

    # # ========================== Reconstruction Test ==========================
    # Bs_sticks, As_sticks = [], []
    # rs_ideal, rs_t_fit, rs_t_cal, rs_qpt_s = [], [], [], []
    # fids_fit, fids_cal, fids_qpt = [], [], []
    # for i, As_op in enumerate(Tests_ops):
    #     rho_ideal = process_tensor.rhose_out_ideal(rho0se, As_op, Us_ops)
    #     rho_ideal = process_tensor.trace_env(rho_ideal)
    #     if (abs(rho_ideal) > ZERO_RHO_TH).any():
    #         if i < B_els:
    #             Bs_sticks.append(As_op)
    #         else:
    #             As_sticks.append(As_op)
    #     else:
    #         print('Prob of As {} is too small, discard!'.format(As_op))
    #         continue
    #     rs_ideal.append(rho_ideal)

    #     rho_fit = PT_fit.reconstruct(As_op, method='choi')
    #     rs_t_fit.append(rho_fit)
    #     fids_fit.append(qmath.fidelity_rho(rho_fit, rho_ideal, imag_atol=0.5))

    #     rho_cal = PT_cal.reconstruct(As_op, method='choi')
    #     rs_t_cal.append(rho_cal)
    #     fids_cal.append(qmath.fidelity_rho(rho_cal, rho_ideal, imag_atol=0.5))

    #     rho_qpt = process_tensor.reconstruct(As_op, Chis, rho0se, method='chi')
    #     rs_qpt_s.append(rho_qpt)
    #     fids_qpt.append(qmath.fidelity_rho(rho_qpt, rho_ideal, imag_atol=0.5))

    # # show
    # bar_width = 0.3
    # ax = plt.figure(tight_layout=True).add_subplot()
    # ax.set_ylim([0, 1.05])
    # xs = range(len(Bs_sticks))
    # ax.set_xticks(xs)
    # ax.set_xticklabels(ops2str(Bs_sticks))
    # line = ax.plot(xs, fids_qpt, 'k', ds='steps-mid', lw=2, label='qpt')
    # line = ax.plot(xs, fids_fit, '.r', ds='steps-mid', lw=2, label='fit')
    # line = ax.plot(xs, fids_cal, '-.b', ds='steps-mid', lw=2, label='cal')
    # ax.legend()

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

    # # Compare rhos
    # out_idxes = [2, 1, 0]
    # for As_ops in Tests_ops:
    #     for out_idx in out_idxes:
    #         A_s = As_ops[:out_idx]
    #         U_s = Us_ops[:out_idx]

    #         # # get rhos at step out_idx conditioned on As on later steps
    #         # # should not pass if we did not herald the state on later steps
    #         # print('out index: ', out_idx, As_ops)
    #         # r_cal = process_tensor.trace(As_ops, T_choi_cal, out_idx)
    #         # r_fit = process_tensor.trace(As_ops, T_choi_fit, out_idx)
    #         # assert np.allclose(r_sim, r_cal), '\n{}\n{}'.format(r_sim, r_cal)
    #         # assert np.allclose(r_cal, r_fit), '\n{}\n{}'.format(r_cal, r_fit)
    #         # print('conditional rhos at step {} allclose to \n{}'.format(
    #         #     out_idx, r_sim))

    #         # average rho at step out_idx without considering the later steps
    #         rse_sim = process_tensor.rhose_out_ideal(rho0se, A_s, U_s)
    #         r_sim = process_tensor.trace_env(rse_sim)
    #         A_I_s = process_tensor.N * ['I']
    #         A_I_s[:out_idx] = As_ops[:out_idx]
    #         r_cal = PT_cal.trace(A_I_s, out_idx=out_idx)
    #         r_fit = PT_fit.trace(A_I_s, out_idx=out_idx)
    #         assert np.allclose(r_sim, r_cal), '\n{}\n{}'.format(r_sim, r_cal)
    #         assert np.allclose(r_fit, r_cal), '\n{}\n{}'.format(r_fit, r_cal)
    #         print('average rhos at step {} allclose to \n{}'.format(
    #             out_idx, r_sim))

    # # Compare CPTP maps
    # step_index = [1]
    # As_ops = Tests_ops[0]
    # As_ops = ['I', 'I']
    # for idx in [0, 1]:
    #     A_s = As_ops[0:idx]
    #     U_s = Us_ops[0:idx]
    #     rhoi_se = process_tensor.rhose_out_ideal(rho0se, A_s, U_s)
    #     chi_sim = qpt.gen_ideal_chi_matrix(Us_ops[idx], PM_ops, rho0=rhoi_se,
    #                                        zero_th=ZERO_RHO_TH)
    #     A_I_s = process_tensor.N * ['I']
    #     A_I_s[:idx] = As_ops[:idx]
    #     A_I_s[idx] = None
    #     PT_lam_cal = ptf.PTensorPM(T_choi=PT_cal.trace(A_I_s, out_idx=idx + 1))
    #     PT_lam_fit = ptf.PTensorPM(T_choi=PT_fit.trace(A_I_s, out_idx=idx + 1))
    #     chi_cal = PT_lam_cal.lam_to_chi()
    #     chi_fit = PT_lam_fit.lam_to_chi()

    #     labels = qpt.pauli_vector_ops
    #     title = 'QPT step No. {} of U:{}'.format(idx, Us_ops)
    #     show_sim_chi = True
    #     figsize = (4, 4)
    #     if show_sim_chi:
    #         fig, ax = None, None
    #         fig, ax = show_chi(chi_sim, labels, alpha=0.5, figsize=figsize,
    #                            title=title + '(sim)', fig=fig, ax=ax)
    #     if not np.allclose(chi_sim, chi_cal):
    #         print('\n{}\n{}'.format(chi_sim, chi_cal))
    #         # fig, ax = None, None
    #         fig, ax = show_chi(chi_cal, labels, title=title + '(sim != cal)',
    #                            fig=fig, ax=ax, figsize=figsize)
    #         show_sim_chi = False

    #     if not np.allclose(chi_cal, chi_fit):
    #         print('\n{}\n{}'.format(chi_cal, chi_fit))
    #         # fig, ax = None, None
    #         fig, ax = show_chi(chi_fit, labels, title=title + '(cal != fit)',
    #                            fig=fig, ax=ax, figsize=figsize)
    #         show_sim_chi = False

    # # Compare qmaps for relative entropy calculation
    # Rho0s_cal, Chis_cal, _ = PT_cal.choi_to_qmaps(T_choi_cal)
    # Rho0s_fit, Chis_fit, _ = PT_fit.choi_to_qmaps(T_choi_fit)
    # for i, (rho_cal, rho_fit) in enumerate(zip(Rho0s_cal, Rho0s_fit)):
    #     if not np.allclose(rho_cal, rho_fit):
    #         print('average rhos at step {} differ: \n {}!=\n {}'.format(
    #             i, rho_cal, rho_fit))
    #     else:
    #         print('rhos at step {} match.'.format(i))
    # for i, (chi_cal, chi_fit) in enumerate(zip(Chis_cal, Chis_fit)):
    #     if not np.allclose(chi_cal, chi_fit):
    #         print('chis at step {} differ: \n {} != \n {}'.format(
    #             i, chi_cal, chi_fit))
    #     else:
    #         print('chis at step {} match.'.format(i))

    # ========================= Non-Markovianity Test =========================
    # Trace out the tensor using A0
    A0_thetas = 1 * np.pi * np.linspace(0.01, 1.0, 30, endpoint=False)
    # A0_thetas = 1 * np.pi * np.array([0.25])
    # 0.25, 0.5, 0.75, 0.99
    r1s_cal = []
    entropies_cal = []
    r1s_fit = []
    entropies_fit = []
    for theta in A0_thetas:
        print("======= theta = {} =======".format(theta))
        if basis == 'cb':
            A0 = qmath.rot_xy(theta, 0)  # UC is supported in full PT
            # A0 = A_op_to_A((theta, 0))
        elif basis == 'pm':
            A0 = (theta, 0)
        else:
            raise Exception('can not reduce map in {}'.format(basis))

        A_N_s = process_tensor.N * [None]
        A_N_s[0] = A0
        T_choi_1p_cal = PT_cal.trace(A_N_s)
        T_choi_1p_fit = PT_fit.trace(A_N_s)
        # T_choi_1p_fit = qmath.matrixize(T_choi_1p_fit)
        # vals = vals - min(vals)
        # vals = vals / np.sum(vals)
        # T_choi_1p_fit = vecs @ np.diag(vals) @ np.linalg.inv(vecs)
        # T_choi_1p_fit = psdm.approximate_correlation_matrix(T_choi_1p_fit)
        # T_choi_1p_fit = np.reshape(T_choi_1p_fit, 3 * [2, 2])
        # T_choi_1p_cal = T_choi_1p_cal / np.cos(theta / 2)**2
        # T_choi_1p_fit = T_choi_1p_fit / np.cos(theta / 2)**2
        PT1p_cal = ptf.ProcessTensor(T_choi=T_choi_1p_cal)
        PT1p_fit = ptf.ProcessTensor(T_choi=T_choi_1p_fit)

        # fig = plt.figure()
        # # offset the choi state
        # # T_choi_1p_cal_square = qmath.matrixize(T_choi_cal)
        # # T_choi_1p_fit_square = qmath.matrixize(T_choi_fit)
        # T_choi_1p_cal_square = qmath.matrixize(T_choi_1p_cal)
        # T_choi_1p_fit_square = qmath.matrixize(T_choi_1p_fit)
        # vals, vecs = np.linalg.eig(T_choi_1p_cal_square)
        # vals = np.array(sorted(vals, key=lambda s: abs(s)))
        # ax = fig.add_subplot()
        # ax.plot(vals.real, '-.', label='cal_1:{}_real'.format(PT1p_cal.N + 1))
        # ax.plot(vals.imag, '-.', label='cal_1:{}_imag'.format(PT1p_cal.N + 1))
        # # # ax = fig.add_subplot(122)
        # vals, vecs = np.linalg.eig(T_choi_1p_fit_square)
        # vals = np.array(sorted(vals, key=lambda s: abs(s)))
        # ax.plot(vals.real, '--', label='fit_1:{}_real'.format(PT1p_fit.N + 1))
        # ax.plot(vals.imag, '--', label='fit_1:{}_imag'.format(PT1p_fit.N + 1))
        # ax.set_title(r'{}: theta = {:4f} $\pi$'.format(Us_ops, theta / np.pi))
        # plt.legend()
        # # plt.legend()

        entropies_cal.append(PT1p_cal.non_markovianity() /
                             np.cos(theta / 2)**2)
        entropies_fit.append(PT1p_fit.non_markovianity() /
                             np.cos(theta / 2)**2)

    fig = plt.figure()
    ax = fig.add_subplot()
    xs = A0_thetas / np.pi
    ax.plot(xs, np.array(entropies_cal).real, '-', label='N_cal real')
    ax.plot(xs, np.array(entropies_cal).imag, '-.', label='N_cal imag')
    ax.plot(xs, np.array(entropies_fit).real, '--', label='N_fit real')
    ax.plot(xs, np.array(entropies_fit).imag, '-.', label='N_fit imag')
    ax.plot([0.5, 0.5], [0, np.max(entropies_cal)], 'k')
    plt.legend()

plt.show()
