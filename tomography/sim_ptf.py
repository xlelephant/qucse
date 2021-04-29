import numpy as np
import matplotlib.pyplot as plt

from qucse_xl import qmath, qops
from qucse_xl.tomography import qst, qpt, ptf, psd
from qucse_xl.show.matrix import show_chi

np.set_printoptions(precision=6, suppress=True, linewidth=1000)
num_steps = 2
Uss_ops = [
    ["CZ", "CNOT", "iSwap"],
    ["CNOT", "CZ", "iSwap"],
]
Uops = Uss_ops[1]

# Test projective measurements
basis = 'pm'
Bss_ops = ptf.basis_ops(basis, num_steps, False)
B_els = len(Bss_ops)
Ass_ops = ptf.basis_ops(basis, num_steps, True)
rho0se = qops.get_init_rho(('I', 'I'))
A_els = len(Ass_ops)
Us_ops = Uops[:num_steps]

# T of this process
PT_cal = ptf.ProcessTensor()
PT_cal.T_choi = PT_cal.cal(rho0se, Us_ops, return_format='choi')

PT_fit = ptf.PTensorPM()
options = {'gtol': 1E-6, 'maxiter': 1000, 'method': None, 'real': True}
options = {
    'ftol': 1E-8,
    'maxiter': 500,
    'method': 'SLSQP',  # SLSQP
    'real': True
}
options = None
PT_fit.T_choi = PT_fit.sim(rho0se, Bss_ops, Us_ops, return_format='choi',
                           options=options)

T_choi_cal_square = qmath.matrixize(PT_cal.T_choi)
T_choi_fit_square = qmath.matrixize(PT_fit.T_choi)

# # Check the eigen value and positive semidefine
# ax = plt.figure().add_subplot()
# vals, vecs = np.linalg.eig(T_choi_cal_square)
# vals = np.array(sorted(vals, key=lambda s: abs(s)))
# ax.plot(vals.real, label='cal_real')
# ax.plot(vals.imag, label='cal_imag')
# vals, vecs = np.linalg.eig(T_choi_fit_square)
# vals = np.array(sorted(vals, key=lambda s: abs(s)))
# ax.plot(vals.real, label='fit_real')
# ax.plot(vals.imag, label='fit_imag')
# plt.legend()

Tests_ops = Bss_ops
process_tensor = ptf.ProcessTensor(T_choi=PT_cal.T_choi)
PM_ops = ptf.QPT_PMs  # this is for bilinear QPT
ZERO_RHO_TH = 1E-6

# # ========================== Reconstruction Test ==========================

# Chis = [
#     qpt.gen_ideal_chi_matrix(U, PM_ops, rho0=rho0se, zero_th=ZERO_RHO_TH)
#     for U in Us_ops
# ]

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

#     rho_fit = PT_fit.predict(As_op, method='choi')
#     rs_t_fit.append(rho_fit)
#     fids_fit.append(qmath.fidelity_rho(rho_fit, rho_ideal, imag_atol=0.5))

#     rho_cal = PT_cal.predict(As_op, method='choi')
#     rs_t_cal.append(rho_cal)
#     fids_cal.append(qmath.fidelity_rho(rho_cal, rho_ideal, imag_atol=0.5))

#     rho_qpt = process_tensor.predict(As_op, Chis, rho0se, method='chi')
#     rs_qpt_s.append(rho_qpt)
#     fids_qpt.append(qmath.fidelity_rho(rho_qpt, rho_ideal, imag_atol=0.5))

# # show
# bar_width = 0.3
# ax = plt.figure(tight_layout=True).add_subplot()
# ax.set_ylim([0, 1.05])
# xs = range(len(Bs_sticks))
# ax.set_xticks(xs)
# ax.set_xticklabels(ptf.ops2str(Bs_sticks))
# line = ax.plot(xs, fids_qpt, 'k', ds='steps-mid', lw=2, label='qpt')
# line = ax.plot(xs, fids_fit, '.r', ds='steps-mid', lw=2, label='fit')
# line = ax.plot(xs, fids_cal, '-.b', ds='steps-mid', lw=2, label='cal')
# ax.legend()

# # =========================== Containment Test ===========================
# rand_stats = 3
# if basis == 'pm' or basis == 'cb':
#     Tests_ops = [[qmath.random_theta_phi() for _ in range(num_steps)]
#                  for _ in range(rand_stats)]
# # elif basis == 'uc':
# #     Tests_ops = [[
# #         qmath.su2euler(qmath.random_su(2)) for _ in range(num_steps)
# #     ] for _ in range(rand_stats)]
# else:
#     raise NotImplementedError

# # Compare rhos
# out_idxes = [2, 1, 0]
# for As_ops in Tests_ops:
#     for out_idx in out_idxes:
#         A_s = As_ops[:out_idx]
#         U_s = Us_ops[:out_idx]

#         # # get rhos at step out_idx conditioned on As on later steps
#         # # should not pass if we did not herald the state on later steps
#         # print('out index: ', out_idx, As_ops)
#         # r_cal = process_tensor.contract(As_ops, T_choi_cal, out_idx)
#         # r_fit = process_tensor.contract(As_ops, T_choi_fit, out_idx)
#         # assert np.allclose(r_sim, r_cal), '\n{}\n{}'.format(r_sim, r_cal)
#         # assert np.allclose(r_cal, r_fit), '\n{}\n{}'.format(r_cal, r_fit)
#         # print('conditional rhos at step {} allclose to \n{}'.format(
#         #     out_idx, r_sim))

#         # average rho at step out_idx without considering the later steps
#         rse_sim = process_tensor.rhose_out_ideal(rho0se, A_s, U_s)
#         r_sim = process_tensor.trace_env(rse_sim)
#         A_I_s = process_tensor.N * ['I']
#         A_I_s[:out_idx] = As_ops[:out_idx]
#         r_cal = PT_cal.contract(A_I_s, out_idx=out_idx)
#         r_fit = PT_fit.contract(A_I_s, out_idx=out_idx)
#         assert np.allclose(r_sim, r_cal), '\n{}\n{}'.format(r_sim, r_cal)
#         assert np.allclose(r_fit, r_cal), '\n{}\n{}'.format(r_fit, r_cal)
#         print('average rhos at step {} allclose to \n{}'.format(
#             out_idx, r_sim))

# Compare CPTP maps
step_index = [1]
for idx in [0, 1]:
    As_ops = ['Pyz+', 'I']
    A_s = As_ops[0:idx]
    U_s = Us_ops[0:idx]
    rhoi_se = process_tensor.rhose_out_ideal(rho0se, A_s, U_s)
    PM_ops = qst.TOMO_BASIS_OPS['pm_octomo']
    chi_sim = qpt.gen_ideal_chi_matrix(Us_ops[idx], PM_ops, rho0=rhoi_se,
                                       zero_th=ZERO_RHO_TH)
    A_I_s = process_tensor.N * ['I']
    As_ops = ['Pyz+', 'I']
    A_I_s[:idx] = As_ops[:idx]
    A_I_s[idx] = None
    PT_lam_cal = ptf.PTensorPM(T_choi=PT_cal.contract(A_I_s, out_idx=idx + 1))
    chi_cal = PT_lam_cal.lam_to_chi(pms=qst.TOMO_BASIS_OPS['pm_octomo'])
    As_ops = ['Pyz+', 'I']
    A_I_s[:idx] = As_ops[:idx]
    A_I_s[idx] = None
    PT_lam_fit = ptf.PTensorPM(T_choi=PT_fit.contract(A_I_s, out_idx=idx + 1))
    chi_fit = PT_lam_fit.lam_to_chi(pms=qst.TOMO_BASIS_OPS['pm_octomo'])

    labels = qpt.pauli_vector_ops
    pt_name = 'QPT step No. {} of U:{}'.format(idx, Us_ops)
    show_sim_chi = False
    figsize = (4, 4)
    if not np.allclose(chi_sim, chi_cal):
        print('\n{}\n{}'.format(chi_sim, chi_cal))
        title = pt_name + '(sim != cal)'
        fig, ax = None, None
        fig, ax = show_chi(chi_cal, labels, title=title, fig=fig, ax=ax,
                           figsize=figsize)
        show_sim_chi = True
    if not np.allclose(chi_cal, chi_fit):
        print('\n{}\n{}'.format(chi_cal, chi_fit))
        title = pt_name + '(cal != fit)'
        fig, ax = None, None
        fig, ax = show_chi(chi_fit, labels, title=title, fig=fig, ax=ax,
                           figsize=figsize)
        show_sim_chi = True
    if show_sim_chi:
        fig, ax = None, None
        title = pt_name + '(sim)'
        fig, ax = show_chi(chi_sim, labels, alpha=0.5, figsize=figsize,
                           title=title, fig=fig, ax=ax)
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

# # ========================= Non-Markovianity Test =========================
# # Trace out the tensor using A0
# A0_thetas = 1 * np.pi * np.linspace(0.01, 1.0, 25, endpoint=False)
# # A0_thetas = 1 * np.pi * np.array([0.25])
# entropies_cal = []
# entropies_fit = []
# for theta in A0_thetas:
#     options = {
#         'ftol': 1E-10,
#         'maxiter': 1000,
#         'method': 'SLSQP',  # SLSQP
#         'real': True
#     }

#     print("======= theta = {} =======".format(theta / np.pi))
#     A_N_s = PT_cal.N * [None]
#     A_N_s[0] = qmath.rot_xy(theta, 0) if basis == 'cb' else 'X'
#     A_N_s[0] = (theta, 0) if basis == 'pm' else 'X'

#     PT_pm_cal = ptf.PTensorPM(T_choi=PT_cal.T_choi)
#     PT_1p_cal = ptf.PTensorPM(PT_pm_cal.contract(A_N_s))
#     print("cal Markovianity for Cal")
#     entropies_cal.append(PT_1p_cal.non_markovianity())
#     # entropies_cal.append(
#     #     PT_1p_cal.non_markovianity(options=options) / np.cos(theta / 2)**2)

#     print("cal Markovianity for PM")
#     PT_1p_fit = ptf.PTensorPM(T_choi=PT_fit.contract(A_N_s))
#     entropies_fit.append(PT_1p_fit.non_markovianity())
#     # entropies_fit.append(
#     #     PT_1p_fit.non_markovianity(options=options) / np.cos(theta / 2)**2)

# fig = plt.figure()
# ax = fig.add_subplot()
# xs = A0_thetas / np.pi
# ax.plot(xs, np.array(entropies_cal).real, '-', label='N_cal real')
# ax.plot(xs, np.array(entropies_cal).imag, '-.', label='N_cal imag')
# ax.plot(xs, np.array(entropies_fit).real, '--', label='N_fit real')
# ax.plot(xs, np.array(entropies_fit).imag, '-.', label='N_fit imag')
# ax.plot([0.5, 0.5], [0, np.max(entropies_cal)], 'k')
# plt.legend()

plt.show()

# fig = plt.figure()
# # offset the choi state
# # PT_1p_cal_square = qmath.matrixize(T_choi_cal)
# # T_choi_1p_fit_square = qmath.matrixize(T_choi_fit)
# PT_1p_cal_square = qmath.matrixize(PT_1p_cal)
# T_choi_1p_fit_square = qmath.matrixize(T_choi_1p_fit)
# vals, vecs = np.linalg.eig(PT_1p_cal_square)
# vals = np.array(sorted(vals, key=lambda s: abs(s)))
# ax = fig.add_subplot()
# ax.plot(vals.real, '-.', label='cal_1:{}_real'.format(PT_1p_cal.N + 1))
# ax.plot(vals.imag, '-.', label='cal_1:{}_imag'.format(PT_1p_cal.N + 1))
# # # ax = fig.add_subplot(122)
# vals, vecs = np.linalg.eig(T_choi_1p_fit_square)
# vals = np.array(sorted(vals, key=lambda s: abs(s)))
# ax.plot(vals.real, '--', label='fit_1:{}_real'.format(PT_1p_fit.N + 1))
# ax.plot(vals.imag, '--', label='fit_1:{}_imag'.format(PT_1p_fit.N + 1))
# ax.set_title(r'{}: theta = {:4f} $\pi$'.format(Us_ops, theta / np.pi))
# plt.legend()
# # plt.legend()
