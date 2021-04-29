import numpy as np
import matplotlib.pyplot as plt

from qucse_xl import qmath, qops
from qucse_xl.tomography import qst, qpt, ptf, psd
from qucse_xl.show.matrix import show_chi

PT_exp = None
np.set_printoptions(precision=6, suppress=True, linewidth=1000)

num_steps = 2
Uss_ops = [
    ["CZ", "CNOT", "iSwap"],
    ["CNOT", "CZ", "iSwap"],
]
Uops = Uss_ops[0]
basis = 'pm'

# Test projective measurements
Bss_ops = ptf.basis_ops(basis, num_steps, False)
Us_ops = Uops[:num_steps]

PT_cal = ptf.ProcessTensor()
rho0se = qops.get_init_rho(('I', 'I'))
PT_cal.T_choi = PT_cal.cal(rho0se, Us_ops, return_format='choi')
print('trace of  PT_cal is ', PT_cal.trace())
# so the trace of a process tensor should be PT.N * 2
# and if the contracted tensor has the prob p, then trace = PT_tr.N * 2 * p

PT_fit = ptf.PTensorPM()
PT_fit.T_choi = PT_fit.sim(rho0se, Bss_ops, Us_ops, return_format='choi')

# ========================= Non-Markovianity Test =========================
# Trace out the tensor using A0
A0_thetas = 1 * np.pi * np.linspace(0.01, 1.0, 12, endpoint=False)
# A0_thetas = 1 * np.pi * np.array([0.25])
entropies_cal = []
entropies_fit = []
entropies_exp = []
exp_lsq_err = []
xs = []
for theta in A0_thetas[::-1]:
    xs.append(theta / np.pi)
    print("======= theta = {} =======".format(theta / np.pi))
    A_N_s = PT_cal.N * [None]
    A_N_s[0] = qmath.rot_xy(theta, 0) if basis == 'cb' else 'X'
    A_N_s[0] = (theta, 0) if basis == 'pm' else A_N_s[0]
    # A_N_s[0] = qmath.rot_xy(theta, 0)
    t_norm = 1 * 2
    pm_probs = np.cos(theta / 2)**2
    print('prob of A0x2 is ', pm_probs * 2)

    # ================ PT_cal non-markovianity ================

    PT_1p_sim = ptf.ProcessTensor()
    PT_1p_cal = ptf.ProcessTensor(T_choi=PT_cal.contract(A_N_s))
    PT1c_prod = ptf.ProcessTensor(T_choi=PT_1p_cal.choi_to_product_state())
    # PT1c_prod = ptf.ProcessTensor(T_choi=PT1c_prod.normalize(PT_1p_cal.T_choi))
    print('tr PTs cal are: ', PT_1p_cal.trace(), PT1c_prod.trace())

    rho1_avg = PT_1p_sim.rhos_out_ideal(rho0se, A_N_s, Us_ops[:1])
    rho2_avg = PT_1p_cal.contract(['I'], out_idx=0)
    rho3_avg = PT1c_prod.contract(['I'], out_idx=0)
    assert np.allclose(rho1_avg, rho2_avg)
    assert np.allclose(rho1_avg, rho3_avg), '{}\n{}'.format(rho1_avg, rho3_avg)

    PT1c_norm = ptf.ProcessTensor(T_choi=PT_1p_cal.normalize())
    NM = PT1c_norm.non_markovianity()
    entropies_cal.append(NM)

    # # check fit pt

    # PT_1p_fit = ptf.PTensorPM(T_choi=PT_fit.contract(A_N_s, options=None))
    # PT1f_prod = ptf.ProcessTensor(T_choi=PT_1p_fit.choi_to_product_state())
    # assert np.allclose(PT1c_prod.T_choi, PT1f_prod.T_choi)
    # rho2_avg = PT_1p_fit.contract(['I'], out_idx=0)
    # rho3_avg = PT1f_prod.contract(['I'], out_idx=0)
    # assert np.allclose(rho2_avg, rho3_avg), '{}\n{}'.format(rho2_avg, rho3_avg)
    # A1 = qmath.random_pm()
    # rho2_avg = PT_1p_fit.predict([A1], method='choi')
    # rho3_avg = PT1f_prod.predict([A1], method='choi')
    # # assert np.allclose(rho2_avg, rho3_avg), '{}\n{}'.format(rho2_avg, rho3_avg)

    # ================ PT_fit (ideal) non-markovianity ================

    # new fitting to ps-matrix
    options = {
        'ftol': 1E-8,
        'maxiter': 200,
        'method': 'SLSQP',  # SLSQP
        'real': False
    }
    # PT_1p_fit = ptf.PTensorPM(T_choi=PT_fit.contract(A_N_s, options=None))
    PT_1p_fit = ptf.PTensorPM(T_choi=PT_fit.contract(A_N_s))
    PT1f_prod = ptf.ProcessTensor(T_choi=PT_1p_fit.choi_to_product_state())
    print('tr PT fit are: ', PT_1p_fit.trace(), PT1f_prod.trace())

    PT1f_norm = ptf.PTensorPM(PT_1p_fit.normalize(factor=PT1f_prod.trace()))
    PT1f_prod_norm = ptf.ProcessTensor(T_choi=PT1f_prod.normalize())
    NM, _ = PT1f_norm.non_markovianity(T_markov=PT1f_prod_norm.T_choi,
                                       T_guess=PT1f_prod_norm.T_choi,
                                       options=options)

    entropies_fit.append(NM)

    if PT_exp is None:
        continue
    # ================ PT_fit (expr) non-markovianity ================
    print("~~~~~~~~~~~~~ Get the nm of experimental data ~~~~~~~~~~~~~")
    PT_1p_exp = ptf.PTensorPM(T_choi=PT_exp.contract(A_N_s))
    PT1e_prod = ptf.ProcessTensor(T_choi=PT_1p_exp.choi_to_product_state())
    print('tr PT exp are: ', PT_1p_exp.trace(), PT1e_prod.trace())

    PT1e_norm = ptf.PTensorPM(PT_1p_exp.normalize(factor=PT1e_prod.trace()))
    PT1e_prod_norm = ptf.ProcessTensor(T_choi=PT1e_prod.normalize())
    NM, lsq_err = PT1e_norm.non_markovianity(T_markov=PT1e_prod_norm.T_choi,
                                             T_guess=PT1e_prod_norm.T_choi,
                                             options=options)

    entropies_exp.append(NM)
    exp_lsq_err.append(lsq_err)

ax = plt.figure().add_subplot()
ax.plot(xs, np.array(entropies_cal).real, '-', label='N_cal')
# ax.plot(xs, np.array(entropies_cal).imag, '-.', label='N_cal imag')
ax.plot(xs, np.array(entropies_fit).real, '--', label='N_fit')
# ax.plot(xs, np.array(entropies_fit).imag, '-.', label='N_fit imag')
if PT_exp is not None:
    ax.plot(xs, np.array(entropies_exp).real, '--', label='N_exp real')
    ax.plot(xs, np.array(entropies_exp).imag, '-.', label='N_exp imag')
    ax.bar(xs,
           np.array(exp_lsq_err) * 1E2,
           abs(xs[1] - xs[0]) * 0.3, fc='k', ec='None', lw=2, alpha=0.5,
           label='lsq err[%] PT_argmin(NM)')

ax.plot([0.5, 0.5], [0, np.max(entropies_cal)], 'k')
ax.plot(0, 1, [np.log(2)] * 2, '-.k')
ax.set_title('Non-markovianity of process {}'.format(Us_ops))
ax.legend()
plt.show()
