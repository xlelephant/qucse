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
basis = 'pm'

# Test projective measurements
Bss_ops = ptf.basis_ops(basis, num_steps, False)
Us_ops = Uops[:num_steps]

PT_cal = ptf.ProcessTensor()
rho0se = qops.get_init_rho(('I', 'I'))
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

# ========================= Non-Markovianity Test =========================
# Trace out the tensor using A0
A0_thetas = 1 * np.pi * np.linspace(0.01, 1.0, 25, endpoint=False)
# A0_thetas = 1 * np.pi * np.array([0.25])
entropies_cal = []
entropies_fit = []
for theta in A0_thetas:
    options = {
        'ftol': 1E-10,
        'maxiter': 1000,
        'method': 'SLSQP',  # SLSQP
        'real': True
    }
    print("======= theta = {} =======".format(theta / np.pi))
    A_N_s = PT_cal.N * [None]
    A_N_s[0] = qmath.rot_xy(theta, 0) if basis == 'cb' else 'X'
    A_N_s[0] = (theta, 0) if basis == 'pm' else A_N_s[0]
    # A_N_s[0] = qmath.rot_xy(theta, 0)
    PT_1p_cal = ptf.ProcessTensor(T_choi=PT_cal.trace(A_N_s))
    PT_1p_fit = ptf.PTensorPM(T_choi=PT_fit.trace(A_N_s, options=None))

    PT1_prod_cal = ptf.ProcessTensor(T_choi=PT_1p_cal.choi_to_product_state())
    PT1_prod_fit = ptf.PTensorPM(T_choi=PT_1p_fit.choi_to_product_state(
        options=None))

    rho1_avg = PT_1p_cal.trace_env(
        PT_1p_cal.rhose_out_ideal(rho0se, A_N_s, Us_ops[:1]))
    rho1_trace_avg = PT_1p_fit.trace(['I'], out_idx=0)
    assert np.allclose(rho1_avg, rho1_trace_avg)
    NM = PT_1p_cal.non_markovianity()
    entropies_cal.append(NM)
    options.update({'ftol': 1E-10,'maxiter': 100})
    NM = PT_1p_fit.non_markovianity(T_choi_ref=PT_1p_cal.T_choi, options=options)  #
    entropies_fit.append(NM)
    print('shrink factor ', 1 / np.cos(theta / 2)**2)
    # P1 = PT1_prod_cal.T_choi
    # P2 = PT1_prod_fit.T_choi
    # print(np.linalg.norm(P1 - P2))

    # A_N_s[1] = 'Pz+'
    # P1 = PT1_prod_fit.predict(A_N_s, method='choi')
    # P2 = PT_1p_fit.predict(A_N_s, method='choi')
    # print(np.linalg.norm(P1 - P2))

xs = A0_thetas / np.pi
ax = plt.figure().add_subplot()
ax.plot(xs, np.array(entropies_cal).real, '-', label='N_cal real')
ax.plot(xs, np.array(entropies_cal).imag, '-.', label='N_cal imag')
ax.plot(xs, np.array(entropies_fit).real, '--', label='N_fit real')
ax.plot(xs, np.array(entropies_fit).imag, '-.', label='N_fit imag')
ax.plot([0.5, 0.5], [0, np.max(entropies_cal)], 'k')
ax.legend()
plt.show()
