import numpy as np
import matplotlib.pyplot as plt

from qucse_xl import qmath, qops
from qucse_xl.tomography import qst, qpt, ptf, psd
from qucse_xl.show.matrix import show_chi

import labrad
import numpy as np
import matplotlib.pyplot as plt
from qucse.util.registry_editor import load_user_sample
from qucse.util.labrad_legacy.dataset_wrapper import DataVaultWrapper

from qucse_xl.tomography import qpt, qst, ptf, xrt_ptf
from qucse_xl import qops, qmath

cxn = labrad.connect()
Sample = load_user_sample(cxn, 'xiang')

dvw = DataVaultWrapper(Sample)

# FIG 3
ptf_idx_init = 1368
ptf_idx_els = 20
pt_set_num = np.concatenate([np.arange(53, 53 + 3),
                             np.arange(0, 17)])[:ptf_idx_els]

Bss_ops = None
Ass_ops = None
Us_ops = None
inits_op = None
noisy = False
pt_sel = 0
fids_se_qpt = []
fids_s_qpt = []
fids_ptf = []
ptensors_fit = []
for idx, d_num in enumerate(pt_set_num):
    data_num = ptf_idx_init + pt_sel + d_num * 2
    print("dataset idx :", data_num, end='')
    data = dvw[data_num]
    if Bss_ops is None:
        Bss_ops = eval(data.parameters['arg: Bss_ops'])
        Ass_ops = eval(data.parameters['arg: Ass_ops'])
        Us_ops = eval(data.parameters['arg: Us_ops'])
        inits_op = eval(data.parameters['arg: inits_op'])
    correct = eval(data.parameters['arg: correct'])
    assert correct == True
    q0 = data.parameters['q4']
    fids = q0['calReadoutFids'] if correct else None

    # data
    # -> rho, prob
    B_els = len(Bss_ops)
    All_ops = Bss_ops + Ass_ops
    rhos, ps = xrt_ptf.get_rhos_from_dataset(data, fids=fids)
    # -> process tensor
    psd_options = {
        # 'gtol': 1E-6,
        'ftol': 1E-8,
        'maxiter': 100,
        'method': 'SLSQP',  # SLSQP
        'real': False
    }
    psd_options = None
    pt_fit = xrt_ptf.get_process_tensor(All_ops, rhos, ps, options=psd_options)
    ptensors_fit.append(pt_fit)

    # theoretical methods
    rho0_se = qops.get_init_rho(inits_op)
    Us = qops.get_ops(Us_ops)

    PT_cal = ptf.ProcessTensor()
    PT_cal.T_choi = PT_cal.cal(rho0_se, Us)

    chi_se = [np.array(q0['Chi_' + '*'.join(s)]) for s in Us_ops]
    chi_ss = [np.array(q0['Chi_TrE_' + '*'.join(s) + '(I*I)']) for s in Us_ops]

    # plot

    rss_exp, p_exp = [], []
    rse_sim, rss_sim, rse_chi, rss_chi, rss_ptf = [], [], [], [], []
    Bs_sticks, As_sticks = [], []
    for i, As in enumerate(All_ops):
        # ------------------ Predictions using Simulation -----------------
        rho_ise = PT_cal.rhose_out_ideal(rho0_se, As, Us)
        if (abs(rho_ise) < xrt_ptf.HRD_TH).all():
            if noisy:
                print('No.{} As {} is too small to be added!'.format(i, As))
            continue
        if i < B_els:
            Bs_sticks.append(As)
        else:
            As_sticks.append(As)
        rse_sim.append(PT_cal.trace_env(rho_ise))

        # simulate the markovian process of S for reference
        chis_id = [
            qpt.gen_ideal_chi_matrix(U, As=qst.TOMO_BASIS_OPS['pm_full'],
                                     rho0=rho0_se, zero_th=ptf.ZERO_RHO_TH,
                                     noisy=False) for U in Us
        ]
        rss_sim.append(PT_cal.predict(As, chis_id, rho0_se, method='chi'))

        # ------- predictions using fitted process map (linear QPT) -------
        rse_chi.append(pt_fit.predict(As, chi_se, rho0_se, method='chi'))
        rss_chi.append(pt_fit.predict(As, chi_ss, rho0_se, method='chi'))

        # ------------ predictions using fitted process tensor ------------
        rss_ptf.append(pt_fit.predict(As, method='matrix'))

        rss_exp.append(rhos[i])
        p_exp.append(ps[i])

    f_mkv, f_se_i, f_s_i, f_se_e, f_s_e, f_pt = [], [], [], [], [], []
    for _, r0, s2, s1, c2, c1, rpt in zip(range(len(All_ops)), rss_exp,
                                          rse_sim, rss_sim, rse_chi, rss_chi,
                                          rss_ptf):
        warn = True
        imtol = 1E-1 if psd_options is not None else 0.7
        f_mkv.append(qmath.fidelity_rho(s1, s2, noisy=warn))
        # f_se_i.append(qmath.fidelity_rho(s2_i, r0, imag_atol=imtol, noisy=warn))
        # f_s_i.append(qmath.fidelity_rho(s1, r0, imag_atol=imtol, noisy=warn))
        f_se_e.append(qmath.fidelity_rho(c2, r0, imag_atol=imtol, noisy=False))
        f_s_e.append(qmath.fidelity_rho(c1, r0, imag_atol=imtol, noisy=False))
        f_pt.append(qmath.fidelity_rho(rpt, r0, imag_atol=imtol, noisy=warn))

    fids_se_qpt.append(f_se_e)
    fids_s_qpt.append(f_s_e)
    fids_ptf.append(f_pt)

    print(" average fids for s_qpt, se_qpt, ptf are: ", np.average(f_s_e),
          np.average(f_se_e), np.average(f_pt))

fid_se_avg = np.average(fids_se_qpt, axis=1)
fid_s_avg = np.average(fids_s_qpt, axis=1)
fid_ptf_avg = np.average(fids_ptf, axis=1)

f_se_qpt = np.average(fids_se_qpt, axis=0)
f_s_qpt = np.average(fids_s_qpt, axis=0)
f_ptf = np.average(fids_ptf, axis=0)

stds_s_chi = [np.std(fs_op) for fs_op in np.transpose(fids_s_qpt)]
stds_se_chi = [np.std(fs_op) for fs_op in np.transpose(fids_se_qpt)]
stds_ptf = [np.std(fs_op) for fs_op in np.transpose(fids_ptf)]

print('fids, std qpt(s): ', np.average(fids_s_qpt), np.average(stds_s_chi))
print('fids, std qpt(se): ', np.average(fids_se_qpt), np.average(stds_se_chi))
print('fids, std of ptf: ', np.average(fids_ptf), np.average(stds_ptf))

# show
bar_width = 0.3
ax = plt.figure(tight_layout=True).add_subplot()
ax.set_ylim([0.4, 1.05])
sel_els = len(Bs_sticks)
xs = range(sel_els)
ax.set_xticks(xs)
ax.set_xticklabels(ptf.ops2str(Bs_sticks))
line = ax.plot(xs, f_mkv[:sel_els], 'k', ds='steps-mid', lw=2,
               label='markov sim')
line = ax.bar(xs, f_s_qpt[:sel_els], bar_width, fc='r', ec='None', lw=2,
              alpha=0.3, label='qpt s')
line = ax.bar(xs, (np.array(f_se_qpt) - np.array(f_s_qpt))[:sel_els],
              bar_width, f_s_qpt[:sel_els], fc='None', ec='r', lw=2, alpha=0.8,
              label='qpt se - s')
# line = ax.plot(xs, f_ptf[:sel_els], '-.b', ds='steps-mid', lw=2, label='ptf')
line = ax.errorbar(xs, f_ptf[:sel_els], yerr=stds_ptf[:sel_els], fmt='-.b',
                   color='b')
ax.legend()

# ==================== Non-Markovianity calculation ====================
# np.set_printoptions(precision=6, suppress=True, linewidth=1000)

num_steps = 2
Uss_ops = [
    ["CZ", "CNOT", "iSwap"],
    ["CNOT", "CZ", "iSwap"],
]
Uops = Uss_ops[pt_sel]
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
A0_thetas = 1 * np.pi * np.linspace(0.01, 1.0, 50, endpoint=False)
# A0_thetas = 1 * np.pi * np.array([0.25])
entropies_cal = []
entropies_fit = []
xs = []
for theta in A0_thetas[::-1]:
    x_val = theta / np.pi
    xs.append(x_val)
    print("===================== theta: {} ====================".format(x_val))
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
        'method': None,  # SLSQP
        'real': False
    }
    # PT_1p_fit = ptf.PTensorPM(T_choi=PT_fit.contract(A_N_s, options=None))
    PT_1p_fit = ptf.PTensorPM(T_choi=PT_fit.contract(A_N_s))
    PT1f_prod = ptf.ProcessTensor(T_choi=PT_1p_fit.choi_to_product_state())
    print('tr PT fit are: ', PT_1p_fit.trace(), PT1f_prod.trace())

    # PT1f_norm = ptf.PTensorPM(PT_1p_fit.normalize(factor=PT1f_prod.trace()))
    # PT1f_prod_norm = ptf.ProcessTensor(T_choi=PT1f_prod.normalize())
    NM, _ = PT_1p_fit.non_markovianity(T_markov=PT1f_prod.T_choi,
                                       T_guess=PT1f_prod.T_choi,
                                       options=options)

    entropies_fit.append(NM / PT1f_prod.trace())

# ======================= experimental data ========================

entropies_exps = []
exp_lsq_errs = []
for PT_exp in ptensors_fit:  # [PT_cal]
    entropies_exp = []
    exp_lsq_err = []
    xs = []
    for theta in A0_thetas[::-1]:
        x_val = theta / np.pi
        xs.append(x_val)
        print("===================== theta: {} ====================".format(
            x_val))
        A_N_s = PT_cal.N * [None]
        A_N_s[0] = qmath.rot_xy(theta, 0) if basis == 'cb' else 'X'
        A_N_s[0] = (theta, 0) if basis == 'pm' else A_N_s[0]
        # A_N_s[0] = qmath.rot_xy(theta, 0)
        t_norm = 1 * 2
        pm_probs = np.cos(theta / 2)**2
        print('prob of A0x2 is ', pm_probs * 2)

        # new fitting to ps-matrix
        options = {
            'ftol': 1E-8,
            'maxiter': 200,
            'method': None,  # SLSQP
            'real': False
        }
        # ================ PT_fit (expr) non-markovianity ================
        print("~~~~~~~~~~~~~ Get the nm of experimental data ~~~~~~~~~~~~~")
        PT_1p_exp = ptf.PTensorPM(T_choi=PT_exp.contract(A_N_s))
        PT1e_prod = ptf.ProcessTensor(T_choi=PT_1p_exp.choi_to_product_state())
        print('tr PT exp are: ', PT_1p_exp.trace(), PT1e_prod.trace())

        # PT1e_norm = ptf.PTensorPM(PT_1p_exp.normalize(factor=PT1e_prod.trace()))
        # PT1e_prod_norm = ptf.ProcessTensor(T_choi=PT1e_prod.normalize())
        NM, lsq_err = PT_1p_exp.non_markovianity(T_markov=PT1e_prod.T_choi,
                                                 T_guess=PT1e_prod.T_choi,
                                                 options=options)

        entropies_exp.append(NM / PT1e_prod.trace())
        exp_lsq_err.append(lsq_err)

    entropies_exps.append(entropies_exp)
    exp_lsq_errs.append(exp_lsq_err)

ax = plt.figure().add_subplot()

entropy_avg = np.average(entropies_exps, axis=0)
lsq_err_avg = np.average(exp_lsq_errs, axis=0)
std_etp_exp = [np.std(etp_exps) for etp_exps in np.transpose(entropies_exps)]

ax.plot(xs, np.array(entropies_cal).real, '-', label='N_cal')
ax.plot(xs, np.array(entropies_fit).real, '--', label='N_fit')
ax.plot(xs, np.array(entropy_avg).real, '--', label='N_exp real')
ax.plot(xs, np.array(entropy_avg).imag, '-.', label='N_exp imag')
ax.errorbar(xs,
            np.array(entropy_avg).real, yerr=std_etp_exp, fmt='-.b', color='b')
ax.plot([0.5, 0.5], [0, np.max(entropies_cal)], 'k')
ax.plot(0, 1, [np.log(2)] * 2, '-.k')
ax.set_title('Non-markovianity of process {}'.format(Us_ops))
ax.legend()
plt.show()
