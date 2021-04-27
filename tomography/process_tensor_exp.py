# FIG 3
ptf_idx_init = 1368
ptf_idx_els = 1
pt_set_num = np.concatenate([np.arange(53, 53 + 3),
                             np.arange(0, 17)])[:ptf_idx_els]

Bss_ops = None
Ass_ops = None
Us_ops = None
inits_op = None
noisy = False
pt_sel = 1
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
    psd_options = None
    psd_options = {
        # 'gtol': 1E-6,
        'ftol': 1E-6,
        'maxiter': 500,
        'method': 'SLSQP',  # SLSQP
        'real': True
    }
    psd_options = {
        'gtol': 1E-10,
        'maxiter': 500,
        'method': None,  # SLSQP
        'real': True
    }
    pt_fit = xrt_ptf.get_process_tensor(All_ops, rhos, ps, options=psd_options)
    ptensors_fit.append(pt_fit)

    # theoretical methods
    rho0_se = qops.get_init_rho(inits_op)
    Us = qops.get_ops(Us_ops)

    pt_cal = ptf.ProcessTensor()
    pt_cal.cal(rho0_se, Us)

    # plot

    rss_exp, p_exp = [], []
    rse_sim, rss_sim, rse_chi, rss_chi, rss_ptf = [], [], [], [], []
    Bs_sticks, As_sticks = [], []
    for i, As in enumerate(All_ops):
        # ------------------ Predictions using Simulation -----------------
        rho_ise = pt_cal.rhose_out_ideal(rho0_se, As, Us)
        if (abs(rho_ise) < xrt_ptf.HRD_TH).all():
            if noisy:
                print('No.{} As {} is too small to be added!'.format(i, As))
            continue
        if i < B_els:
            Bs_sticks.append(As)
        else:
            As_sticks.append(As)
        rse_sim.append(pt_cal.trace_env(rho_ise))

        # simulate the markovian process of S for reference
        chis_id = [
            qpt.gen_ideal_chi_matrix(U, As=qst.TOMO_BASIS_OPS['pm_full'],
                                     rho0=rho0_se, zero_th=ptf.ZERO_RHO_TH,
                                     noisy=False) for U in Us
        ]
        rss_sim.append(pt_cal.predict(As, chis_id, rho0_se, method='chi'))

        # ------- predictions using fitted process map (linear QPT) -------
        chi_se = [np.array(q0['Chi_' + '*'.join(s)]) for s in Us_ops]
        chi_ss = [
            np.array(q0['Chi_TrE_' + '*'.join(s) + '(I*I)']) for s in Us_ops
        ]
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
        imtol = 1E-2 if psd_options is not None else 0.7
        warn = True
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

# ========================= Non-Markovianity Test =========================\
T_mat_fit_avg = np.average(np.array([pt_fit.T_mat for pt_fit in ptensors_fit]),
                           axis=0)
pt_fit = ptf.PTensorPM(T_mat=T_mat_fit_avg)
pt_fit.T_choi = pt_fit.matrix_to_choi()

# Trace out the tensor using A0
basis = 'pm'
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

    A_N_s = pt_cal.N * [None]
    A_N_s[0] = A0
    # psd_options.update({'ftol': 1E-16, 'maxiter': 1000})
    T_choi_1p_cal = pt_cal.trace(A_N_s)
    T_choi_1p_fit = pt_fit.trace(A_N_s, options=psd_options)  # fitting
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

    # entropies_cal.append(PT1p_cal.non_markovianity() / np.cos(theta / 2)**2)
    # entropies_fit.append(PT1p_fit.non_markovianity() / np.cos(theta / 2)**2)
    entropies_cal.append(PT1p_cal.non_markovianity())
    entropies_fit.append(PT1p_fit.non_markovianity())  # fitting

fig = plt.figure()
ax = fig.add_subplot()
xs = A0_thetas / np.pi
ax.plot(xs, np.array(entropies_cal).real, '-', label='N_cal real')
ax.plot(xs, np.array(entropies_cal).imag, '-.', label='N_cal imag')
ax.plot(xs, np.array(entropies_fit).real, '--', label='N_fit real')
ax.plot(xs, np.array(entropies_fit).imag, '-.', label='N_fit imag')
ax.plot([0.5, 0.5], [0, np.max(entropies_cal)], 'k')
plt.legend()
