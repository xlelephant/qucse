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

pt_sel = 1
Bss_ops = ptf.basis_ops(group='pm', steps=2, complement=False)
Ass_ops = ptf.basis_ops(group='pm', steps=2, complement=True)
if pt_sel == 0:
    Us_ops = (("CZ",), ("CNOT",))
elif pt_sel == 1:
    Us_ops = (("CNOT",),("CZ",))
inits_op = ('I', 'I')
noisy = False

B_els = len(Bss_ops)
All_ops = Bss_ops + Ass_ops

# theoretical methods
rho0_se = qops.get_init_rho(inits_op)
Us = qops.get_ops(Us_ops)

PT_cal = ptf.ProcessTensor()
PT_cal.T_choi = PT_cal.cal(rho0_se, Us)

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

    # simulate the Markovian process of S for reference

    chis_id = [
        qpt.gen_ideal_chi_matrix(U, As=qst.TOMO_BASIS_OPS['pm_octomo'],
                                 rho0=rho0_se, zero_th=ptf.ZERO_RHO_TH,
                                 noisy=False) for U in Us
    ]
    # rhose1 = ptf.ProcessTensor().rhose_out_ideal(rho0_se, As[:1], Us[:1])
    rss_sim.append(PT_cal.predict(As, chis_id, rho0_se, method='chi'))



f_mkv = []
for _, s2, s1, in zip(range(len(All_ops)), rse_sim, rss_sim):
    imtol = 1E-1
    warn = True
    f_mkv.append(qmath.fidelity_rho(s1, s2, noisy=warn))

if pt_sel == 0:
    fids_se_qpt = np.loadtxt("./qucse_xl/analysis_scripts/ptf_data/fids_(cz-cnot)_se_qpt.csv")
    fids_s_qpt = np.loadtxt("./qucse_xl/analysis_scripts/ptf_data/fids_(cz-cnot)_s_qpt.csv")
    fids_ptf = np.loadtxt("./qucse_xl/analysis_scripts/ptf_data/fids_(cz-cnot)_ptf.csv")
elif pt_sel == 1:
    fids_se_qpt = np.loadtxt("./qucse_xl/analysis_scripts/ptf_data/fids_(cnot-cz)_se_qpt.csv")
    fids_s_qpt = np.loadtxt("./qucse_xl/analysis_scripts/ptf_data/fids_(cnot-cz)_s_qpt.csv")
    fids_ptf = np.loadtxt("./qucse_xl/analysis_scripts/ptf_data/fids_(cnot-cz)_ptf.csv")
# fid_se_avg = np.average(fids_se_qpt, axis=1)
# fid_s_avg = np.average(fids_s_qpt, axis=1)
# fid_ptf_avg = np.average(fids_ptf, axis=1)

# f_se_qpt = np.average(fids_se_qpt, axis=0)
f_s_qpt = np.average(fids_s_qpt, axis=0)
f_ptf = np.average(fids_ptf, axis=0)

stds_s_qpt = np.array([np.std(fs_op) for fs_op in np.transpose(fids_s_qpt)])
stds_se_qpt = np.array([np.std(fs_op) for fs_op in np.transpose(fids_se_qpt)])
stds_ptf = np.array([np.std(fs_op) for fs_op in np.transpose(fids_ptf)])

print('fids, std qpt(s): ', np.average(fids_s_qpt), np.average(stds_s_qpt))
# print('fids, std qpt(se): ', np.average(fids_se_qpt), np.average(stds_se_qpt))
print('fids, std of ptf: ', np.average(fids_ptf), np.average(stds_ptf))
print('fids, markov:', np.average(f_mkv))


# show
ax = plt.figure(tight_layout=True).add_subplot()
ax.set_ylim([0.42, 1.08])
All_sticks = Bs_sticks + As_sticks
sel_els = len(All_sticks)
offset = 53 # 53 (xz) # 62 (yz) # 215 (zx) 233 (zy)
offset = 9 * 4 - 1
offset = 0
offset = 9 * 3
idx_xp = slice(0 + offset,8 + offset)
idx_sel = idx_xp
xs = range(sel_els)[idx_sel]
line = ax.plot(xs, f_mkv[idx_sel], 'k', ds='steps-mid', lw=2,
               label='markov sim')
# line = ax.scatter(xs, f_s_qpt[idx_sel], '-r', lw=1, label='qpt s')
# line = ax.errorbar(xs, f_s_qpt[idx_sel], yerr=stds_s_qpt[idx_sel], fmt='.r',
#                    marker='o', color='r')
line = ax.violinplot(fids_s_qpt[:,idx_sel], list(xs),
                  showmeans=True,
                  showmedians=False)
# line = ax.boxplot(fids_s_qpt[:,idx_sel], positions=list(xs), whis=2.0)

line = ax.plot(xs, f_ptf[idx_sel], '-')
line = ax.scatter(xs, f_ptf[idx_sel], s=18, c='b', label='ptf')
# line = ax.errorbar(xs, f_ptf[idx_sel], yerr=100*stds_ptf[idx_sel], fmt='-.b',
#                    marker='o')
# line = ax.boxplot(fids_ptf[:,idx_sel], positions=list(xs), whis=1000000.0)
for x in xs:
    line = ax.text(x-0.5, 1.02, s='{:.4f}'.format(f_ptf[x]))
ax.set_xticks(xs)
ax.set_xticklabels(ptf.ops2str(All_sticks[idx_sel]))
ax.legend()
