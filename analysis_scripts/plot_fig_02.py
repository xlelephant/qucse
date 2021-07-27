import labrad
import itertools
import numpy as np
import matplotlib.pyplot as plt
from qucse.util.registry_editor import load_user_sample
from qucse.util.labrad_legacy.dataset_wrapper import DataVaultWrapper
from qucse_xl import qmath, qops
from qucse_xl.tomography import qst, qpt, ptf, xrt_ptf
from qucse_xl.show.matrix import show_chi
from qucse.analysis import readout

# cxn = labrad.connect()
Sample = load_user_sample(cxn, 'xiang')

dvw = DataVaultWrapper(Sample)
np.set_printoptions(precision=6, suppress=True, linewidth=1000)

# FIG 2.b

pm_idx_init = 1040  # 1514
pm_idx_els = 20
select_pm = 'Py-'
pm_ops = ptf.basis_ops('pm')
pm_op_els = len(pm_ops)
fids = np.zeros((pm_op_els, pm_idx_els), dtype=complex)
chis = np.zeros((pm_op_els, pm_idx_els, 4, 4), dtype=complex)
for x, pm_op_x in enumerate(pm_ops):
    for idx in np.arange(0, pm_idx_els):
        data = dvw[pm_idx_init + x + idx * pm_op_els]
        # tomo_ops = eval(data.parameters['arg: tomo_ops'])
        # pm_ss = eval(data.parameters['arg: pm_ss'])
        tomo_ops = data.parameters['arg: tomo_ops']
        pm_ss = data.parameters['arg: pm_ss']
        inits_op = eval(data.parameters['arg: inits_op'])
        correct = eval(data.parameters['arg: correct'])
        pm_op = data.parameters['arg: pm_op']
        assert pm_op == pm_op_x[0], "{} != {}".format(pm_op, pm_op_x)
        assert correct == False
        tomo_els = len(tomo_ops)
        qst_basis = 'octomo'
        pm_0s = data[:, 1::3 * len(pm_ss)]
        N = 1
        probs = [
            data[:, 2 + i::3 * len(pm_ss)].reshape(-1) for i in range(2**N)
        ]
        probs = np.vstack(probs).T.reshape(-1, tomo_els, 2**N)

        rho0_se = qops.get_init_rho(inits_op)
        rhos_se = qops.get_init_rho([(op, 'I') for op in tomo_ops],
                                    rho0=rho0_se)
        rhos_in = [ptf.ProcessTensor().trace_env(r_se) for r_se in rhos_se]
        F = readout.fidelity_matrix(qubits, measure) if correct else None
        rhos_out = [
            qst.qst_mle(prob, qst_basis, F, disp=False) for prob in probs
        ]
        rhos_prob = [np.average(p_0s) for p_0s in pm_0s]
        rhos_out = [p * rho for p, rho in zip(rhos_prob, rhos_out)]
        chi_expr = qpt.qpt(rhos_in, rhos_out)

        A_ideal = qops.get_op(pm_op)
        chi_ideal, _, _ = qpt.gen_ideal_chi_matrix(A_ideal, None, tomo_ops)
        fidelity = (np.trace(np.dot(chi_ideal, chi_expr)) /
                    np.trace(np.dot(chi_ideal, chi_ideal.T.conj()))).real
        fids[x, idx] = fidelity
        chis[x, idx] = chi_expr
        print('Fidelity {} No. {} is: {}'.format(pm_op, idx, fidelity))
        if select_pm == pm_op:
            sel_idx = x

# FIG 2.c
select_NN = 16  # 4
gate2_idx_init = 1220  # 1313
gate2_idx_els = 20  # 21
fids = np.zeros((2, gate2_idx_els), dtype=complex)
chis = np.zeros((2, gate2_idx_els, select_NN, select_NN), dtype=complex)
for x in range(2):
    for idx in np.arange(0, gate2_idx_els):
        data = dvw[gate2_idx_init + x + idx * 2]
        # experimental chi
        correct = eval(data.parameters['arg: correct'])
        environment = eval(data.parameters['arg: environment'])
        measure = eval(data.parameters['arg: measure'])
        inits_op = eval(data.parameters['arg: inits_op'])
        uc_se_op = eval(data.parameters['arg: uc_se_op'])
        Qubits = [data.parameters[q] for q in ['q4', 'q3']]
        assert correct == True
        dset_els = len(data[0, 1:])
        tomo_els = len(qst.TOMO_BASIS_OPS['octomo'])
        N = int(np.log2(dset_els) / np.log2(2 * tomo_els))
        tomo_ops = tomo.op_products(qst.TOMO_BASIS_OPS['octomo'], N)
        rho0_0 = qops.get_init_rho(inits_op)
        rhos_in = [qops.get_init_rho(tomo_op, rho0_0) for tomo_op in tomo_ops]
        F = readout.fidelity_matrix(Qubits, measure) if correct else None
        rhos_out = []
        probs = data[:, 1:].reshape(-1, tomo_els**N, 2**N)
        for prob in probs:
            rhos_out.append(
                qst.qst_mle(prob, qops.get_ops(tomo_ops), F, disp=False))
        chi_expr = qpt.qpt(rhos_in, rhos_out)

        # ideal chi
        U = qops.get_op(uc_se_op)
        chi_ideal = qpt.gen_ideal_chi_matrix(U, tomo_ops, rho0=rho0_0)

        # compare
        fidelity = qmath.fidelity_chi(chi_expr, chi_ideal)
        fids[x, idx] = fidelity
        chis[x, idx] = chi_expr
        print('Fidelity = {:.4f}'.format(fidelity))

# FIG 2.d
select_NN = 4
gate2_idx_inits = [1313, 1314, 1454]
skip_els = [2, 2, 1]
gate2_idx_els = 20
panel_els = len(skip_els)
fids = np.zeros((panel_els, gate2_idx_els), dtype=complex)
chis_expr = np.zeros((panel_els, gate2_idx_els, select_NN, select_NN),
                     dtype=complex)
chis_ideal = np.zeros((panel_els, gate2_idx_els, select_NN, select_NN),
                      dtype=complex)
for x in range(panel_els):
    for idx in np.arange(0, gate2_idx_els):
        data = dvw[gate2_idx_inits[x] + idx * skip_els[x]]
        # experimental chi
        correct = eval(data.parameters['arg: correct'])
        environment = eval(data.parameters['arg: environment'])
        assert environment is not None
        measure = eval(data.parameters['arg: measure'])
        inits_op = eval(data.parameters['arg: inits_op'])
        Qubits = [data.parameters[q] for q in ['q4', 'q3']]
        assert correct == True
        N = 1
        tomo_ops = qst.TOMO_BASIS_OPS['octomo']
        tomo_els = len(qst.TOMO_BASIS_OPS['octomo'])
        rho0_0 = qops.get_init_rho(inits_op)
        F = readout.fidelity_matrix(Qubits, measure) if correct else None

        if x == 2:
            Bs_ops = eval(data.parameters['arg: Bs_ops'])
            Us_ops = eval(data.parameters['arg: Us_ops'])
            As = eval(data.parameters['arg: As'])
            chi_expr = xrt_ptf.get_conditional_map(data, Bs_ops, F)
            # ideal chi
            rho1_0 = qops.get_init_rho(list(As) + ['I'], rho0_0)
            rho1_0 = qops.get_init_rho(Us_ops[0], rho1_0)
            U = qops.get_op(Us_ops[1])
            chi_ideal = qpt.gen_ideal_chi_matrix(
                U, qst.TOMO_BASIS_OPS['pm_octomo'], rho1_0)
        else:
            uc_se_op = eval(data.parameters['arg: uc_se_op'])
            rho0_s = ptf.ProcessTensor().trace_env(rho0_0)
            rhos_in = [
                qops.get_init_rho(tomo_op, rho0_s) for tomo_op in tomo_ops
            ]
            rhos_out = []
            probs = data[:, 1:].reshape(-1, tomo_els**N, 2**N)
            for prob in probs:
                rhos_out.append(
                    qst.qst_mle(prob, qops.get_ops(tomo_ops), F, disp=False))
            chi_expr = qpt.qpt(rhos_in, rhos_out)
            # ideal chi
            U = qops.get_op(uc_se_op)
            chi_ideal = qpt.gen_ideal_chi_matrix(U, tomo_ops, rho0_0)

        # compare
        fidelity = qmath.fidelity_chi(chi_expr, chi_ideal, normalize=True)
        fids[x, idx] = fidelity
        chis_expr[x, idx] = chi_expr
        chis_ideal[x, idx] = chi_ideal
        print('Fidelity = {:.4f}'.format(fidelity))

# select_good = [[0,1,2,3,4,5,11,12,13,14,15,16]]
fid_avg = np.average(fids, axis=1)
fid_std = np.std(fids, axis=1)
xs = range(len(fids))
ax = plt.figure(tight_layout=True).add_subplot()
ax.plot(xs, fid_avg, '*')
ax.errorbar(xs, fid_avg, yerr=fid_std, fmt='-o', color='k')
ax.set_ylim([0.9, 1.01])
ax.set_xticks(xs)
# ax.set_xticklabels(ptf.ops2str(pm_ops))
ax.set_title('Fidelity of Chi')

sel_idx = 0
op_str = 'CZ(I)'  # select_pm
fidelity = fid_avg[sel_idx].real
std_var = fid_std[sel_idx].real
chi_expr = np.average(chis_expr[sel_idx, ...], axis=0)
chi_ideal = np.average(chis_ideal[sel_idx, ...], axis=0)
fig, ax = None, None
title = r'Ideal $\chi$({})'.format(op_str)
qpt_labels = list(itertools.product(qpt.pauli_vector_ops, repeat=N))
fig, ax = show_chi(chi_ideal, qpt_labels, alpha=0.2, title=title, fig=fig,
                   ax=ax, threshold=0.0)
# fig, ax = None, None
title = r'Fidelity of $\chi$({}) = {:1.2f}$\pm${:1.2f}%'.format(
    op_str, fidelity * 100, std_var * 100)
fig, ax = show_chi(chi_expr, qpt_labels, title=title, fig=fig, ax=ax,
                   threshold=0.1)
