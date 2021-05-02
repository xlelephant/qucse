import numpy as np

from . import qst, ptf

HRD_TH = 1E-4


def _get_rhos_uc(dataset, fids, qst_basis):
    N = 1
    tomo_els = len(qst.TOMO_BASIS_OPS[qst_basis])
    data = np.array(dataset)[:, 1:]

    probs = data.reshape(-1, tomo_els, 2**N)
    if fids is not None:
        f00, f11 = fids
        F = np.array([[f00, 1 - f00], [1 - f11, f11]])
        rhos = [qst.qst_mle(prob, qst_basis, F, disp=False) for prob in probs]
    else:
        rhos = [qst.qst(prob, qst_basis) for prob in probs]
    return rhos


def _get_rhos_pm(dataset, fids, qst_basis):
    N = 1
    tomo_els = len(qst.TOMO_BASIS_OPS[qst_basis])
    data = np.array(dataset)[:, 1:]

    p01_els = 3
    steps = int(np.log2(len(data[0, :]) / tomo_els / p01_els))
    ps_els = int(2**steps)
    ps_0s = data[:, 0::p01_els * ps_els]
    # average over all tomos of As
    ps_rs = [np.average(p_0s) for p_0s in ps_0s]

    # rho out probs of each basis
    probs = [data[:, 1 + i::p01_els * ps_els].reshape(-1) for i in range(2**N)]
    probs = np.vstack(probs).T.reshape(-1, tomo_els, 2**N)
    if fids is not None:
        f00, f11 = fids
        F = np.array([[f00, 1 - f00], [1 - f11, f11]])
        rhos = [qst.qst_mle(prob, qst_basis, F, disp=False) for prob in probs]
    else:
        rhos = [qst.qst(prob, qst_basis) for prob in probs]
    return rhos, ps_rs


def get_rhos_from_dataset(dataset, fids=None, qst_basis='octomo'):
    if len(dataset[:0]) == 12:
        rhos = _get_rhos_uc(dataset, fids=fids, qst_basis=qst_basis)
        ps = None
    else:
        rhos, ps = _get_rhos_pm(dataset, fids=fids, qst_basis=qst_basis)
    return rhos, ps


def get_process_tensor(Bss_ops, rhos, ps=None, herald_th=HRD_TH, noisy=True,
                       options=None):
    """Experimentally derive the process tensor"""
    valid_idx = []
    if ps is None:
        ptensor = ptf.PTensorUC()
    else:
        ptensor = ptf.PTensorPM()
        for i, p in zip(range(len(Bss_ops)), ps):
            if p < herald_th:
                print("Warning, heralds rate {} < {}, discard op {}.".format(
                    p, herald_th, Bss_ops[i]))
            else:
                valid_idx.append(i)
        rho_out = [ps[i] * rhos[i] for i in valid_idx]
        Bss_ops = np.array(Bss_ops)[valid_idx]
    if options is not None:
        ptensor.T_mat = ptensor.least_square_psd_fit(Bss_ops, rho_out,
                                                     disp=noisy,
                                                     options=options)
    else:
        ptensor.T_mat = ptensor.least_square_fit(Bss_ops, rho_out, disp=noisy)
    return ptensor
