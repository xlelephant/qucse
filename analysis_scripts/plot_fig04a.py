import numpy as np
import matplotlib.pyplot as plt
from qucse_xl import qmath, qops
from qucse_xl.tomography import qpt, ptf
from qucse.analysis.plotting import tomography
from importlib import reload
import mpl_toolkits.mplot3d as plt3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors
from matplotlib import colorbar as mcolorbar
from matplotlib import cm

U2_1 = qops.get_ops(['CNOT', 'CZ'])
rhos2s = []
theta_colors = []
rhos2chi = []
distances = []
A0s = ['Pz+', 'Pzy+', 'Py+']
A0 = 'Py-'
A0 = (np.pi / 2 + 0.005, 0)
ax = None
draw_surface = True
if draw_surface:
    phi_els = 33
    theta_els = 65
    reshape_size = (phi_els, theta_els)
else:
    phi_els = 1
    theta_els = 65
for phi in np.linspace(0, 2 * np.pi, phi_els):
    for theta in np.linspace(0, 1 * np.pi, theta_els):
        As = qops.get_ops([A0, (theta, phi)])  # (np.pi/2+0.001, 0)
        rho0 = qops.get_init_rho(('I', 'I'))
        rho1 = ptf.ProcessTensor().rhose_out_ideal(rho0, As[:1], U2_1[:1])
        Chi_cz_s = qpt.gen_ideal_chi_matrix(U2_1[1], ptf.QPT_PMs, rho0=rho1,
                                            zero_th=0)
        rhoA1 = qops.get_op((theta, phi))
        rho2_chi = qpt.cal_process_rho(rhoA1, Chi_cz_s)
        theta_colors.append(theta)
        rhos2chi.append(rho2_chi)

        rho2 = ptf.ProcessTensor().rhose_out_ideal(rho0, As, U2_1)
        rhos2 = qmath.unit_trace(ptf.ProcessTensor().trace_env(rho2))
        rhos2s.append(rhos2)
        distances.append(qmath.trace_distance(rhos2, rho2_chi))
if draw_surface:
    bloch_alpha = 0.1
    line_alpha = 0.8
    line_width = 1.0
    surface_alpha = 0.7
    ax = plt3.Axes3D(plt.figure(figsize=(6, 6)))
    x, y, z = np.array([qmath.rho2bloch(rho) for rho in rhos2s]).T
    x = np.reshape(x, reshape_size)
    y = np.reshape(y, reshape_size)
    z = np.reshape(z, reshape_size)
    cmap = plt.get_cmap('rainbow')
    norm = mcolors.Normalize(0, np.pi)
    rvs_normed = norm(np.pi - np.array(theta_colors))
    colors = cmap(np.reshape(rvs_normed, reshape_size))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors,
                    alpha=surface_alpha, linewidth=0.0)
    ax = tomography.drawBlochSphere(alpha=bloch_alpha, alpha_l=line_alpha,
                                    ax=ax, linewidth=line_width, labels=False)

    ax = plt3.Axes3D(plt.figure(figsize=(6, 6)))
    x, y, z = np.array([qmath.rho2bloch(rho) for rho in rhos2chi]).T
    x = np.reshape(x, reshape_size)
    y = np.reshape(y, reshape_size)
    z = np.reshape(z, reshape_size)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors,
                    alpha=surface_alpha, linewidth=0.0)
    ax = tomography.drawBlochSphere(alpha=bloch_alpha, alpha_l=line_alpha,
                                    ax=ax, linewidth=line_width, labels=False)

    # cax, kw = mcolorbar.make_axes(ax, shrink=.75, orientation='vertical')
    # cb = mcolorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    # cb.set_ticks([0, np.pi / 2, np.pi])
    # cb.set_ticklabels((r'$0$', r'$\pi/2$', r'$\pi$'))
else:
    ax = tomography.plotTrajectory(rhos2s, axm=ax, markersize=0)
    #, view_init=(0,90)
    ax = tomography.plotTrajectory(rhos2chi, axm=ax, markersize=4, state=2,
                                   maker='>')

plt.show()
