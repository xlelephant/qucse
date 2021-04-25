import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import colorbar as mcolorbar


def complex_phase_cmap(name='twilight'):
    # https://matplotlib.org/3.3.3/gallery/color/colormap_reference.html
    if name == 'phase_colormap':
        cdict = {
            'blue': ((0.00, 0.0, 0.0), (0.25, 0.0, 0.0), (0.50, 1.0, 1.0),
                     (0.75, 1.0, 1.0), (1.00, 0.0, 0.0)),
            'green': ((0.00, 0.0, 0.0), (0.25, 1.0, 1.0), (0.50, 0.0, 0.0),
                      (0.75, 1.0, 1.0), (1.00, 0.0, 0.0)),
            'red': ((0.00, 1.0, 1.0), (0.25, 0.5, 0.5), (0.50, 0.0, 0.0),
                    (0.75, 0.0, 0.0), (1.00, 1.0, 1.0))
        }
        cmap = mcolors.LinearSegmentedColormap(name, cdict, 256)
    else:
        cmap = plt.get_cmap(name)
    return cmap


def bar3d_complex(matrix, xlabels=None, ylabels=None, title=None, fig=None,
                  ax=None, colorbar=True, z_limits=None, phase_limits=None,
                  threshold=None, alpha=None):
    n = np.size(matrix)
    bar_width = 0.618
    xpos, ypos = np.meshgrid(range(matrix.shape[0]), range(matrix.shape[1]))
    xpos = xpos.T.flatten() - 0.5 * bar_width
    ypos = ypos.T.flatten() - 0.5 * bar_width
    zpos = np.zeros(n)

    dx = dy = bar_width * np.ones(n)
    Mvec = matrix.flatten()
    dz = abs(Mvec)

    if phase_limits:
        phase_min = phase_limits[0]
        phase_max = phase_limits[1]
    else:
        phase_min = -np.pi
        phase_max = np.pi
    norm = mcolors.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()

    # check small values and set them to negative number (white)
    # idx, = np.where(abs(Mvec) < 0.01)
    # Mvec[idx] = abs(Mvec[idx])
    colors = cmap(norm(np.angle(Mvec)))

    if threshold is not None:
        colors[:, 3] = 1 * (dz > threshold)
        colors[:, 3] = 0 * (dz <= threshold)

    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig, azim=-35, elev=35)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=alpha,
             shade=False, linewidth=1.5, edgecolor='k')

    if title:
        ax.set_title(title)

    if xlabels:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelsize=10)

    if ylabels:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
    ax.tick_params(axis='y', labelsize=10)
    # ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    # ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, -0.5))

    if z_limits and isinstance(z_limits, list):
        ax.set_zlim3d(z_limits)
    else:
        ax.set_zlim3d([0, 1])

    if colorbar:
        cax, kw = mcolorbar.make_axes(ax, shrink=.75, orientation='vertical')
        cb = mcolorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb.set_ticklabels(
            (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
        cb.set_label('arg')

    return fig, ax


def show_chi(chi, labels, title=None, fig=None, ax=None, figsize=(6, 6),
             alpha=0.9, colorbar=True, threshold=None):

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection='3d', position=[0, 0, 1, 1])
    else:
        colorbar = False

    title = r"$\chi$" if not title else title
    ylabels = xlabels = ["".join(lbl) for lbl in labels]
    return bar3d_complex(chi, xlabels, ylabels, title=title, ax=ax,
                         alpha=alpha, colorbar=colorbar, threshold=threshold)
