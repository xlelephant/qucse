from numpy import pi

PI_FRAC_DICT = {
    'zero': 0.0,
    'piHalf3': 1.5,
    'pi': 1.0,
    'piHalf': 0.5,
    'piQuad': 0.25,
    'piOct': 0.125,
    'piPent': 0.2,
    'piHex': 1. / 6,
    'piHept': 1. / 7,
    'piNon': 1. / 9,
    'piDec': 0.1,
    'piUndec': 1. / 11,
    'piDodec': 1. / 12,
    'piTridec': 1. / 13,
    'piTetradec': 1. / 14,
    'piPentadec': 1. / 15,
    'piIcosahedral': 1. / 20,
    'piHexadec': 0.0625,
}

FRAC_PI_DICT = {v: k for k, v in PI_FRAC_DICT.items()}

PHASE_DICT = {
    # pi-pulse
    'I': 0.0,
    '-I': 0.0,
    'X': 0.0,
    '-X': pi,
    'Y': 1 / 2. * pi,
    '-Y': 3 / 2. * pi,
    'XY': 1 / 4. * pi,
    'YX': 3 / 4. * pi,
    '-XY': 5 / 4. * pi,
    '-YX': 7 / 4. * pi,
    # piHalf-pulse
    'X/2': 0.0,
    '-X/2': pi,
    'Y/2': 1 / 2. * pi,
    '-Y/2': 3 / 2. * pi,
    'XY/2': 1 / 4. * pi,
    'YX/2': 3 / 4. * pi,
    '-XY/2': 5 / 4. * pi,
    '-YX/2': 7 / 4. * pi,
    # piQuad-pulse
    'X/4': 0.0,
    '-X/4': pi,
    'Y/4': 1 / 2. * pi,
    '-Y/4': 3 / 2. * pi,
    'XY/4': 1 / 4. * pi,
    'YX/4': 3 / 4. * pi,
    '-XY/4': 5 / 4. * pi,
    '-YX/4': 7 / 4. * pi,
    # piQuad3-pulse
    'X3/4': 0.0,
    '-X3/4': pi,
    'Y3/4': 1 / 2. * pi,
    '-Y3/4': 3 / 2. * pi,
    'XY3/4': 1 / 4. * pi,
    'YX3/4': 3 / 4. * pi,
    '-XY3/4': 5 / 4. * pi,
    '-YX3/4': 7 / 4. * pi,
    # another key set
    # pi-pulse
    'i': 0.0,
    'x': 0.0,
    'xy': 0.25 * pi,
    'y': 0.5 * pi,
    'yx': 0.75 * pi,
    # pi-half-amp
    'x+': 0.0,
    'xy+': 0.25 * pi,
    'y+': 0.5 * pi,
    'yx+': 0.75 * pi,
    'x-': 1.0 * pi,
    'xy-': 1.25 * pi,
    'y-': 1.5 * pi,
    'yx-': 1.75 * pi,
    # another key set (for state preparation)
    0.5: 0.0,
    0.5j: 0.5 * pi,
    -0.5: 1.0 * pi,
    -0.5j: -0.5 * pi,
}

# converts state(along axis) to operator on |0>
STATE_AXIS_OP_DICT = {
    # z-axis
    'z+': 'I',
    'z-': 'X',
    # on xy-plane
    'x+': 'Y/2',
    'y+': '-X/2',
    'x-': '-Y/2',
    'y-': 'X/2',
    # -- end of octomo
    'xy+': 'YX/2',
    'yx+': '-XY/2',
    'xy-': '-YX/2',
    'yx-': 'XY/2',
    # off xy-plane
    'xz+': 'Y/4',
    'zx+': '-Y/4',
    'xz-': '-Y3/4',
    'zx-': 'Y3/4',
    'yz+': '-X/4',
    'zy+': 'X/4',
    'yz-': 'X3/4',
    'zy-': '-X3/4',
    # another key set
    0.5: 0.5j,
    -0.5: -0.5j,
    0.5j: -0.5,
    -0.5j: 0.5,
}
