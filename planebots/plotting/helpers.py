import logging

import numpy as np

# rc('text', usetex=True)
from planebots.vision.detection import retrieveCamPos

logger = logging.getLogger(__name__)


def deltas(ts):
    dts = np.zeros_like(ts)
    for i in range(len(ts) - 1):
        j = i + 1
        dt = ts[j] - ts[i]
        dts[j] = dt
    return dts


def plotSerieElisa(filename, marker, dictname, tdur=-1, arrayIndex=-1, tshift=0, t0=-1, retNaN=False):
    """If the array index = -1, a single dimension series is expected"""
    eli = np.load(filename)
    ts = eli['s_z_ts'][:, marker, :].ravel()
    if t0 == -1:
        t0 = ts[0]
    else:
        t0 = t0
    tstart = tshift
    idx_start = np.where(ts - t0 > tstart)[0][0]
    if tdur == -1:
        timespan = ts[-2].ravel() - t0 - tstart
        idx_end = len(ts)
    else:
        timespan = min(tdur, ts[-2] - t0 - tstart)
        idx_end = np.where(ts - ts[idx_start] >= timespan)[0][0]
        idx_end = min(idx_end, len(ts))
    idx_len = idx_end - idx_start

    if arrayIndex == -1:
        serie = eli[dictname][idx_start:idx_end, marker, :]
        t_nonan = np.where(np.isnan(serie[:, 0]) == False)
    else:
        serie = eli[dictname][idx_start:idx_end, marker, arrayIndex]
        nonan = np.where(np.isnan(serie[:]) == False)
        t_nonan = ts[nonan]
    serie_nonan = serie[np.where(np.isnan(serie) == False)]
    if retNaN:
        return ts, serie
    else:
        return t_nonan, serie_nonan


def plotSerieVision(filename, marker, dictname, tdur=-1, arrayIndex=-1, tshift=0, t0=-1, retNaN=False):
    """If the array index = -1, a single dimension series is expected"""
    vis = np.load(filename)
    ts = vis['s_ts'][:].ravel()
    tstart = tshift
    if t0 == -1:
        t0 = ts[0].ravel()
    pos, ax = retrieveCamPos(vis['s_rvec'][0], vis['s_tvec'][0])
    idx_start = np.where(ts - t0 > tstart)[0][0]
    if tdur == -1:
        timespan = ts[-2].ravel() - t0 - tstart
        idx_end = len(ts)
    else:
        timespan = min(tdur, ts[-2] - t0 - tstart)
        try:
            idx_end = np.where(ts - ts[idx_start] >= timespan)[0][0]
            idx_end = min(idx_end, len(ts))
        except ValueError as e:
            logger.error(e)
    idx_len = idx_end - idx_start

    times = ts[idx_start:idx_end]
    if arrayIndex == -1:
        serie = vis[dictname][idx_start:idx_end, marker, :]
        t_nonan = np.where(np.isnan(serie[:, 0]) == False)
    else:
        serie = vis[dictname][idx_start:idx_end, marker, arrayIndex]
        nonan = np.where(np.isnan(serie[:]) == False)
        t_nonan = ts[nonan]
    serie_nonan = serie[np.where(np.isnan(serie) == False)]

    if retNaN:
        return ts, serie
    else:
        return t_nonan, serie_nonan


markerArgs = {'linewidth': 0, 'markersize': 4, 'marker': 'x', 'fillstyle': None, 'markeredgewidth': 0.5}

dcscLinewidth = 6.125  # inches

default_width = 5.78853  # in inches
default_ratio = (5 ** .5 - 1.0) / 2.0  # golden mean

# sizes = xx-small, x-small, small, medium, large, x-large, xx-large
xs = 'x-small'
small = "small"
mplrcParams = ({
    # rc params for plotting with xelatex pgf:
    "text.usetex": True,
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    'font.size': 12,
    "axes.titlesize": small,  # fontsize of the axes title
    "axes.labelsize": small,  # fontsize of the x any y labels
    'legend.fontsize': small,
    'xtick.labelsize': small,
    'ytick.labelsize': small,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "pgf.preamble": [
        # put LaTeX preamble declarations here
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        # macros defined here will be available in plots, e.g.:
        r"\newcommand{\vect}[1]{#1}",
        # You can use dummy implementations, since you LaTeX document
        # will render these properly, anyway.
    ],
})

import matplotlib.pyplot as plt

"""
Returns a figure with an appropriate size and tight layout.
"""


def figure(width=default_width, ratio=default_ratio, pad=0, *args, **kwargs):
    fig = plt.figure(figsize=(width, width * ratio), *args, **kwargs)
    fig.set_tight_layout({
        'pad': pad
    })
    return fig


"""
Returns subplots with an appropriate figure size and tight layout.
"""


def subplots(width=default_width, ratio=default_ratio, *args, **kwargs):
    fig, axes = plt.subplots(figsize=(width, width * ratio), *args, **kwargs)
    fig.set_tight_layout({
        'pad': 0
    })
    return fig, axes


"""
Save both a PDF and a PGF file with the given filename.
"""


def savefig(filename, *args, **kwargs):
    plt.savefig(filename + '.pdf', *args, **kwargs)
    plt.savefig(filename + '.pgf', *args, **kwargs)
