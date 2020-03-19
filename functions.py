#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:49:29 2017

@author: pjh523
"""

from skimage.filters import threshold_otsu as _threshold_otsu
from skimage.feature import blob_log as _blob_log
from skimage.segmentation import clear_border as _clear_border
from matplotlib import pyplot as _plt
import numpy as _np
from scipy.spatial import cKDTree as _cKDTree

import gwy_grains as _gwy_grains
import logging as _logging

try:
    from ncempy.io import dm as _dm
except ImportError:
    _logging.warning("ncempy package not loaded, as Python2 interpreter detected.")


def nearest_neighbour_distribution_2D_impenetrable(r, sigma, rho, clip=True):
    """

    Produce expected NND probability distribution in 2D given a random (Poisson) arrangement of particles (hard disks in 2D).

    NB. distribution is valid for r > sigma.

    Equation obtained from 
    
    Parameters
    ----------
    r: float
        Nearest neighbour distances to model.
    sigma: float
        Disk (particle) diameter.
    rho: float
        Particle number density.
    clip: bool, default is True
        If True then D is 0 for all r < sigma.
        Overall function is returned otherwise without clipping.

    Returns
    -------
    D: numpy.ndarray
        Probability density for each point in r.

    References
    ----------
    [1] S Torquato et al 1990 J. Phys. A: Math. Gen. 23 L103. DOI: 10.1088/0305-4470/23/3/005

    """

    # disk volume (area in 2D)
    vol = _np.pi * (sigma / 2.0) ** 2
    # reduced density
    eta = rho * vol

    # scaled distance
    x = r / sigma

    D = (
        _np.exp(-4 * eta * (x ** 2 - 1 + eta * (x - 1)) / (1 - eta) ** 2)
        * 4
        * eta
        * (2 * x - eta)
        / (sigma * (1 - eta) ** 2)
    )

    # enforce that model should be 0 for all r < sigma
    if clip:
        D[r < sigma] = 0

    return D


def nearest_neighbour_distribution_2D_points(r, rho):
    """

    Nearest neighbour distribution probability desnity for a random arrangement of points in 2D, ie. on a surface [1].
    
    Parameters
    ----------
    r: float
        Radius to evaluate NND probability density.
    rho: float
        Number density.
    
    Returns
    -------
    p: float
        Probability density at radius r.
    
    References
    ----------
    [1] Nearest-neighbour distribution function for systems on interacting particles, Torquato et al., https://doi.org/10.1088/0305-4470/23/3/005

    """
    return rho * (2 * _np.pi * r) * _np.exp(-rho * _np.pi * r ** 2)


def nnd_stats(nnd):
    """returns mean and standard deviation of nnd from the array obtained
    by compute_nearest_neighbours. Has k columns assigned to k=1,2,3..."""
    means = _np.mean(nnd, axis=0)  # average down the columns
    std = _np.std(nnd, axis=0)  # std of nnd
    # equiavalent to 1st nnd, 2nd nnd, 3rd nnd... etc.

    return means, std


def plot_multiple_nnd_stats(nnd, ax=None, gap=0.01):
    """Plots nnd stats for all, and each individual image
    (discernible by colour). Plots mean ± std.
    
    Stats for each image are separated by gap on plot
    
    returns ax
    """
    if ax is None:
        f, ax = _plt.subplots()

    for ind, n in enumerate(nnd):  # plot nnd
        mean, std = nnd_stats(n)  # stats from each individual image
        for k, val in enumerate(mean):
            x = k + 1 - (len(nnd) / 2) * gap + (gap * ind)  # x axis positioning
            ax.errorbar(x, val, yerr=std[k], color="r", marker="x")
            # plot each image stats

    # stack arrays for all stats
    nnd_all = _np.vstack(nnd)

    no, k = nnd_all.shape
    mean_all, std_all = nnd_stats(nnd_all)  # calc overall stats
    for i in range(k):
        ax.errorbar(i + 1, mean_all[i], yerr=std[i], color="b", marker="x")

    ax.set_xlabel("k'th nearest neighbour")
    ax.set_xticks([i for i in range(k + 1)])

    xlim_min = 1 - ((len(nnd) + 2 * gap) / 2) * gap
    xlim_max = k + ((len(nnd) + 2 * gap) / 2) * gap
    # min and max data point (k=1, k) plus some padding
    ax.set_xlim(xlim_min, xlim_max)

    return ax


# def parse_multiple_nnd_from_JSON(json_list, knn=1, plot=True, scale_data=False):
#     """
#     Parse nnd from Gwyddion analysis JSON files.
#     Locations in image are stored as 'x_center', 'y_center'

#     Beneficial over nnd by image as the grain locations are consistent with
#     area analysis etc.

#     Image analysis also underestimates distances becuase it finds more blobs.

#     Parameters
#     ----------
#     json_list: list-like
#         A list of JSON file names each containing grain data.
#     knn: int
#         Number of nearest neighbours to calculate. Default is 1.
#     plot: boolean
#         If True, the nearest neighbours are shown on an MPL figure. Default is
#         True.
#     scale_data: boolean
#         If True the nnd data is scaled by the image pixel size. NB. at some
#         point Gwyddion (v2.50?) went from 'x_center' as pixel coordiantes to
#         real coordinates, hence the need for this argument. Default is False.

#     Returns
#     -------
#     nnd_total: numpy array
#         The nnd for each grain. Column 0 is 1st nn, column 1 is 2nd nn etc.
#     """
#     nnd_total = []

#     for json in json_list:
#         dm3_fpath = _gwy_grains.get_image_from_JSON(json)
#         dm3 = _DM3(dm3_fpath)

#         data = _gwy_grains.parse_grain_JSON(json)
#         nnd, loc, idx = nearest_neighbour_from_coordinates(
#             data["x_center"], data["y_center"], knn=knn
#         )

#         # if the analysis needs scaling by pixel size
#         _scale, unit = dm3.pxsize
#         scale = _scale * _dm3_functions.DM3_SCALE_DICT[unit.decode()]

#         if scale_data:
#             nnd_total.append(nnd * scale)
#         else:
#             nnd_total.append(nnd)

#         if plot:
#             # plots and formats axis
#             f, ax = _dm3_functions.plot_image(dm3)
#             if scale_data:
#                 # if the data needs scaling by pixel size => in pixels
#                 plot_nnd(ax, loc, idx)
#             else:
#                 # if the data doesn't need scaling then locations need to be
#                 # divided by pixel size such that they are in pixels
#                 plot_nnd(ax, loc / scale, idx)

#     nnd_total = _np.concatenate(nnd_total)
#     print("NND mean: {} m.".format(nnd_total.mean(axis=0)))

#     return nnd_total
