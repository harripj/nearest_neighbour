#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:49:29 2017

@author: pjh523
"""

import numpy as _np


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
