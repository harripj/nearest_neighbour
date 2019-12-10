#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:49:29 2017

@author: pjh523
"""

from skimage.segmentation import clear_border as _clear_border
import numpy as _np
from scipy import spatial as _spatial


def nearest_neighbour_from_coordinates(
    coords, knn=1, buffer_size=None, pc_shape=None, **kwargs
):
    """
    Computes the nearest neighbour from a binary image using a KDtree.
    
    Ideally each region of interest (ROI) has been reduced to a single pixel
    (eg. use create_location_array) this prevents self ROI nnd counting.
    
    (knn) makes the function evaluate the k\'th nearest neighbour
    ie. knn=1 means 1st nearest neighbour
    
    Parameters
    ----------
    coords: numpy.ndarray
        List of coordinates to query.
    knn: int
        Query k\'th nearest neighbour.
    buffer_size: int or None
        Points closer to the edge of the point cloud than this buffer are not queried.
        Default is None, in which case all points are queried. Edge limit is defined by pc_shape.
    pc_shape: tuple, same length as coords.ndim
        Must be defined if buffer_size is defined. The size of the point cloud, needed to
        know the limits from which points from the edge are to be excluded from the query.
        eg. for 2D point cloud constructed from features of an image 512x512 pixels, 
        pc_shape = (512, 512) = image.shape. Default is None.
    **kwargs:
        Keyword arguments to pass to scipy.spatial.cKDTree.query.
        
    Returns
    -------
    nnd: numpy.ndarray
        Nearest neighbour distances for the corresponding coordinates.
        Number of columns = knn with k\'th column corresponding to k\'th nnd.
    nnd_index: numpy.ndarray
        Array of indices corresponding to the found nearest neighbour for each query in loc_arr.
    queried: numpy.ndarray
        Array of the locations of the queried points.
        
    """
    tree = _spatial.cKDTree(coords)

    if buffer_size is not None:
        assert (
            pc_shape is not None and len(pc_shape) == coords.ndim
        ), "pc_shape must be set and must be of length as the same number of dimensions as coords."
        # generate binary image with pixels set as True from coords, and then clear edges with buffer
        mask = _np.zeros(pc_shape)
        mask[tuple(_np.split(_np.around(coords).astype(int), coords.ndim, axis=1))] = 1
        mask = _clear_border(mask, buffer_size=buffer_size)

        # queried coordinates are those which are still present in mask even after edge filtering
        q_coords = _np.column_stack(_np.where(mask))
        dist, index = tree.query(q_coords, k=knn + 1)

        # delete self contribution from results
        dist = _np.delete(dist, 0, axis=1)
        index = _np.delete(index, 0, axis=1)

        return dist, index, q_coords
        # reset coords which are to be queried
    else:
        # query the tree for knn+1 closest locations: +1 because closest will be self
        dist, index = tree.query(coords, k=knn + 1, **kwargs)

        # delete self contribution from results
        dist = _np.delete(dist, 0, axis=1)
        index = _np.delete(index, 0, axis=1)

        return dist, index, coords


def plot_nnd(locations, nnd_index, ax, queried=None, **kwargs):
    """
    Plots the corresponding pairs on an axis, ax.
    loactions and nnd_index best obtained from:
    nearest_neighbour_particle()

    NB. if plot looks off, try flipping coordinates eg. np.fliplr(locations) and np.fliplr(queried).
    (this would be an xy vs ij indexing problem).

    Parameters
    ----------
    locations: np.ndarray
        Array of all locations (obtained from nearest neighbour function).
    nnd_index: np.ndarray
        Array of corresponding NN index in location array.
    ax: plt.axes
        Axes to plot on. Works best when image is already plotted on axes (low zorder).
    queried: np.ndarray
        Array of locations actually queried. If None then assumed to be the same as locations.
        Default is None.
    **kwargs:
        Keyword arguments to pass to ax.plot.

    """
    # default plotting args
    if not len(kwargs):
        kwargs = dict(color="w", marker=None, lw=1)

    # if queried is undefined then queried=locations (all)
    if queried is None:
        queried = locations

    for index, particle_indices in enumerate(nnd_index):
        # for each queried particle, get its location
        x0, y0 = queried[index]
        for particle_index in particle_indices:
            # get the corresponding nnd particle(s)
            x1, y1 = locations[particle_index]
            # corresponding particle locations
            ax.plot((x0, x1), (y0, y1), **kwargs)  # plot them!


def calculate_Voronoi_volumes(points, limit_bounds=True):
    """
    Compute Voronoi diagram and calculate Voronoi volumes.
    Open Voronoi regions are returned with volume np.inf.

    Parameters
    ----------
    points: (M, N) ndarray
        Array of M points in N dimensions over which to produce Voronoi diagram.
    limit_bounds: bool, default True
        If True only points with vertices less than points.max(axis=0) and greater than points.min(axis=0) are considered.
        Other regions are considered 'open' and the calculated volume is np.inf.

    Returns
    -------
    volumes: (M,) ndarray
    vor: scipy.spatial.Voronoi
        Constructed Voronoi class.
        Easily plotted with scipy.spatial.voronoi_plot_2d.

    """
    # format input
    points = _np.asarray(points)
    # construct Voronoi
    vor = _spatial.Voronoi(points)
    # out array
    volumes = _np.empty(len(points), dtype=float)

    for i in _np.arange(vor.npoints):
        # i is linked to point index
        # each point i is linked to an associated region -> point_region[i], which is an index
        # each region is constrained by vertices indexed as vertices[regions]
        region = vor.regions[vor.point_region[i]]

        # if any region is labelled -1 then it is not closed
        if _np.any(_np.array(region) < 0):
            volumes[i] = _np.inf
        else:
            verts = vor.vertices[region]
            # if vertex is outide of image bounds
            if limit_bounds and _np.any(
                _np.logical_or(verts > vor.max_bound, verts < vor.min_bound)
            ):
                volumes[i] = _np.inf
            else:
                hull = _spatial.ConvexHull(verts)
                volumes[i] = hull.volume

    return volumes, vor
