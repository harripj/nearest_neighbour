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
# from dm3_lib import DM3 as _DM3
import dm3_functions as _dm3_functions
import gwy_grains as _gwy_grains
import logging as _logging
try:
    from ncempy.io import dm as _dm
except ImportError:
    _logging.warning('ncempy package not loaded, as Python2 interpreter detected.')


def nearest_neighbour_chained_from_coordinates(x, y, radius, knn=10):
    '''
    Computes the nearest neighbours from their coordinates using a KDtree.
    The data is then grouped into 'sets' which are chained by at most 'radius'
    distance apart from at least one other point in the chain.
    
    Parameters
    ----------
    x, y: array-like
        List of coordinates in any reference frame, eg. pixels, or real distances.
    radius: float
        Maximum distance from any point in the chain for a queried point to be considered part 
        of that chain.
    knn: int
        Consider knn nearest neighbours for each point to test for 'chaining'.
        Best to keep this high, ie. greater than average chain length, but low enough 
        to keep CPU time down. Default is 10.
    
    Returns
    -------
    nnd: numpy.ndarray
        Nearest neighbour distance for each point considered.
    loc_arr: numpy.ndarray
        Coordinates of chained points for queried point.
    nnd_index: list
        List of arrays corresponding to the index of points found
        for each point in loc_arr.
    chain_index: numpy.ndarray
        Each point in a chain is denoted the same ID which can be used to identify
        chained points.
        
    '''
    if len(x) != len(y):
        raise ValueError('x and y must be the same shape.')
    
    
    # x, y = _np.where(binary_image) #particle centers are 1 and bg is 0
    loc_arr = _np.column_stack((x, y)) #concatenate arrays into correct format for tree
    
    tree = _cKDTree(loc_arr)
    
    nnd = [] # _np.zeros((len(x), knn))
    nnd_index = [] # _np.zeros_like(nnd)
    
    # check for all data has x and y positions
    chain_index = _np.zeros_like(x, dtype='int')
    
    for ind, (x, y) in enumerate(loc_arr):
        dist, index = tree.query((x, y), k=knn+1,
                                 distance_upper_bound=radius)
        # query must be +1,
        # as x, y is in tree => 1st result would be self.
            
        # if len(dist) == knn, then there are potentially
        # more pts within radius
        # dist returns inf for all queries > distance_upper_bound
        if not (dist==_np.inf).any():
            pass
        
        #first elements are the same as the cluster in question
        #ie. distance == 0 => remove them
        dist = _np.delete(dist, 0)
        index = _np.delete(index, 0)
        
        # handle inf
        index = index[dist<_np.inf]
        dist = dist[dist<_np.inf]
        
        # set row
        nnd.append(dist)
        nnd_index.append(index)
        
        # if previously assigned
        if chain_index[ind] != 0:
            number = chain_index[ind]
        # if previosuly unassigned
        else:
            number = chain_index.max() + 1
            
        chain_index[ind] = number
        
        # if chain_index of any found particles from query is previosuly assigned
        if (chain_index[index] != 0).any():
            # get the assignment number
            for val in chain_index[index]:
                if val != 0:
                    # make all found clusters with that assignment number == to current assignment number
                    chain_index[chain_index==val] = number
        
        # assign all found particles (inc. 0's) to assignment number
        chain_index[index] = number
                    
    return nnd, loc_arr, nnd_index, chain_index

def plot_chained_nn(x, y, grain_field, chain_index):
    '''
    x and y are PIXEL coordinate information of the particles:
    ie. (x_center/scale) and (y_center/scale),
        where scale is the original image scale in distance per pixel.
    
    grain_field is the reshaped original numbered_grain image data from
    gwyddion, ie. every pixel has a number corresponding to a particle:
        number != 0
    or background:
        number == 0.
    which is returned by gwyddion as a flattened array. This must be reshaped
    to be 2d -> typically by _np.reshape(grain_field, DM3(fname).size) for DM3
    data.
    
    Plots on mpl axes if plot is True.
    
    Returns the chained image where all particles in one chain have same ID.
        '''
    assert len(x) == len(y), 'x and y data must be pixel information and \
                         of same shape.'
        
    # image of zeros. chained particles will be added to this by changing 
    # bounding pixels of particles with the same number and therefore colour.
    chained_image = _np.zeros_like(grain_field)
        
    for i in range(len(x)):
        x_pix, y_pix = x[i], y[i]
        # get the number value of the cluster
        # -> gwyddion numbers all binding pixels of particles with one unique number
        # x = column, y = row
        particle_number = grain_field[y_pix, x_pix]
        
        # crescent shaped clusters will often have x_center and y_center on bg
        # this will therefore label background, which is incorrect.
        # this can be mitigated my checking whether the value at this pixel == 0:
        # which means there is no grain here. If this is the case check for closest
        # pixel which is non-zero, and choose grain by this method    
        if particle_number == 0:
            # coordinates where original image is non-zero, ie. non background.
            row, col, = _np.where(grain_field != 0)
            # difference between closest non-bg pixel and particle center
            delta = _np.sqrt((row-y_pix)**2 + (col-x_pix)**2)
            closest_index = _np.argmin(delta)
            
            # select the correct non-bg pixel (particle) and get its
            # assigned ID
            found_row, found_col = row[closest_index], col[closest_index]
            particle_number = grain_field[found_row, found_col]
        
        # get the correct bounding pixels
        mask = grain_field == particle_number
        # and plot on image
        chained_image[mask] = chain_index[i]
        
    return chained_image
