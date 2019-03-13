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


def compute_nearest_neighbours(image, ax=None, knn=1,
                               blob_size_px=5, peaks=True, threshold=0.2,
                               overlap=0.95):
    '''
    Envelope function to compute and display nearest neighbour
    distances of particles on a surface.
    
    Compute k\'th nearest neighbours.
    
    Accepts 2d image array as i_nput.
    
    returns nnd, ax
    --- nnd is an array of nearest neighbour distances
    --- ax is MPL axes
    
    NB. use set(nnd) to avoid double counting on 1st NN
    
    threshold=0.2 default for blob_log -> seems to work well'''
    if ax is None:
        #generate new figure
        f, ax = _plt.subplots()

    ax.imshow(image)
    
    single_pixel_image = create_location_array(image, peaks=peaks,
                                               blob_size_px=blob_size_px,
                                               threshold=threshold,
                                               plot=False,
                                               overlap=overlap)
    nnd, locations, nnd_index = nearest_neighbour_from_coordinates(
                                                single_pixel_image, knn=knn)
    
    plot_nnd(ax, locations, nnd_index)
    
    return nnd, ax

def parse_multiple_nnd_from_image(files, particle_size, knn=1,
                                  max_mag=6e6, min_mag=4e5):
    '''Computes k nearest neighbour distances for every DM3 image file
    in 'files'.
    
    Ignores images which are not square (messes with overall distance),
    images with mag greater than max_mag:
    - image too close for accurate cluster recognition (works mostly tbf...)
    - main problem is that often there is only one cluster in the image => crashes nn script
    
    plots the found nearest neighbours on the original image.
    
    returns a list of nnd arrays corresponding to 'files.' Units are m
    
    NOTE. this will take time to run ~1 min for 25 files.'''
    nnd_list = []
    for file in files:
        dm3 = _dm.dmReader(file) #open file
        
        try:
            x, y = dm3['data'].size
            assert x == y, 'Image not square.'
            # scale not const. between x and y => nnd wrong
            assert _dm3_functions.get_magnification(file) < max_mag, \
                'Magnification too high.'
            # mag too high for accurate cluster recognition.
            # main problem is that often there is only one cluster in the image
            # => crashes nn script
            assert _dm3_functions.get_magnification(file) > min_mag, \
                'Magnification too low.'
            # min_mag useful for when the clusters are too small for accurate
            # size distribution (gwy) and therefore error high on distance
            # from gwy, Pt923 (~3nm) is badly resolved under 500000x mag.
        except AssertionError as ae:
            print('Skipping image: {}'.format(ae))
            continue
        
        (_scalex, _scaley), (unitx, unity) = dm3['pixelSize'], dm3['pixelUnit']
        assert _scalex == _scaley, 'Image scale not equal in plane.'
        assert unitx == unity, 'Image scale (units) not equal in plane.'
        # m per pixel
        scale = _scalex * _dm3_functions.DM3_SCALE_DICT[unitx]
        
        # particle size in pixels on image
        particle_px = particle_size / scale # ...=scale_y is nm_per_pixel
        # get image data
        image = dm3['data']
        # generate MPL plot
        f, ax = _plt.subplots() 
        # do the calculations
        nnd, ax = compute_nearest_neighbours(image, ax, knn=knn,
                                             blob_size_px=particle_px,
                                             threshold=0.2)
        nnd *= scale # get distance in nm
        ax.set_title('knn={}'.format(knn))
        nnd_list.append(nnd) #append second nearest neighbour data
    
    return nnd_list
    
def create_location_array(image, peaks=True, threshold=0.2, 
                          blob_size_px=5, plot=True, overlap=0.95):
    '''creates a binary array of zeros
    location of found particle centres are 1's
    
    use relevant pixel sizes for blob_size_px -> it is used in skimage.blob_log
    min_sigma = blob_size_px-1, max_sigma = blob_size_px+1
    blob_size_log best obtained by particle_size*pixels_per_nm
    (pyPJH.dm3_lib.functions.distance_per_pixel())'''
    
    #single_pixel_image = _np.zeros_like(image) #create binary image
    #each particle is represented by single pixel
    
    if peaks:
        binary_image = image > _threshold_otsu(image)
        # regions of interest are peaks in the image
        # creates a binary image with pixels filled with 1 above threshold
    else:
        binary_image = image < _threshold_otsu(image)
        # regions of interest are 'troughs' or 'holes' in the image
    found = _blob_log(binary_image, min_sigma=blob_size_px/2,
                      max_sigma=2*blob_size_px, threshold=threshold,
                      overlap=overlap)
    
    if plot: #plot found particles
        f, ax = _plt.subplots()
        ax.imshow(image)
        
        for y, x, r in found:
            #plot found particles as circles over the original image
            ax.add_artist(_plt.Circle((x, y), r, fill=False, edgecolor='w'))
    
        #create binary pixels where particles exist
    # returns 3 numpy arrays of x, y, and sigma
    return _np.split(found, 3, axis=1)
