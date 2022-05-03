# -*- coding: utf-8 -*-

"""
Created on Sat Nov 18 22:49:29 2017

@author: pjh523
"""

import numpy as _np
from scipy.spatial import cKDTree as _cKDTree


def create_edge_image(mask, delta=(-1, 1), nonzero=False):
    """
    Creates an image of egdes from a binary or greyscale image eg. numbered,
    skimage.measure.label, or Gwyddion number_grains() image. Background must
    be denoted with zeroes.
    
    The original image is rolled by delta typically ±1 to find the edges in all
    of the images dimensions. The labelling of the original image is preserved.
    
    Parameters
    ----------
    image: numpy.ndarray
        Often a 2d (but can be 1d or 3d) array. Typically this image is
        greyscale, ie. skimage.measure.label.
    delta: tuple of ints
        The shift over which to find the edges, typically ±1.
        Default is (-1,1).
        
    Returns
    -------
    edges: array
        The found edges, maintains the original labelling. 
        Same shape as input array.
        
    """
    # create empty array to hold found edges
    edges = _np.zeros_like(mask)

    for axis in _np.arange(mask.ndim):  # works for 2d and 3d
        for shift in delta:
            # roll the image along one direction by shift in delta
            shifted_image = _np.roll(mask, shift, axis=axis)
            # the edges of the grains are found where the data is non-zero
            # in the original image and zero after rolling by ± 1
            changed = shifted_image != mask
            if nonzero:
                changed = _np.logical_and(changed, shifted_image > 0)
            # this preserves the original grain numbering system
            edges[changed] = mask[changed]

    return edges


def nearest_neighbour_edges(mask, k=1, n_jobs=-1, is_edges=False):
    """

    Calculate the k nearest neighbours from the edges of features using a labelled image.
    
    Parameters
    ----------
    
    mask: (M, N [,P])np.ndarray
        The labelled image.
    k: int
        Number of nearest edge locations to return.
    n_jobs: int
        Straight pass through to scipy.spatial.cKDTree.query. Default is -1.
    is_edges: bool
        If True then it is assumed that mask is an image of edges.
        If False then it is assumed that the features in mask are filled.
        
    Returns
    -------
    
    NB. returned arrays are consistent with the labelling in edges, ie. in order of labelling. N is the number of grains, ie. edges.max().
    nnd: (N, k) np.ndarray, 
        The nearest neighbour distance for each queried value in edge_image.
    queried: (N, k, edges.ndim) np.ndarray 
        Coordinates of the point within each grain responsible for the NN distance.
    results: (N, k, edges.ndim) np.ndarray
        Coordinates of the closest neighbouring point from a different grain.
        
    """
    # sort out input
    if is_edges:
        edges = mask
    else:
        edges = create_edge_image(mask)

    nz = _np.nonzero(edges)
    coords = _np.stack(nz, axis=1)
    labels = edges[nz]  # retain original labelling

    # make tree
    tree = _cKDTree(coords)

    # total number of pixels for every label
    counts = _np.bincount(edges.ravel())
    # ignore label 0 -> background
    # to get a result for every point, need to query counts.max()+k
    dists, index = tree.query(coords, k=counts[1:].max() + k)

    # want first result with different labelling
    result = labels[index] != labels[:, _np.newaxis]
    closest = _np.argmax(result, axis=1)

    dout = []
    iout = []
    qout = []

    for l in _np.unique(labels):
        # only get indices of one mask value
        mask = _np.equal(labels, l)

        # get the closest k distances from the selected points
        d = dists[mask, [closest[mask] + i for i in range(k)]]
        # sort by closest distance
        idx = _np.unravel_index(_np.argsort(d, axis=None)[:k], d.shape)

        dout.append(d[idx])
        iout.append(index[mask, [closest[mask] + i for i in range(k)]][idx])
        # nonzero(mask)[0] -> only care about label index
        # [idx[1]] -> and want which queried point it came from
        qout.append(_np.nonzero(mask)[0][idx[1]])

    return _np.stack(dout), coords[_np.stack(qout)], coords[_np.stack(iout)]


def plot_edge_neighbours(loc1, loc2, ax, **kwargs):
    """
    Plots the found nearest neighbours on the corresponding image.
    
    Parameters
    ----------
    queried_locations: array
        Coordinates, best obtained from nearest_neighbour_from_edge_image.
    result_locations: array
        Coordinates, best obtained from nearest_neighbour_from_edge_image.
    ax: plt.axes
        axes to plot on.
            
    """
    # sort out kwargs for plot
    kwargs.setdefault("color", "w")
    kwargs.setdefault("marker", None)
    kwargs.setdefault("lw", 1)

    for l1, l2 in zip(loc1, loc2):
        for i in range(l1.shape[0]):
            ax.plot([l1[i][1], l2[i][1]], [l1[i][0], l2[i][0]], **kwargs)


# def _DEPR_nearest_neighbour_from_edge_image(edge_image, n_jobs=-1):
#     """
#     From an edge_image, ie. a pixelated image where the ROI are numbered,
#     (best obtained using create_edge_image), the 1st nearest neigbour distance of
#     neighbouring particles is calculated.

#     Parameters
#     ----------

#     edge_image: np.ndarray (2D tested, should work for any number of dimensions)
#         The numbered (labelled) data of edges of features.
#     n_jobs: int
#         Straight pass through to scipy.spatial.cKDTree.query. Default is -1.

#     Returns
#     -------

#     NB. returned arrays are consistent with the labelling in edge_image, ie. in order of labelling.
#     nnd: np.ndarray, ndim=1
#         The nearest neighbour distance for each queried value in edge_image.
#     queried_locations: np.ndarray, shape=(len(labels), edge_image.ndim))
#         Coordinates of the closest pixel responsible for the NN value in the
#         queried grain.
#     result_locations: np.ndarray, shape=(len(labels), edge_image.ndim))
#         Coordinates of the closest pixel responsible for the NN value in a
#         neighbouring grain.

#     """

#     edge_data = _np.stack(_np.nonzero(edge_image), axis=1)
#     # make tree
#     tree = _cKDTree(edge_data)

#     labels = _np.unique(edge_image)
#     # delete background grain, defined as 0
#     labels = _np.delete(labels, 0)

#     # holder arrays for final result, to be returned
#     final_nnd = _np.empty(labels.size)
#     final_locations_neighbour = _np.empty((labels.size, edge_image.ndim))
#     final_locations_queried = _np.empty((labels.size, edge_image.ndim))

#     # loop over each grain value to query
#     for n_index, n in enumerate(labels):
#         # original location of grain on edge_image, labelled by n
#         queried_grain_mask = edge_image == n

#         # query tree where value is n for k cloest points
#         queried_points = _np.stack(_np.nonzero(queried_grain_mask), axis=1)
#         # must query one more point than in edge mask, otherwise no other particles would be found
#         knn = len(queried_points) + 1
#         nnd, index_nnd = tree.query(queried_points, k=knn)

#         # get the image coordinates of from the returned index of the query
#         coords = tree.data[_np.concatenate(index_nnd)].astype(int)

#         # get the index of the valid results (!=n) in image coords
#         valid_indices, = _np.where(
#             _np.concatenate(
#                 edge_image[tuple(_np.split(coords.T, edge_image.ndim, axis=0))] != n
#             )
#         )
#         # get nnd of valid results
#         nnd_c = _np.concatenate(nnd)
#         closest_index = _np.argmin(nnd_c[valid_indices])
#         # set final values in arrays
#         final_nnd[n_index] = nnd_c[valid_indices][closest_index]
#         final_locations_neighbour[n_index] = coords[valid_indices][closest_index]
#         # get final value for closest point in queried edge using index math (floor div)
#         queried_index = valid_indices[closest_index]
#         final_locations_queried[n_index] = queried_points[queried_index // knn]

#     return final_nnd, final_locations_queried, final_locations_neighbour

##### ----- OLD CODE BELOW, 3x as slow, keep here for legacy puposes for now, March 2019

#     if labels is None:
#         labels = _np.arange(edge_image.max()) + 1
#     # resultant image holder -> USE FOR PLOT
#     # _result = _np.zeros_like(edge_image)
#     # just use edge data
#     edge_data = _np.stack(_np.where(edge_image!=0), axis=1)
#     # make tree
#     tree = _cKDTree(edge_data)

#     _sz = len(labels)
#     final_nnd = _np.empty(_sz)
#     final_queried_grain_location = _np.empty((_sz, edge_image.ndim))
#     final_result_location = _np.empty((_sz, edge_image.ndim))

# # loop over each grain value to query
# for n_index, n in enumerate(labels):
#     # original location of grain on edge_image, labelled by n
#     original_grain_mask = (edge_image == n)

#     # query tree where value is n for k cloest points
#     queried_points = _np.stack(_np.where(original_grain_mask), axis=1)
#     # must query one more point than in edge mask, otherwise no other particles would be found
#     nnd, tree_index = tree.query(queried_points, k=len(queried_points)+1, n_jobs=n_jobs)

#     # concatenate points into one big array
#     #pixels = _np.concatenate(tree.data[tree_index].astype(int), axis=0)
#     # tree.data is float by default

#     # plot all found points
#     # split into columns to allow fancy indexing
#     #_result[_np.split(pixels, pixels.ndim, axis=1)] = 2*n
#     # reshow original grain, the values produce contrast
#     #_result[original_grain_mask] = n

#     # array length needed to find cloest => 1 for each data point in grain edge queried
#     _x, _y = _np.nonzero(original_grain_mask)
#     _size = len(_x)
#     _nnd = _np.empty(_size)
#     _queried_location = _np.empty((_size, edge_image.ndim))
#     _result_location = _np.empty((_size, edge_image.ndim))

#     # loop over nnd for each point
#     for i, _n in enumerate(nnd):
#         # each loop is for one queried point in the grain -> returns k found nn
#         # => _n.size = k
#         _indices = tree_index[i]
#         # becuase the grain points are queried,
#         # each loop over nnd corresponds to data in the query
#         self_data = queried_points[i]

#         ### use these lines below when assuming that queried point is in return values
#         ### that is to say the nnd of this point will be zero.
#         #self_index = _indices[_n.argmin()]
#         #self_data = tree.data[self_index].astype(int) # should be indices

#         # for each queried point, get the locations of found neighbours
#         neighbour_locations = tree.data[_indices].astype(int)
#         # get value from original data (not part of queried grain (!=n))
#         valid_points = edge_image[tuple(_np.split(neighbour_locations,
#                                          neighbour_locations.ndim,
#                                          axis=1))] != n
#         # get index of closest point that isn't self grain
#         closest = _n[valid_points.flatten()].argmin()
#         final_tree_index = _indices[valid_points.flatten()][closest]
#         closest_location = tree.data[final_tree_index].astype(int)

#         _nnd[_i] = _n[valid_points.flatten()].min()
#         _queried_location[_i] = self_data
#         _result_location[_i] = closest_location

#     _min = _nnd.argmin()

#     final_nnd[n_index] = _nnd[_min]
#     final_queried_grain_location[n_index] = _queried_location[_min]
#     final_result_location[n_index] = _result_location[_min]

# # returns nnd, location of pt on grain responsible for query, and location odf closest result for that pt
# return final_nnd, final_queried_grain_location, final_result_location


# def _DEPR_plot_edge_neighbours(
#     queried_locations, result_locations, ax, image=None, **kwargs
# ):
#     """
#     Plots the found nearest neighbours on the corresponding image.

#     Parameters
#     ----------
#     queried_locations: array
#         Coordinates, best obtained from nearest_neighbour_from_edge_image.
#     result_locations: array
#         Coordinates, best obtained from nearest_neighbour_from_edge_image.
#     ax: plt.axes
#         axes to plot on.
#     image: np.ndarray, ndim=2
#         Image to plot. Actual data or edge_image recommended. Default is None,
#         in this case image is not plotted on axes.

#     """
#     # sort out kwargs for plot
#     if len(kwargs) == 0:
#         kwargs = dict(color="w", marker=None, lw=1)

#     if image is not None:
#         ax.matshow(image)

#     for i, (_qrow, _qcol) in enumerate(queried_locations):
#         _rrow, _rcol = result_locations[i]
#         ax.plot((_qcol, _rcol), (_qrow, _rrow), **kwargs)
