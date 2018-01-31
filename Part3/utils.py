import cartopy.util as cutil
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sgeom

import seaborn as sns

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import from_levels_and_colors

# ==================================================================================================

def sample_data_3d(nlons, nlats):
    """Returns `lons`, `lats`, and fake `data`

    adapted from:
    http://scitools.org.uk/cartopy/docs/v0.15/examples/axes_grid_basic.html
    """
    
    dlat = 180. / nlats / 2
    dlon = 360. / nlons

    lat = np.linspace(-90 + dlat, 90 - dlat, nlats)   
    lon = np.linspace(0, 360 - dlon, nlons)

    lons, lats = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    wave = 0.75 * (np.sin(2 * lats) ** 8) * np.cos(4 * lons)
    mean = 0.5 * np.cos(2 * lats) * ((np.sin(2 * lats)) ** 2 + 2)
    data = wave + mean
    
    return lon, lat, data

# ==================================================================================================


def from_levels_and_cmap(levels, cmap, extend='neither'):
    """
    
    Parameters
    ----------
    levels : sequence of numbers
        The quantization levels used to construct the :class:`BoundaryNorm`.
        Values ``v`` are quantizized to level ``i`` if
        ``lev[i] <= v < lev[i+1]``.
    cmap : string
        Valid colormap identifier.
    extend : {'neither', 'min', 'max', 'both'}, optional
        The behaviour when a value falls out of range of the given levels.
        See :func:`~matplotlib.pyplot.contourf` for details.

    ..note::
      Adapted from xarray.

    """
    
    if extend == 'both':
        ext_n = 2
    elif extend in ['min', 'max']:
        ext_n = 1
    else:
        ext_n = 0

    # subtract 1 because there is one less level than numbers
    pal = sns.color_palette(cmap, n_colors=len(levels) + ext_n - 1)
    cmap, norm = from_levels_and_colors(levels, pal, extend=extend)
    
    return cmap, norm


# ==================================================================================================

# from xarray
def infer_interval_breaks(x, y, clip=False):
    """"
    find edges of gridcells, given their centers
    """
    
    if len(x.shape) == 1:
        x = _infer_interval_breaks(x)
        y = _infer_interval_breaks(y)
    else:
        # we have to infer the intervals on both axes
        x = _infer_interval_breaks(x, axis=1)
        x = _infer_interval_breaks(x, axis=0)
        y = _infer_interval_breaks(y, axis=1)
        y = _infer_interval_breaks(y, axis=0)

    if clip:
        y = np.clip(y, -90, 90)
        
    return x, y


# from xarray
def _infer_interval_breaks(coord, axis=0):
    """
    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    >>> _infer_interval_breaks([[0, 1], [3, 4]], axis=1)
    array([[-0.5,  0.5,  1.5],
           [ 2.5,  3.5,  4.5]])
    """
    coord = np.asarray(coord)
    deltas = 0.5 * np.diff(coord, axis=axis)
    if deltas.size == 0:
        deltas = np.array(0.0)
    first = np.take(coord, [0], axis=axis) - np.take(deltas, [0], axis=axis)
    last = np.take(coord, [-1], axis=axis) + np.take(deltas, [-1], axis=axis)
    trim_last = tuple(slice(None, -1) if n == axis else slice(None)
                      for n in range(coord.ndim))
    return np.concatenate([first, coord[trim_last] + deltas, last], axis=axis)


# ==================================================================================================


def resize_colorbar_vert(cbax, ax1, ax2=None, size=0.04, pad=0.05, shift='symmetric', shrink=None):
    """
    automatically resize colorbars on draw
    
    See below for Example
    
    Parameters
    ----------
    
    cbax : colorbar Axes
        Axes of the colorbar.
    ax1 : Axes
        Axes to adjust the colorbar to.
    ax2 : Axes, optional 
        If the colorbar should span more than one Axes. Default: None.
    size : float
        Width of the colorbar in Figure coordinates. Default: 0.04.
    pad : float
        Distance of the colorbar to the axes in Figure coordinates.
         Default: 0.05.
    shift : 'symmetric' or float in 0..1
        Fraction of the total height that the colorbar is shifted upwards.
        See Note. Default: 'symmetric'
    shrink : None or float in 0..1
        Fraction of the total height that the colorbar is shrunk.
        See Note. Default: None.
        
    Note
    ----   
    
    shift='symmetric', shrink=None  -> colorbar extends over the whole height
    shift='symmetric', shrink=0.1   -> colorbar is 10 % smaller, and centered
    shift=0., shrink=0.1            -> colorbar is 10 % smaller, and aligned with the bottom
    shift=0.1, shrink=None          -> colorbar is 10 % smaller, and aligned with the top   
    
    Exaples
    -------
    # example with 1 axes
    
    f = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    h = ax.pcolormesh([[0, 1]])
    
    ax.coastlines()

    cbax = f.add_axes([0, 0, 0.1, 0.1])
    cbar = plt.colorbar(h, orientation='vertical', cax=cbax)

    func = resize_colorbar_vert(cbax, ax)
    f.canvas.mpl_connect('draw_event', func)

    ax.set_global()

    plt.draw()
    
    
    # =========================
    # example with 2 axes
    
    f, axes = plt.subplots(2, 1, subplot_kw=dict(projection=ccrs.Robinson()))

    for ax in axes:
        ax.coastlines() 
        ax.set_global()
        h = ax.pcolormesh([[0, 1]])
    
    cbax = f.add_axes([0, 0, 0.1, 0.1])
    cbar = plt.colorbar(h, orientation='vertical', cax=cbax)
        
    func = resize_colorbar_vert(cbax, axes[0], axes[1], size=0.04, pad=.04, shrink=None, shift=0.1)
    
    f.canvas.mpl_connect('draw_event', func)

    cbax.set_xlabel('[°C]', labelpad=10)

    plt.draw()
    
    # =========================
    # example with 3 axes & 2 colorbars
    
    f, axes = plt.subplots(3, 1, subplot_kw=dict(projection=ccrs.Robinson()))

    for ax in axes:
        ax.coastlines() 
        ax.set_global()

    h0 = ax.pcolormesh([[0, 1]])
    h1 = ax.pcolormesh([[0, 1]])
    h2 = ax.pcolormesh([[0, 1]], cmap='Blues')

    cbax = f.add_axes([0, 0, 0.1, 0.1])
    cbar = plt.colorbar(h1, orientation='vertical', cax=cbax)
    func = utils.resize_colorbar_vert(cbax, axes[0], axes[1])
    f.canvas.mpl_connect('draw_event', func)

    cbax = f.add_axes([0, 0, 0.1, 0.11])
    cbar = plt.colorbar(h2, orientation='vertical', cax=cbax)
    func = utils.resize_colorbar_vert(cbax, axes[2])
    f.canvas.mpl_connect('draw_event', func)

    plt.draw()
    
    
    See also
    --------
    resize_colorbar_horz
    """
    
    shift, shrink = _parse_shift_shrink(shift, shrink)

    # swap axes if ax1 is above ax2
    if ax2 is not None:
        posn = ax1.get_position()
        posn2 = ax2.get_position()

        ax1, ax2 = (ax1, ax2) if posn.y0 < posn2.y0 else (ax2, ax1)
    
    # inner function is called by event handler
    def inner(event=None): 
        
        
        posn = ax1.get_position()
        
        # determine total height of all axes
        if ax2 is not None:    
            posn2 = ax2.get_position()
            full_height = posn2.y0 - posn.y0 + posn2.height
        else:
            full_height = posn.height
        
        # calculate position
        left = posn.x0 + posn.width + pad
        bottom = posn.y0 + shift * full_height
        width = size
        height = full_height - shrink * full_height
        
        pos = [left, bottom, width, height]
        
        cbax.set_position(pos)
        
    return inner

# ====================================


def resize_colorbar_horz(cbax, ax1, ax2=None, size=0.05, pad=0.05, shift='symmetric', shrink=None):
    """
    automatically resize colorbars on draw
    
    See below for Example
    
    Parameters
    ----------
    cbax : colorbar Axes
        Axes of the colorbar.
    ax1 : Axes
        Axes to adjust the colorbar to.
    ax2 : Axes, optional 
        If the colorbar should span more than one Axes. Default: None.
    size : float
        Height of the colorbar in Figure coordinates. Default: 0.04.
    pad : float
        Distance of the colorbar to the axes in Figure coordinates.
         Default: 0.1.
    shift : 'symmetric' or float in 0..1
        Fraction of the total width that the colorbar is shifted to the right.
        See Note. Default: 'symmetric'
    shrink : None or float in 0..1
        Fraction of the total width that the colorbar is shrunk.
        See Note. Default: None.
        
    Note
    ----   
    
    shift='symmetric', shrink=None  -> colorbar extends over the whole width
    shift='symmetric', shrink=0.1   -> colorbar is 10 % smaller, and centered
    shift=0., shrink=0.1            -> colorbar is 10 % smaller, and aligned with the left hand side
    shift=0.1, shrink=None          -> colorbar is 10 % smaller, and aligned with the right hand side
    
    Exaples
    -------
    # example with 1 axes
    
    f = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines()

    cbax = f.add_axes([0, 0, 0.1, 0.1])
    cbar = plt.colorbar(h, orientation='horizontal', cax=cbax)

    func = utils.resize_colorbar_horz(cbax, ax)
    f.canvas.mpl_connect('draw_event', func)

    ax.set_global()

    plt.draw()
    
    
    # =========================
    # example with 2 axes
    
    f, axes = plt.subplots(1, 2, subplot_kw=dict(projection=ccrs.Robinson()))

    for ax in axes:
        ax.coastlines() 
        ax.set_global()

        h = ax.pcolormesh([[0, 1]])

    cbax = f.add_axes([0, 0, 0.1, 0.1])
    cbar = plt.colorbar(h, orientation='horizontal', cax=cbax)

    func = resize_colorbar_horz(cbax, axes[0], axes[1], size=0.04, pad=.04, shrink=None, shift=0.1)

    f.canvas.mpl_connect('draw_event', func)

    cbax.set_ylabel('[°C]', labelpad=10, rotation=0, ha='right', va='center')

    plt.draw()
    
    # =========================
    # example with 3 axes & 2 colorbars
    
    f, axes = plt.subplots(1, 3, subplot_kw=dict(projection=ccrs.Robinson()))

    for ax in axes:
        ax.coastlines() 
        ax.set_global()

    h0 = ax.pcolormesh([[0, 1]])
    h1 = ax.pcolormesh([[0, 1]])
    h2 = ax.pcolormesh([[0, 1]], cmap='Blues')


    cbax = f.add_axes([0, 0, 0.1, 0.1])
    cbar = plt.colorbar(h1, orientation='horizontal', cax=cbax)
    func = utils.resize_colorbar_horz(cbax, axes[0], axes[1], size=0.04)
    f.canvas.mpl_connect('draw_event', func)

    cbax = f.add_axes([0, 0, 0.1, 0.11])
    cbar = plt.colorbar(h2, orientation='horizontal', cax=cbax)
    func = utils.resize_colorbar_horz(cbax, axes[2], size=0.04)
    f.canvas.mpl_connect('draw_event', func)

    plt.draw()
    
    
    See also
    --------
    resize_colorbar_vert
    """
        
    shift, shrink = _parse_shift_shrink(shift, shrink)
    
    if ax2 is not None:
        posn = ax1.get_position()
        posn2 = ax2.get_position()

        # swap axes if ax1 is right ax2
        ax1, ax2 = (ax1, ax2) if posn.x0 < posn2.x0 else (ax2, ax1)
    
    
    def inner(event=None): 
        
        posn = ax1.get_position()
        
        if ax2 is not None:
            posn2 = ax2.get_position()
            full_width = posn2.x0 - posn.x0 + posn2.width
        else:
            full_width = posn.width
        
        
        left = posn.x0 + shift * full_width
        bottom = posn.y0 - (pad + size)
        width = full_width - shrink * full_width
        height = size
        
        pos = [left, bottom, width, height]
        
        cbax.set_position(pos)
        
    return inner

# ====================================


def _parse_shift_shrink(shift, shrink):

    if shift == 'symmetric':
        if shrink is None:
            shrink = 0

        shift = shrink / 2.
    
    else:
        if shrink is None:
            shrink = shift
            
            
    assert (shift >= 0.) & (shift <= 1.), "'shift' must be in 0...1"
    assert (shrink >= 0.) & (shrink <= 1.), "'shrink' must be in 0...1"

    if shift > shrink:
        msg = ("Warning: 'shift' is larger than 'shrink', colorbar\n" 
               "will extend beyond the axes!")
        print(msg)
    
    return shift, shrink


# test code for _parse_shift_shrink
assert _parse_shift_shrink('symmetric', None) == (0., 0.)
assert _parse_shift_shrink('symmetric', 1) == (0.5, 1)
assert _parse_shift_shrink(1, None) == (1, 1)
assert _parse_shift_shrink(1, 1) == (1, 1)

# ==================================================================================================


def cyclic_dataarray(da, coord='lon'):
    """ Add a cyclic coordinate point to a DataArray along a specified
    named coordinate dimension.
    >>> import xarray as xr
    >>> data = xr.DataArray([[1, 2, 3], [4, 5, 6]],
    ...                      coords={'x': [1, 2], 'y': range(3)},
    ...                      dims=['x', 'y'])
    >>> cd = cyclic_dataarray(data, 'y')
    >>> print cd.data
    array([[1, 2, 3, 1],
           [4, 5, 6, 4]])
           
    Note
    -----
    After: https://github.com/darothen/plot-all-in-ncfile/blob/master/plot_util.py
    
    """
    import xarray as xr
    
    assert isinstance(da, xr.DataArray)

    lon_idx = da.dims.index(coord)
    cyclic_data, cyclic_coord = cutil.add_cyclic_point(da.values,
                                                 coord=da.coords[coord],
                                                 axis=lon_idx)

    # Copy and add the cyclic coordinate and data
    new_coords = dict(da.coords)
    new_coords[coord] = cyclic_coord
    new_values = cyclic_data

    new_da = xr.DataArray(new_values, dims=da.dims, coords=new_coords)

    # Copy the attributes for the re-constructed data and coords
    for att, val in da.attrs.items():
        new_da.attrs[att] = val
    for c in da.coords:
        for att in da.coords[c].attrs:
            new_da.coords[c].attrs[att] = da.coords[c].attrs[att]

    return new_da

# ----------------------------------------------------------------------

def set_map_layout(axes, width=17.0):
    """
    set figure height, given width

    Needs to be called after all plotting is done.
       
    Parameters
    ----------
    axes : ndarray of (Geo)Axes
        Array with all axes of the figure.
    width : float
        Width of the full figure in cm. Default 17

    ..note: currently only works if all the axes have the same aspect
    ratio.
    """

    if isinstance(axes, plt.Axes):
        ax = axes
    else:
        # assumes the first of the axes is representative for all
        ax = axes.flat[0]
    
    # read figure data
    f = ax.get_figure()

    bottom = f.subplotpars.bottom
    top = f.subplotpars.top
    left = f.subplotpars.left
    right = f.subplotpars.right
    hspace = f.subplotpars.hspace
    wspace = f.subplotpars.wspace

    # data ratio is the aspect
    aspect = ax.get_data_ratio()
    # get geometry tells how many subplots there are
    nrow, ncol, __ = ax.get_geometry()


    # width of one plot, taking into account
    # left * wf, (1-right) * wf, ncol * wp, (1-ncol) * wp * wspace
    wp = (width - width * (left + (1-right))) / (ncol + (ncol-1) * wspace) 

    # height of one plot
    hp = wp * aspect

    # height of figure
    height = (hp * (nrow + ((nrow - 1) * hspace))) / (1. - (bottom + (1 - top)))


    f.set_figwidth(width / 2.54)
    f.set_figheight(height / 2.54)

# ----------------------------------------------------------------------


def _get_label_attr(labelpad, size, weight):

    if labelpad is None:
        labelpad = mpl.rcParams['axes.labelpad']

    if size is None:
        size = mpl.rcParams['axes.labelsize']

    if weight is None:
        weight = mpl.rcParams['axes.labelweight']
    
    return labelpad, size, weight

# --------------


def ylabel_map(s, labelpad=None, size=None, weight=None, y=0.5, ax=None, **kwargs):
    """
    add ylabel to cartopy plot

    Parameters
    ----------
    s : string
        text to display
    labelpad : float, optional
        Distance of labels to axes. Defaults to mpl.rcParams['axes.labelpad']
        which is usually 4.
    size : float or fontsize, optional
        Fontsize, defaults to mpl.rcParams['axes.labelsize'], usually
        'medium'.
    weight : string, optional
        Fontweight, defaults to mpl.rcParams['axes.labelweight'], usually
        'normal'.
    y : float
        y position in axes coordinates. Default 0.5
    ax : matplotlib axis
        axis to add the label
    **kwargs : keyword arguments
        see matplotlib text help

    Returns
    -------
    h : handle
        text handle of the created text field

    ..note::
    http://stackoverflow.com/questions/35479508/cartopy-set-xlabel-set-ylabel-not-ticklabels

    """
    if ax is None:
        ax = plt.gca()
    
    labelpad, size, weight = _get_label_attr(labelpad, size, weight)
    
    va = kwargs.pop('va', 'bottom')
    ha = kwargs.pop('ha', 'center')
    rotation = kwargs.pop('rotation', 'vertical')
    rotation_mode = kwargs.pop('rotation_mode', 'anchor')
    
    transform = kwargs.pop('transform', ax.transAxes)

    h = ax.annotate(s, xy=(0, y), xycoords=transform,
                xytext=(-labelpad, 0), textcoords='offset points',
                va=va, ha=ha, rotation=rotation,
                rotation_mode=rotation_mode,
                size=size, weight=weight,
                **kwargs)
    
    return h

# ----------------------------------------------------------------------


def xlabel_map(s, labelpad=None, size=None, weight=None, x=0.5, ax=None, **kwargs):
    """
    add xlabel to cartopy plot

    Parameters
    ----------
    s : string
        text to display
    labelpad : float, optional
        Distance of labels to axes. Defaults to mpl.rcParams['axes.labelpad']
        which is usually 4.
    size : float or fontsize, optional
        Fontsize, defaults to mpl.rcParams['axes.labelsize'], usually
        'medium'.
    weight : string, optional
        Fontweight, defaults to mpl.rcParams['axes.labelweight'], usually
        'normal'.
    x : float, optional
        x position in axes coordinates. Default 0.5
    ax : matplotlib axis
        axis to add the label
    **kwargs : keyword arguments
        see matplotlib text help

    Returns
    -------
    h : handle
        text handle of the created text field

    ..note::
    http://stackoverflow.com/questions/35479508/cartopy-set-xlabel-set-ylabel-not-ticklabels

    """
    if ax is None:
        ax = plt.gca()
    
    labelpad, size, weight = _get_label_attr(labelpad, size, weight)
    
    va = kwargs.pop('va', 'top')
    ha = kwargs.pop('ha', 'center')
    rotation = kwargs.pop('rotation', 'horizontal')
    rotation_mode = kwargs.pop('rotation_mode', 'anchor')
    
    transform = kwargs.pop('transform', ax.transAxes)

    h = ax.annotate(s, xy=(x, 0), xycoords=transform,
                xytext=(0, -labelpad), textcoords='offset points',
                va=va, ha=ha, rotation=rotation,
                rotation_mode=rotation_mode,
                size=size, weight=weight,
                **kwargs)
    
    return h

# ----------------------------------------------------------------------


def yticklabels(y_ticks, labelpad=None, size=None, weight=None, ax=None,
                ha='right', va='center', bbox_props=dict(ec='none', fc='none'), **kwargs):
    
    """
    draw yticklabels on map plots - may or may not work
    
    Parameters
    ----------
    y_ticks : 1D array
        Position of the y_ticks.
    labelpad : float, optional
        Distance of labels to axes. Defaults to mpl.rcParams['axes.labelpad']
        which is usually 4.
    size : float or fontsize, optional
        Fontsize, defaults to mpl.rcParams['axes.labelsize'], usually
        'medium'.
    weight : string, optional
        Fontweight, defaults to mpl.rcParams['axes.labelweight'], usually
        'normal'.
    ax : GeoAxes, optional
        Axes to add the labels to. Default plt.gca(.
    ha : string
        Horizontal alignment, default: 'right'.
    va : string
        Vertical alignment, default: 'center'.
    bbox_props : dict
        Properties of the bounding box. Default: dict(ec='none', fc='none')
    kwargs : additional arguments
        Passed to ax.annotate
    
    """
    

    plt.draw()
    
    # get ax if necessary
    if ax is None:
        ax = plt.gca()

    labelpad, size, weight = _get_label_attr(labelpad, size, weight)
    
    boundary_pc = _get_boundary_platecarree(ax)

    # ensure labels are on rhs and not in the middle
    if len(boundary_pc) == 1:
        lonmin, lonmax = -180, 180
    else:
        lonmin, lonmax = 0, 360
    
    # get the y_limit
    y_lim = boundary_pc.bounds[1::2] 
    
    # remove all points not on map for labeling
    y_label_points = [y for y in y_ticks if y_lim[0] <= y <= y_lim[1]]

    # get a transform instance that mpl understands
    transform = ccrs.PlateCarree()._as_mpl_transform(ax)

    if np.isscalar(labelpad):
        labelpad = [labelpad, 0]
    
    # loop through points    
    for y in y_label_points:

        msg = LATITUDE_FORMATTER(y)
        
        x = _determine_intersection(boundary_pc, [lonmin, y], [lonmax, y])
    
        if x.size > 0:
            x = x[0, 0]                
            lp = labelpad[0] + labelpad[1] * np.abs(y) / 90
            
            h = ax.annotate(msg, xy=(x, y), xycoords=transform, ha=ha, va=va, size=size,
                            weight=weight, xytext=(-lp, 0), textcoords='offset points',
                            bbox=bbox_props, **kwargs)
    


def xticklabels(x_ticks, labelpad=None, size=None, weight=None, ax=None,
                ha='center', va='top', bbox_props=dict(ec='none', fc='none'), **kwargs):
    
    """
    draw xticklabels on map plots - may or may not work
    
    Parameters
    ----------
    x_ticks : 1D array
        Position of the x ticks.
    labelpad : float, optional
        Distance of labels to axes. Defaults to mpl.rcParams['axes.labelpad']
        which is usually 4.
    size : float or fontsize, optional
        Fontsize, defaults to mpl.rcParams['axes.labelsize'], usually
        'medium'.
    weight : string, optional
        Fontweight, defaults to mpl.rcParams['axes.labelweight'], usually
        'normal'.
    ax : GeoAxes, optional
        Axes to add the labels to. Default plt.gca(.
    ha : string
        Horizontal alignment, default: 'center'.
    va : string
        Vertical alignment, default: 'top'.
    bbox_props : dict
        Properties of the bounding box. Default: dict(ec='none', fc='none')
    kwargs : additional arguments
        Passed to ax.annotate
    
    """
    
    plt.draw()
    
    # get ax if necessary
    if ax is None:
        ax = plt.gca()

    labelpad, size, weight = _get_label_attr(labelpad, size, weight)
    
    boundary_pc = _get_boundary_platecarree(ax)
   
    # get the x_limit
    x_lim = boundary_pc.bounds[::2]     
    
    # remove all points not on map for labeling
    x_label_points = [x for x in x_ticks if x_lim[0] <= x <= x_lim[1]]

    # get a transform instance that mpl understands
    transform = ccrs.PlateCarree()._as_mpl_transform(ax)

    # loop through points
    for x in x_label_points:

        msg = LONGITUDE_FORMATTER(x)
        
        y = _determine_intersection(boundary_pc, [x, -90], [x, 90])
        if y.size > 0:
            y = y[0, 1]                
            
            h = ax.annotate(msg, xy=(x, y), xycoords=transform, ha=ha, va=va, size=size,
                            weight=weight, xytext=(0, -labelpad), textcoords='offset points',
                            bbox=bbox_props, **kwargs)
    
        

def _get_boundary_platecarree(ax):
    # get the bounding box of the map in lat/ lon coordinates
    # after ax._get_extent_geom
    proj = ccrs.PlateCarree()
    boundary_poly = sgeom.Polygon(ax.outline_patch.get_path().vertices)
    eroded_boundary = boundary_poly.buffer(-ax.projection.threshold)
    boundary_pc = proj.project_geometry(eroded_boundary, ax.projection)
    
    return boundary_pc

def _determine_intersection(polygon, xy1, xy2):

    p1 = sgeom.Point(xy1)
    p2 = sgeom.Point(xy2)
    ls = sgeom.LineString([p1, p2])

    return np.asarray(polygon.boundary.intersection(ls))






