import numpy as np
import scipy as sp
import xarray as xr


def theil_ufunc(da, dim="time", alpha=0.1):
    """theil sen slope for xarray

    Wraps sp.stats.theilslopes in xr.apply_ufunc

    Parameters
    ==========
    da : xr.DataArray
        DataArray to calculate the theil sen slope over
    dim : list of str, optional
        Dimensions to reduce the array over. Default: "time"
    alpha : float, optional
        Significance level in [0, 0.5].

    Returns
    =======
    slope : xr.DataArray
        Median slope of the array
    significance : xr.DataArray
        Array indicating significance. True if significant,
        False otherwise
    """

    def theil_(pt, alpha):

        isnan = np.isnan(pt)
        # return NaN for all-nan vectors
        if isnan.all():
            return np.nan, np.nan

        # use masked-stats if any is nan
        if isnan.any():
            pt = np.ma.masked_invalid(pt)
            slope, inter, lo_slope, up_slope = sp.stats.mstats.theilslopes(
                pt, alpha=alpha
            )

        else:
            slope, inter, lo_slope, up_slope = sp.stats.theilslopes(pt, alpha=alpha)

        # theilslopes does not return siginficance but a
        # confidence interval assume it is significant
        # if both are on the same side of 0
        significance = np.sign(lo_slope) == np.sign(up_slope)

        return slope, significance

    kwargs = dict(alpha=alpha)
    dim = [dim] if isinstance(dim, str) else dim

    # use xr.apply_ufunc to handle vectorization
    theil_slope, theil_sign = xr.apply_ufunc(
        theil_,
        da,
        input_core_dims=[dim],
        vectorize=True,
        output_core_dims=((), ()),
        kwargs=kwargs,
    )

    return theil_slope, theil_sign
