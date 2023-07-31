import numpy as np
from typing import Callable
from pyvlasiator.vlsv import Vlsv


def plot(
    self: Vlsv,
    var: str = "",
    ax=None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
):
    """
    Plot 1d data.

    Parameters
    ----------
    var : str
        Variable name from the VLSV file.

    Returns
    -------
    [matplotlib.lines.Line2D]
        A list of Line2D.
    """
    import matplotlib.pyplot as plt

    fig = kwargs.pop(
        "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
    )
    if ax is None:
        ax = fig.gca()

    if not self.has_variable(var):
        raise ValueError(f"Variable {var} not found in the file")

    x = np.linspace(self.coordmin[0], self.coordmax[0], self.ncells[0])

    data = self.read_variable(var)
    axes = ax.plot(x, data)

    return axes


def pcolormesh(self: Vlsv):
    #TODO: WIP
    return self._plot2d(
        pcolormesh,
        var,
        ax,
        comp,
        axisunit,
        colorscale,
        addcolorbar,
        vmin,
        vmax,
        extent,
        **kwargs,
    )


def _plot2d(
    self: Vlsv,
    f: Callable,
    meta,
    var: str = "",
    axisunit: str = "EARTH",
    extent: list = [0.0, 0.0, 0.0, 0.0],
    ax=None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
):
    """
    Plot 2d data.

    Parameters
    ----------
    var : str
        Variable name from the VLSV file.

    Returns
    -------

    """
    #TODO: WIP
    import matplotlib.pyplot as plt

    fig = kwargs.pop(
        "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
    )
    if ax is None:
        ax = fig.gca()

    if not meta.has_variable(var):
        raise ValueError(f"Variable {var} not found in the file")

    pArgs = set_args(meta, var, axisunit)
    data = prep2d(meta, var, comp)

    x1, x2 = get_axis(pArgs)

    if var in ("fg_b", "fg_e", "vg_b_vol", "vg_e_vol") or var.endswith("vg_v"):
        _fillinnerBC(data)

    norm, ticks = set_colorbar(colorscale, vmin, vmax, data)

    range1 = range(
        np.searchsorted(x1, extent[0]), np.searchsorted(x1, extent[1], side="right")
    )
    range2 = range(
        np.searchsorted(x2, extent[2]), np.searchsorted(x2, extent[3], side="right")
    )

    c = f(x1, x2, data, **kwargs)

    return c


Vlsv.plot = plot
