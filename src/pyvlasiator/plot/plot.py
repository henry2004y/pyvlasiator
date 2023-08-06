import numpy as np
import math
from typing import Callable
from collections import namedtuple
from enum import Enum
from pyvlasiator.vlsv import Vlsv
from pyvlasiator.vlsv.reader import _getdim2d
from pyvlasiator.vlsv.variables import RE


class ColorScale(Enum):
    Linear = 1
    Log = 2
    SymLog = 3


class AxisUnit(Enum):
    EARTH = 1
    SI = 2


# Plotting arguments
PlotArgs = namedtuple(
    "PlotArgs",
    [
        "axisunit",
        "sizes",
        "plotrange",
        "origin",
        "idlist",
        "indexlist",
        "str_title",
        "strx",
        "stry",
        "cb_title",
    ],
)


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
    axes = ax.plot(x, data, **kwargs)

    return axes


def pcolormesh(
    self: Vlsv,
    var: str = "",
    axisunit: AxisUnit = AxisUnit.EARTH,
    colorscale: ColorScale = ColorScale.Linear,
    addcolorbar: bool = True,
    vmin: float = float("-inf"),
    vmax: float = float("inf"),
    extent: list = [0.0, 0.0, 0.0, 0.0],
    comp: int = -1,
    ax=None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
):
    fig, ax = set_figure(ax, figsize, **kwargs)

    return _plot2d(
        self,
        ax.pcolormesh,
        var=var,
        ax=ax,
        comp=comp,
        axisunit=axisunit,
        colorscale=colorscale,
        addcolorbar=addcolorbar,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        **kwargs,
    )


def contourf(
    self: Vlsv,
    var: str = "",
    axisunit: AxisUnit = AxisUnit.EARTH,
    colorscale: ColorScale = ColorScale.Linear,
    addcolorbar: bool = True,
    vmin: float = float("-inf"),
    vmax: float = float("inf"),
    extent: list = [0.0, 0.0, 0.0, 0.0],
    comp: int = -1,
    ax=None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
):
    fig, ax = set_figure(ax, figsize, **kwargs)

    return _plot2d(
        self,
        ax.contourf,
        var=var,
        ax=ax,
        comp=comp,
        axisunit=axisunit,
        colorscale=colorscale,
        addcolorbar=addcolorbar,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        **kwargs,
    )


def contour(
    self: Vlsv,
    var: str = "",
    axisunit: AxisUnit = AxisUnit.EARTH,
    colorscale: ColorScale = ColorScale.Linear,
    addcolorbar: bool = True,
    vmin: float = float("-inf"),
    vmax: float = float("inf"),
    extent: list = [0.0, 0.0, 0.0, 0.0],
    comp: int = -1,
    ax=None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
):
    fig, ax = set_figure(ax, figsize, **kwargs)

    return _plot2d(
        self,
        ax.contour,
        var=var,
        ax=ax,
        comp=comp,
        axisunit=axisunit,
        colorscale=colorscale,
        addcolorbar=addcolorbar,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        **kwargs,
    )


def _plot2d(
    meta: Vlsv,
    f: Callable,
    var: str = "",
    axisunit: AxisUnit = AxisUnit.EARTH,
    colorscale: ColorScale = ColorScale.Linear,
    addcolorbar: bool = True,
    vmin: float = float("-inf"),
    vmax: float = float("inf"),
    extent: list = [0.0, 0.0, 0.0, 0.0],
    comp: int = -1,
    ax=None,
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
    if not meta.has_variable(var):
        raise ValueError(f"Variable {var} not found in the file")

    pArgs = set_args(meta, var, axisunit)
    data = prep2d(meta, var, comp)

    x1, x2 = get_axis(pArgs.axisunit, pArgs.plotrange, pArgs.sizes)

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

    set_plot(c, ax, pArgs, ticks, addcolorbar)

    return c


def streamplot(
    meta: Vlsv,
    var: str,
    ax=None,
    comp: str = "xy",
    axisunit: AxisUnit = AxisUnit.EARTH,
    **kwargs,
):
    X, Y, v1, v2 = set_vector(meta, var, comp, axisunit)
    fig, ax = set_figure(ax, **kwargs)

    s = ax.streamplot(X, Y, v1, v2, **kwargs)

    return s


def set_vector(meta: Vlsv, var: str, comp: str, axisunit: AxisUnit):
    ncells = meta.ncells
    maxamr = meta.maxamr
    coordmin, coordmax = meta.coordmin, meta.coordmax

    if "x" in comp:
        v1_ = 0
        if "y" in comp:
            dir = 2
            v2_ = 1
            sizes = _getdim2d(ncells, maxamr, "z")
            plotrange = (coordmin[0], coordmax[0], coordmin[1], coordmax[1])
        else:
            dir = 1
            v2_ = 2
            sizes = _getdim2d(ncells, maxamr, "y")
            plotrange = (coordmin[0], coordmax[0], coordmin[2], coordmax[2])
    else:
        dir = 0
        v1_, v2_ = 1, 2
        sizes = _getdim2d(ncells, maxamr, "x")
        plotrange = (coordmin[1], coordmax[1], coordmin[2], coordmax[2])

    data = meta.read_variable(var)

    if not var.startswith("fg_"):  # vlasov grid
        if data.ndim != 2 and data.shape[0] == 3:
            raise ValueError("Vector variable required!")
        if meta.maxamr == 0:
            data = data.reshape((sizes[1], sizes[0], 3))
            v1 = data[:, :, v1_]
            v2 = data[:, :, v2_]
        else:
            idlist, indexlist = getslicecell(
                meta, -coordmin[dir], dir, coordmin[dir], coordmax[dir]
            )
            v2D = data[:, indexlist]
            v1 = refineslice(meta, idlist, v2D[v1_, :], dir)
            v2 = refineslice(meta, idlist, v2D[v2_, :], dir)
    else:
        v1 = data[:, :, v1_]
        v2 = data[:, :, v2_]

    x, y = get_axis(axisunit, plotrange, sizes)

    return x, y, v1, v2


def set_figure(ax, figsize: tuple = (10, 6), **kwargs):
    import matplotlib.pyplot as plt

    fig = kwargs.pop(
        "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
    )
    if ax is None:
        ax = fig.gca()

    return fig, ax


def set_args(
    meta: Vlsv,
    var: str,
    axisunit: AxisUnit = AxisUnit.EARTH,
    normal: str = "",
    origin: float = 0.0,
) -> PlotArgs:
    """
    Set plot-related arguments of `var` in `axisunit`.

    Parameters
    ----------
    var : str
        Variable name from the VLSV file.
    axisunit : AxisUnit
        Unit of the axis.
    normal : str
        Normal direction of the 2D slice, "x", "y", or "z".
    origin : float
        Origin of the 2D slice.

    Returns
    -------
    PlotArgs

    See Also
    --------
    :func:`pcolormesh`
    """
    ncells, coordmin, coordmax = meta.ncells, meta.coordmin, meta.coordmax

    if normal == "x":
        seq = (1, 2)
        dir = 0
    elif normal == "y" or (ncells[1] == 1 and ncells[2] != 1):  # polar
        seq = (0, 2)
        dir = 1
    elif normal == "z" or (ncells[2] == 1 and ncells[1] != 1):  # ecliptic
        seq = (0, 1)
        dir = 2
    else:
        raise ValueError("1D data detected. Please use 1D plot functions.")

    plotrange = (coordmin[seq[0]], coordmax[seq[0]], coordmin[seq[1]], coordmax[seq[1]])
    axislabels = tuple(("X", "Y", "Z")[i] for i in seq)
    # Scale the sizes to the highest refinement level for data to be refined later
    sizes = tuple(ncells[i] << meta.maxamr for i in seq)

    if not normal:
        idlist, indexlist = [], []
    else:
        sliceoffset = origin - coordmin[dir]
        idlist, indexlist = meta.getslicecell(
            sliceoffset, dir, coordmin[dir], coordmax[dir]
        )

    if axisunit == AxisUnit.EARTH:
        unitstr = r"$R_E$"
    else:
        unitstr = r"$m$"
    strx = axislabels[0] + " [" + unitstr + "]"
    stry = axislabels[1] + " [" + unitstr + "]"

    str_title = f"t={meta.time:4.1f}s"

    datainfo = meta.read_variable_meta(var)

    if not datainfo.variableLaTeX:
        cb_title = datainfo.variableLaTeX + " [" + datainfo.unitLaTeX + "]"
    else:
        cb_title = ""

    return PlotArgs(
        axisunit,
        sizes,
        plotrange,
        origin,
        idlist,
        indexlist,
        str_title,
        strx,
        stry,
        cb_title,
    )


def prep2d(meta: Vlsv, var: str, comp: int = -1):
    """
    Obtain data of `var` for 2D plotting. Use `comp` to select vector components.

    Parameters
        ----------
        meta : Vlsv
            Metadata corresponding to the file.
        var : str
            Name of the variable.
        comp : int
            Vector component. -1 refers to the magnitude of the vector.
        Returns
        -------
        numpy.ndarray
    """
    dataRaw = _getdata2d(meta, var)

    if dataRaw.ndim == 3:
        if comp != -1:
            data = dataRaw[:, :, comp]
        else:
            data = np.linalg.norm(dataRaw, axis=2)
    else:
        data = dataRaw

    return data


def _getdata2d(meta: Vlsv, var: str):
    if meta.ndims() != 2:
        raise ValueError("2D outputs required")
    sizes = [i for i in meta.ncells if i != 1]
    data = meta.read_variable(var)
    if data.ndim == 1 or data.shape[-1] == 1:
        data = data.reshape((sizes[1], sizes[0]))
    else:
        data = data.reshape((sizes[1], sizes[0], 3))

    return data


def get_axis(axisunit: AxisUnit, plotrange: tuple, sizes: tuple):
    """
    Get the 2D domain axis.
    """
    if axisunit == AxisUnit.EARTH:
        x = np.linspace(plotrange[0] / RE, plotrange[1] / RE, sizes[0])
        y = np.linspace(plotrange[2] / RE, plotrange[3] / RE, sizes[1])
    else:
        x = np.linspace(plotrange[0], plotrange[1], sizes[0])
        y = np.linspace(plotrange[2], plotrange[3], sizes[1])

    return x, y


def _fillinnerBC(data: np.ndarray):
    """
    Fill sparsity/inner boundary cells with NaN.
    """
    data[data == 0] = np.nan


def set_colorbar(
    colorscale: ColorScale = ColorScale.Linear,
    v1: float = np.nan,
    v2: float = np.nan,
    data: np.ndarray = np.array([1.0]),
    linthresh: float = 1.0,
    logstep: float = 1.0,
    linscale: float = 0.03,
):
    import matplotlib

    vmin, vmax = set_lim(v1, v2, data, colorscale)
    if colorscale == ColorScale.Linear:
        levels = matplotlib.ticker.MaxNLocator(nbins=255).tick_values(vmin, vmax)
        norm = matplotlib.colors.BoundaryNorm(levels, ncolors=256, clip=True)
        ticks = matplotlib.ticker.LinearLocator(numticks=9)
    elif colorscale == ColorScale.Log:  # logarithmic
        norm = matplotlib.colors.LogNorm(vmin, vmax)
        ticks = matplotlib.ticker.LogLocator(base=10, subs=range(0, 9))
    else:  # symmetric log
        logthresh = int(math.floor(math.log10(linthresh)))
        minlog = int(math.ceil(math.log10(-vmin)))
        maxlog = int(math.ceil(math.log10(vmax)))
        # TODO: fix this!
        # norm = matplotlib.colors.SymLogNorm(linthresh, linscale, vmin, vmax, base=10)
        # ticks = [ [-(10.0**x) for x in minlog:-logstep:logthresh]..., 0.0,
        #   [10.0**x for x in logthresh:logstep:maxlog]..., ]

    return norm, ticks


def set_lim(vmin: float, vmax: float, data, colorscale: ColorScale = ColorScale.Linear):
    if colorscale in (ColorScale.Linear, ColorScale.SymLog):
        if math.isinf(vmin):
            v1 = np.nanmin(data)
        else:
            v1 = vmin
        if math.isinf(vmax):
            v2 = np.nanmax(data)
        else:
            v2 = vmax
    else:  # logarithmic
        datapositive = data[data > 0.0]
        if math.isinf(vmin):
            v1 = np.minimum(datapositive)
        else:
            v1 = vmin
        if math.isinf(vmax):
            v2 = np.nanmax(data)
        else:
            v2 = vmax

    return v1, v2


def set_plot(c, ax, pArgs: PlotArgs, ticks, addcolorbar: bool):
    """
    Configure customized plot.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    str_title = pArgs.str_title
    strx = pArgs.strx
    stry = pArgs.stry
    cb_title = pArgs.cb_title

    if addcolorbar:
        cb = plt.colorbar(c, ax=ax, ticks=ticks, fraction=0.04, pad=0.02)
        if not cb_title:
            cb.ax.set_ylabel(cb_title)
        cb.ax.tick_params(direction="in")

    ax.set_title(str_title, fontweight="bold")
    ax.set_xlabel(strx)
    ax.set_ylabel(stry)
    ax.set_aspect("equal")

    # Set border line widths
    for loc in ("left", "bottom", "right", "top"):
        ax.spines[loc].set_linewidth(2.0)

    ax.xaxis.set_tick_params(width=2.0, length=3)
    ax.yaxis.set_tick_params(width=2.0, length=3)


# Append plotting functions
Vlsv.plot = plot
Vlsv.pcolormesh = pcolormesh
Vlsv.contourf = contourf
Vlsv.contour = contour
Vlsv.streamplot = streamplot