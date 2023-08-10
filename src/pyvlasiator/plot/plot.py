import numpy as np
import math
import warnings
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

    if meta.ndims() == 3 or meta.maxamr > 0:
        # check if origin and normal exist in kwargs
        normal = kwargs["normal"] if "normal" in kwargs else 1
        origin = kwargs["origin"] if "origin" in kwargs else 0.0
        kwargs.pop("normal", None)
        kwargs.pop("origin", None)

        pArgs = set_args(meta, var, axisunit, normal, origin)
        data = prep2dslice(meta, var, normal, comp, pArgs)
    else:
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
            sizes = _getdim2d(ncells, maxamr, 2)
            plotrange = (coordmin[0], coordmax[0], coordmin[1], coordmax[1])
        else:
            dir = 1
            v2_ = 2
            sizes = _getdim2d(ncells, maxamr, 1)
            plotrange = (coordmin[0], coordmax[0], coordmin[2], coordmax[2])
    else:
        dir = 0
        v1_, v2_ = 1, 2
        sizes = _getdim2d(ncells, maxamr, 0)
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
            idlist, indexlist = meta.getslicecell(
                -coordmin[dir], dir, coordmin[dir], coordmax[dir]
            )
            v2D = data[:, indexlist]
            v1 = meta.refineslice(idlist, v2D[v1_, :], dir)
            v2 = meta.refineslice(idlist, v2D[v2_, :], dir)
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
    dir: int = -1,
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
    dir : int
        Normal direction of the 2D slice, 0 for x, 1 for y, and 2 for z.
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

    if dir == 0:
        seq = (1, 2)
    elif dir == 1 or (ncells[1] == 1 and ncells[2] != 1):  # polar
        seq = (0, 2)
        dir = 1
    elif dir == 2 or (ncells[2] == 1 and ncells[1] != 1):  # ecliptic
        seq = (0, 1)
        dir = 2
    else:
        raise ValueError("1D data detected. Please use 1D plot functions.")

    plotrange = (coordmin[seq[0]], coordmax[seq[0]], coordmin[seq[1]], coordmax[seq[1]])
    axislabels = tuple(("X", "Y", "Z")[i] for i in seq)
    # Scale the sizes to the highest refinement level for data to be refined later
    sizes = tuple(ncells[i] << meta.maxamr for i in seq)

    if dir == -1:
        idlist, indexlist = np.empty(0, dtype=int), np.empty(0, dtype=int)
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


def prep2dslice(meta: Vlsv, var: str, dir: int, comp: int, pArgs: PlotArgs):
    origin = pArgs.origin
    idlist = pArgs.idlist
    indexlist = pArgs.indexlist

    data3D = meta.read_variable(var)

    if var.startswith("fg_") or data3D.ndim > 2:  # field or derived quantities, fsgrid
        ncells = meta.ncells * 2**meta.maxamr
        if not dir in (0,1,2): 
            raise ValueError(f"Unknown normal direction {dir}")

        sliceratio = (origin - meta.coordmin[dir]) / (
            meta.coordmax[dir] - meta.coordmin[dir]
        )
        if not (0.0 <= sliceratio <= 1.0):
            raise ValueError("slice plane index out of bound!")
        # Find the cut plane index for each refinement level
        icut = int(np.floor(sliceratio * ncells[dir]))
        if dir == 0:
            if comp != -1:
                data = data3D[icut, :, :, comp]
            else:
                data = np.linalg.norm(data3D[icut, :, :, :], axis=3)
        elif dir == 1:
            if comp != -1:
                data = data3D[:, icut, :, comp]
            else:
                data = np.linalg.norm(data3D[:, icut, :, :], axis=3)
        elif dir == 2:
            if comp != -1:
                data = data3D[:, :, icut, comp]
            else:
                data = np.linalg.norm(data3D[:, :, icut, :], axis=3)
    else:  # moments, dccrg grid
        # vlasov grid, AMR
        if data3D.ndim == 1:
            data2D = data3D[indexlist]

            data = meta.refineslice(idlist, data2D, dir)
        elif data3D.ndim == 2:
            data2D = data3D[indexlist, :]

            if comp in (0, 1, 2):
                slice = data2D[:, comp]
                data = meta.refineslice(idlist, slice, dir)
            elif comp == -1:
                datax = meta.refineslice(idlist, data2D[:, 0], dir)
                datay = meta.refineslice(idlist, data2D[:, 1], dir)
                dataz = meta.refineslice(idlist, data2D[:, 2], dir)
                data = np.fromiter(
                    (np.linalg.norm([x, y, z]) for x, y, z in zip(datax, datay, dataz)),
                    dtype=float,
                )
            else:
                slice = data2D[:, comp]
                data = meta.refineslice(idlist, slice, dir)

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


def vdfslice(
    meta: Vlsv,
    location: tuple | list,
    ax=None,
    limits: tuple = (float("-inf"), float("inf"), float("-inf"), float("inf")),
    verbose: bool = False,
    species: str = "proton",
    unit: AxisUnit = AxisUnit.SI,
    unitv: str = "km/s",
    vmin: float = float("-inf"),
    vmax: float = float("inf"),
    slicetype: str = None,
    vslicethick: float = 0.0,
    center: str = None,
    weight: str = "particle",
    flimit: float = -1.0,
    **kwargs,
):
    v1, v2, r1, r2, weights, strx, stry, str_title = prep_vdf(
        meta,
        location,
        species,
        unit,
        unitv,
        slicetype,
        vslicethick,
        center,
        weight,
        flimit,
        verbose,
    )

    import matplotlib
    import matplotlib.pyplot as plt

    if math.isinf(vmin):
        vmin = np.min(weights)
    if math.isinf(vmax):
        vmax = np.max(weights)

    if verbose:
        print(f"Active f range is {vmin}, {vmax}")

    if not ax:
        ax = plt.gca()

    norm = matplotlib.colors.LogNorm(vmin, vmax)

    h = ax.hist2d(v1, v2, bins=(r1, r2), weights=weights, norm=norm, shading="flat")

    ax.set_title(str_title, fontweight="bold")
    ax.set_xlabel(strx)
    ax.set_ylabel(stry)
    ax.set_aspect("equal")
    ax.grid(color="grey", linestyle="-")
    ax.tick_params(direction="in")

    cb = plt.colorbar(h[3], ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(which="both", direction="in")
    cb_title = cb.ax.set_ylabel("f(v)")

    # TODO: Draw vector of magnetic field direction
    # if slicetype in ("xy", "xz", "yz"):

    return h[3]  # h[0] is 2D data, h[1] is x axis, h[2] is y axis


def prep_vdf(
    meta: Vlsv,
    location: tuple | list,
    species: str = "proton",
    unit: AxisUnit = AxisUnit.SI,
    unitv: str = "km/s",
    slicetype: str = None,
    vslicethick: float = 0.0,
    center: str = None,
    weight: str = "particle",
    flimit: float = -1.0,
    verbose: bool = False,
):
    ncells = meta.ncells

    if species in meta.meshes:
        vmesh = meta.meshes[species]
    else:
        raise ValueError(f"Unable to detect population {species}")

    if not slicetype in (None, "xy", "xz", "yz", "bperp", "bpar1", "bpar2"):
        raise ValueError(f"Unknown type {slicetype}")

    if unit == AxisUnit.EARTH:
        location = [loc * RE for loc in location]

    # Set unit conversion factor
    unitvfactor = 1e3 if unitv == "km/s" else 1.0

    # Get closest cell ID from input coordinates
    cidReq = meta.getcell(location)
    cidNearest = meta.getnearestcellwithvdf(cidReq)

    # Set normal direction
    if not slicetype:
        if ncells[1] == 1 and ncells[2] == 1:  # 1D, select xz
            slicetype = "xz"
        elif ncells[1] == 1:  # polar
            slicetype = "xz"
        elif ncells[2] == 1:  # ecliptic
            slicetype = "xy"
        else:
            slicetype = "xy"

    if slicetype in ("xy", "yz", "xz"):
        if slicetype == "xy":
            dir1, dir2, dir3 = 0, 1, 2
            ŝ = [0.0, 0.0, 1.0]
        elif slicetype == "xz":
            dir1, dir2, dir3 = 0, 2, 1
            ŝ = [0.0, 1.0, 0.0]
        elif slicetype == "yz":
            dir1, dir2, dir3 = 1, 2, 0
            ŝ = [1.0, 0.0, 0.0]
        v1size = vmesh.vblocks[dir1] * vmesh.vblock_size[dir1]
        v2size = vmesh.vblocks[dir2] * vmesh.vblock_size[dir2]

        v1min, v1max = vmesh.vmin[dir1], vmesh.vmax[dir1]
        v2min, v2max = vmesh.vmin[dir2], vmesh.vmax[dir2]
    elif slicetype in ("bperp", "bpar1", "bpar2"):
        # TODO: WIP
        pass

    if not math.isclose((v1max - v1min) / v1size, (v2max - v2min) / v2size):
        warnings.warn("Noncubic vgrid applied!")

    cellsize = (v1max - v1min) / v1size

    r1 = np.linspace(v1min / unitvfactor, v1max / unitvfactor, v1size + 1)
    r2 = np.linspace(v2min / unitvfactor, v2max / unitvfactor, v2size + 1)

    vcellids, vcellf = meta.read_vcells(cidNearest, species)

    V = meta.getvcellcoordinates(vcellids, species)

    if center:
        if center == "bulk":  # centered with bulk velocity
            if meta.has_variable("moments"):  # From a restart file
                Vcenter = meta.read_variable("restart_V", cidNearest)
            elif meta.has_variable(species * "/vg_v"):  # Vlasiator 5
                Vcenter = meta.read_variable(species * "/vg_v", cidNearest)
            elif meta.has_variable(species * "/V"):
                Vcenter = meta.read_variable(species * "/V", cidNearest)
            else:
                Vcenter = meta.read_variable("V", cidNearest)
        elif center == "peak":  # centered on highest VDF-value
            Vcenter = np.maximum(vcellf)

        V = np.array(
            [np.fromiter((v[i] - Vcenter for i in range(3)), dtype=float) for v in V]
        )

    # Set sparsity threshold
    if flimit < 0:
        if meta.has_variable(species + "/vg_effectivesparsitythreshold"):
            flimit = meta.readvariable(
                species + "/vg_effectivesparsitythreshold", cidNearest
            )
        elif meta.has_variable(species + "/EffectiveSparsityThreshold"):
            flimit = meta.read_variable(
                species + "/EffectiveSparsityThreshold", cidNearest
            )
        else:
            flimit = 1e-16

    # Drop velocity cells which are below the sparsity threshold
    findex_ = vcellf >= flimit
    fselect = vcellf[findex_]
    Vselect = V[findex_]

    if slicetype in ("xy", "yz", "xz"):
        v1select = np.array([v[dir1] for v in Vselect])
        v2select = np.array([v[dir2] for v in Vselect])
        vnormal = np.array([v[dir3] for v in Vselect])

        vec = ("vx", "vy", "vz")
        strx = vec[dir1]
        stry = vec[dir2]
    elif slicetype in ("bperp", "bpar1", "bpar2"):
        v1select = np.empty(len(Vselect), dtype=Vselect.dtype)
        v2select = np.empty(len(Vselect), dtype=Vselect.dtype)
        vnormal = np.empty(len(Vselect), dtype=Vselect.dtype)
        # TODO: WIP
        # for v1s, v2s, vn, vs in zip(v1select, v2select, vnormal, Vselect):
        #    v1s, v2s, vn = Rinv * Vselect

        # if slicetype == "bperp":
        #    strx = r"$v_{B \times V}$"
        #    stry = r"$v_{B \times (B \times V)}$"
        # elif slicetype == "bpar2":
        #    strx = r"$v_{B}$"
        #    stry = r"$v_{B \times V}$"
        # elif slicetype == "bpar1":
        #    strx = r"$v_{B \times (B \times V)}$"
        #    stry = r"$v_{B}$"

    unitstr = f" [{unitv}]"
    strx += unitstr
    stry += unitstr

    if vslicethick < 0:  # Set a proper thickness
        if any(
            i == 1.0 for i in ŝ
        ):  # Assure that the slice cut through at least 1 vcell
            vslicethick = cellsize
        else:  # Assume cubic vspace grid, add extra space
            vslicethick = cellsize * (math.sqrt(3) + 0.05)

    # Weights using particle flux or phase-space density
    fweight = (
        fselect * np.linalg.norm([v1select, v2select, vnormal])
        if weight == "flux"
        else fselect
    )

    # Select cells within the slice area
    if vslicethick > 0.0:
        ind_ = abs(vnormal) <= 0.5 * vslicethick
        v1, v2, fweight = v1select[ind_], v2select[ind_], fweight[ind_]
    else:
        v1, v2 = v1select, v2select

    v1 = [v / unitvfactor for v in v1]
    v2 = [v / unitvfactor for v in v2]

    str_title = f"t = {meta.time:4.1f}s"

    if verbose:
        print(f"Original coordinates : {location}")
        print(f"Original cell        : {meta.getcellcoordinates(cidReq)}")
        print(f"Nearest cell with VDF: {meta.getcellcoordinates(cidNearest)}")
        print(f"CellID: {cidNearest}")

        if center == "bulk":
            print(f"Transforming to plasma frame, travelling at speed {Vcenter}")
        elif center == "peak":
            print(f"Transforming to peak f-value frame, travelling at speed {Vcenter}")

        print(f"Using VDF threshold value of {flimit}.")

        if vslicethick > 0:
            print("Performing slice with a counting thickness of $vslicethick")
        else:
            print(f"Projecting total VDF to a single plane")

    return v1, v2, r1, r2, fweight, strx, stry, str_title


# Append plotting functions
Vlsv.plot = plot
Vlsv.pcolormesh = pcolormesh
Vlsv.contourf = contourf
Vlsv.contour = contour
Vlsv.streamplot = streamplot
Vlsv.vdfslice = vdfslice
