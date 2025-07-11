import pytest
import requests
import tarfile
import os
import numpy as np
from pyvlasiator.vlsv import Vlsv
import pyvlasiator.plot
import matplotlib as mpl
import packaging.version

filedir = os.path.dirname(__file__)

if os.path.isfile(filedir + "/data/bulk.1d.vlsv"):
    pass
else:
    url = (
        "https://raw.githubusercontent.com/henry2004y/vlsv_data/master/testdata.tar.gz"
    )
    testfiles = url.rsplit("/", 1)[1]
    r = requests.get(url, allow_redirects=True)
    open(testfiles, "wb").write(r.content)

    path = filedir + "/data"

    if not os.path.exists(path):
        os.makedirs(path)

    with tarfile.open(testfiles) as file:
        file.extractall(path)
    os.remove(testfiles)


class TestVlsv:
    dir = "tests/data/"
    files = (dir + "bulk.1d.vlsv", dir + "bulk.2d.vlsv", dir + "bulk.amr.vlsv")

    def test_load(self):
        meta = Vlsv(self.files[0])
        assert meta.__repr__().startswith("File")
        assert meta.time == 10.0

    def test_load_error(self):
        with pytest.raises(FileNotFoundError):
            meta = Vlsv("None")

    def test_read_variable(self):
        meta = Vlsv(self.files[0])
        assert np.array_equal(meta.cellindex, np.arange(9, -1, -1, dtype=np.uint64))
        # unsorted ID
        cellid = meta.read_variable("CellID", sorted=False)
        assert np.array_equal(cellid, np.arange(10, 0, -1, dtype=np.uint64))
        # sorted var by default
        data = meta.read_variable("vg_boundarytype")
        assert np.array_equal(data, np.array([4, 4, 1, 1, 1, 1, 1, 1, 3, 3]))
        # ID finding (noAMR)
        loc = [2.0, 0.0, 0.0]
        id = meta.getcell(loc)
        coords = meta.getcellcoordinates(id)
        assert coords == pytest.approx([3.0, 0.0, 0.0])
        assert meta.read_variable("proton/vg_rho", id) == pytest.approx(1.2288102e0)
        assert meta.read_variable("proton/vg_rho", [id]) == pytest.approx(1.2288102e0)
        # Nearest ID with VDF stored
        assert meta.getnearestcellwithvdf(id) == 5

    def test_read_vspace(self):
        meta = Vlsv(self.files[0])
        vcellids, vcellf = meta.read_vcells(5)
        V = meta.getvcellcoordinates(vcellids)
        assert V[-1] == pytest.approx((2.45, 1.95, 1.95))

    def test_read_amr(self):
        metaAMR = Vlsv(self.files[2])
        assert metaAMR.maxamr == 2
        # AMR data reading, DCCRG grid
        sliceoffset = abs(metaAMR.coordmin[1])
        idlist, indexlist = metaAMR.getslicecell(
            sliceoffset, 1, metaAMR.coordmin[1], metaAMR.coordmax[1]
        )

        # ID finding (AMR)
        loc = [2.5e6, 2.5e6, 2.5e6]  # exact cell center
        id = metaAMR.getcell(loc)
        assert metaAMR.getcellcoordinates(id) == pytest.approx(loc)

        data = metaAMR.read_variable("proton/vg_rho")
        dataslice = metaAMR.refineslice(idlist, data[indexlist], 1)
        assert np.sum(dataslice) == pytest.approx(7.6903526e8)

    def test_read_fg_variable(self):
        metaAMR = Vlsv(self.files[2])
        data = metaAMR.read_variable("fg_e")
        ncells, namr = metaAMR.ncells, metaAMR.maxamr
        assert data.shape == (
            ncells[0] * namr**2,
            ncells[1] * namr**2,
            ncells[2] * namr**2,
            3,
        ) and data[4, 0, 0, :] == pytest.approx([7.603512e-07, 2e-04, -2e-04])


class TestPlot:
    dir = "tests/data/"
    files = (dir + "bulk.1d.vlsv", dir + "bulk.2d.vlsv", dir + "bulk.amr.vlsv")
    mpl.use("Agg")
    version_str = mpl.__version__
    version = packaging.version.parse(version_str)

    def test_1d_plot(self):
        meta = Vlsv(self.files[0])
        v = meta.plot("proton/vg_rho")[0].get_ydata()
        assert np.array_equal(v, meta.read_variable("proton/vg_rho"))
        v = meta.scatter("proton/vg_rho").get_offsets()
        assert np.array_equal(v[0].data[0], -10.0)

    def test_2d_plot(self):
        meta = Vlsv(self.files[1])
        v = meta.pcolormesh("proton/vg_rho").get_array()
        assert v[99, 60] == pytest.approx(999535.8) and v.data.size == 6300
        v = meta.pcolormesh(
            "proton/vg_rho", axisunit=pyvlasiator.plot.AxisUnit.SI
        ).get_array()
        assert v[99, 60] == pytest.approx(999535.8) and v.data.size == 6300
        v = meta.contour("proton/vg_rho").get_array()
        assert v[-3] == 4000000.0
        v = meta.contourf("proton/vg_rho").get_array()
        assert v[-3] == 3600000.0
        v = meta.contourf(
            "proton/vg_rho", colorscale=pyvlasiator.plot.ColorScale.Log
        ).get_array()
        assert v[-3] == 3600000.0
        v = meta.pcolormesh("vg_b_vol").get_array()
        assert v[0, 1] == pytest.approx(3.0051286e-09)
        v = meta.pcolormesh(
            "vg_b_vol", comp=1, colorscale=pyvlasiator.plot.ColorScale.SymLog
        ).get_array()
        assert v[0, 1] == pytest.approx(-9.284285146238247e-12)
        v = meta.pcolormesh("fg_b", comp=0).get_array()
        assert v[99, 60] == pytest.approx(-2.999047e-09) and v.data.size == 6300

    def test_3d_amr_slice(self):
        meta = Vlsv(self.files[2])
        v = meta.pcolormesh("proton/vg_rho").get_array()
        assert v[15, 31] == pytest.approx(1.0483886e6) and v.data.size == 512

    def test_stream_plot(self):
        meta = Vlsv(self.files[1])
        p = meta.streamplot("proton/vg_v", comp="xy")
        assert type(p) == mpl.streamplot.StreamplotSet

    def test_vdf_plot(self):
        meta = Vlsv(self.files[0])
        loc = [2.0, 0.0, 0.0]
        v = meta.vdfslice(loc, verbose=True).get_array()
        assert v[19, 25] == 238.24398578141802


def load(files):
    """
    Benchmarking VLSV loading.
    """
    meta = Vlsv(files[0])
    meta = Vlsv(files[1])
    meta = Vlsv(files[2])
    return meta


def test_load(benchmark):
    dir = "tests/data/"
    files = (dir + "bulk.1d.vlsv", dir + "bulk.2d.vlsv", dir + "bulk.amr.vlsv")
    result = benchmark(load, files)

    assert type(result) == Vlsv
