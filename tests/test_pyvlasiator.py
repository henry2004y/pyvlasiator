import pytest
import requests
import tarfile
import os
import numpy as np
from pyvlasiator.vlsv.reader import VlsvReader

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


class TestVlsvReader:
    dir = "tests/data/"
    files = (dir + "bulk.1d.vlsv", dir + "bulk.1d.vlsv", dir + "bulk.amr.vlsv")

    def test_load(self):
        meta = VlsvReader(self.files[0])
        assert meta.__repr__().startswith("File")
        assert meta.time == 10.0

    def test_load_error(self):
        with pytest.raises(FileNotFoundError):
            meta = VlsvReader("None")

    def test_read_variable(self):
        meta = VlsvReader(self.files[0])
        assert np.array_equal(meta.cellindex, np.arange(9, -1, -1, dtype=np.uint64))
        # unsorted ID
        cellid = meta.read_variable("CellID", sorted=False)
        assert np.array_equal(cellid, np.arange(10, 0, -1, dtype=np.uint64))
        # sorted var by default
        data = meta.read_variable("vg_boundarytype")
        assert np.array_equal(data, np.array([4, 4, 1, 1, 1, 1, 1, 1, 3, 3]))
        # ID finding (noAMR)
        # loc = [2.0, 0.0, 0.0]
        # id = getcell(meta, loc)
        # coords = getcellcoordinates(meta, id)
        # assert coords == [3.0, 0.0, 0.0]
        # assert meta.read_variable("proton/vg_rho", id)[1] == 1.2288102e0

    def test_read_vspace(self):
        meta = VlsvReader(self.files[0])
        vcellids, vcellf = meta.read_vcells(5)
        V = meta.getvcellcoordinates(vcellids)
        assert V[-1] == pytest.approx((2.45, 1.95, 1.95))

    def test_read_amr(self):
        metaAMR = VlsvReader(self.files[2])
        assert metaAMR.maxamr == 2

    def test_read_fg_variable(self):
        metaAMR = VlsvReader(self.files[2])
        data = metaAMR.read_variable("fg_e")
        ncells, namr = metaAMR.ncells, metaAMR.maxamr
        assert data.shape == (
            ncells[0] * namr**2,
            ncells[1] * namr**2,
            ncells[2] * namr**2,
            3,
        ) and data[0, 0, 4, :] == pytest.approx(
            [7.603512e-07, 2.000000e-04, -2.000000e-04]
        )
