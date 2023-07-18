import pytest
import requests
import tarfile
import os
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
    file = "tests/data/bulk.1d.vlsv"

    def test_load(self):
        meta = VlsvReader(self.file)
        assert meta.__repr__().startswith("File")
        assert meta.time == 10.0

    def test_load_error(self):
        with pytest.raises(FileNotFoundError):
            meta = VlsvReader("None")

    def test_read_variable(self):
        meta = VlsvReader(self.file)
        n = meta.read_variable("proton/vg_rho")
        assert len(n) == 10 and n[0] == pytest.approx(1.0214901)

    def test_read_vspace(self):
        meta = VlsvReader(self.file)
        vcellids, vcellf = meta.read_vcells(5)
        V = meta.getvcellcoordinates(vcellids)
        assert V[-1] == pytest.approx((2.45, 1.95, 1.95))
