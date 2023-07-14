import pytest

from pyvlasiator.vlsv.reader import VlsvReader


class TestVlsvReader:
    file = "tests/data/bulk.1d.vlsv"

    def test_load(self):
        meta = VlsvReader(self.file)
        assert meta.__repr__().startswith("File")
        assert meta.time == 10.0

    def test_load_error(self):
        with pytest.raises(FileNotFoundError):
            meta = VlsvReader("None")

    def test_variable(self):
        meta = VlsvReader(self.file)
        n = meta.read_variable("proton/vg_rho")
        assert len(n) == 10 and n[0] == pytest.approx(1.0214901)
