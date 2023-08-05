from xml.etree import ElementTree
import numpy as np
import os
from collections import namedtuple
from pyvlasiator.vlsv.variables import units_predefined


class VMeshInfo:
    def __init__(self, vblocks, vblock_size, vmin, vmax, dv):
        self.vblocks = vblocks
        self.vblock_size = vblock_size
        self.vmin = vmin
        self.vmax = vmax
        self.dv = dv
        self.cellwithVDF = np.empty(0, dtype=np.uint64)
        self.nblock_C = np.empty(0, dtype=np.int64)


"Variable information from the VLSV footer."
VarInfo = namedtuple(
    "VarInfo", ["unit", "unitLaTeX", "variableLaTeX", "unitConversion"]
)


class Vlsv:
    def __init__(self, filename: str):
        self.dir, self.name = os.path.split(filename)
        self.fid = open(filename, "rb")
        self.xmlroot = ElementTree.fromstring("<VLSV></VLSV>")
        self.celldict = {}
        self.maxamr = -1
        self.vg_indexes_on_fg = np.array([])  # SEE: map_vg_onto_fg(self)

        self._read_xml_footer()

        if self.has_parameter(name="time"):  # Vlasiator 5.0+
            self.time = self.read(name="time", tag="PARAMETER")
        elif self.has_parameter(name="t"):
            self.time = self.read(name="t", tag="PARAMETER")
        else:
            self.time = -1.0

        # Check if the file is using new or old VLSV format
        # Read parameters
        meshName = "SpatialGrid"
        bbox = self.read(tag="MESH_BBOX", mesh=meshName)
        if bbox is None:
            try:
                # Vlasiator 4- files where the mesh is defined with parameters
                self.ncells = (
                    (int)(self.read_parameter("xcells_ini")),
                    (int)(self.read_parameter("ycells_ini")),
                    (int)(self.read_parameter("zcells_ini")),
                )
                self.block_size = (1, 1, 1)
                self.coordmin = (
                    self.read_parameter("xmin"),
                    self.read_parameter("ymin"),
                    self.read_parameter("zmin"),
                )
                self.coordmax = (
                    self.read_parameter("xmax"),
                    self.read_parameter("ymax"),
                    self.read_parameter("zmax"),
                )
            except:  # dummy values
                self.ncells = (1, 1, 1)
                self.block_size = (1, 1, 1)
                self.coordmin = (0.0, 0.0, 0.0)
                self.coordmax = (1.0, 1.0, 1.0)

        else:
            # Vlasiator 5+ file
            nodeX = self.read(tag="MESH_NODE_CRDS_X", mesh=meshName)
            nodeY = self.read(tag="MESH_NODE_CRDS_Y", mesh=meshName)
            nodeZ = self.read(tag="MESH_NODE_CRDS_Z", mesh=meshName)
            self.ncells = tuple(i.item() for i in bbox[0:3])
            self.block_size = tuple(i.item() for i in bbox[3:6])
            self.coordmin = (nodeX[0], nodeY[0], nodeZ[0])
            self.coordmax = (nodeX[-1], nodeY[-1], nodeZ[-1])

        self.dcoord = tuple(
            map(
                lambda i: (self.coordmax[i] - self.coordmin[i]) / self.ncells[i],
                range(0, 3),
            )
        )
        # TODO: use TypedDict
        self.meshes = {}

        # Iterate through the XML tree, find all populations
        self.species = []

        for child in self.xmlroot.findall("BLOCKIDS"):
            if "name" in child.attrib:
                popname = child.attrib["name"]
            else:
                popname = "avgs"

            if not popname in self.species:
                self.species.append(popname)

            bbox = self.read(tag="MESH_BBOX", mesh=popname)
            if bbox is None:
                if self.read_parameter("vxblocks_ini") is not None:
                    # Vlasiator 4- files where the mesh is defined with parameters
                    vblocks = (
                        (int)(self.read_parameter("vxblocks_ini")),
                        (int)(self.read_parameter("vyblocks_ini")),
                        (int)(self.read_parameter("vzblocks_ini")),
                    )
                    vblock_size = (4, 4, 4)
                    vmin = (
                        self.read_parameter("vxmin"),
                        self.read_parameter("vymin"),
                        self.read_parameter("vzmin"),
                    )
                    vmax = (
                        self.read_parameter("vxmax"),
                        self.read_parameter("vymax"),
                        self.read_parameter("vzmax"),
                    )

                else:  # no velocity space
                    vblocks = (0, 0, 0)
                    vblock_size = (4, 4, 4)
                    vmin = (0.0, 0.0, 0.0)
                    vmax = (0.0, 0.0, 0.0)
                    dv = (1.0, 1.0, 1.0)

            else:  # Vlasiator 5+ file with bounding box
                nodeX = self.read(tag="MESH_NODE_CRDS_X", mesh=popname)
                nodeY = self.read(tag="MESH_NODE_CRDS_Y", mesh=popname)
                nodeZ = self.read(tag="MESH_NODE_CRDS_Z", mesh=popname)
                vblocks = (*bbox[0:3],)
                vblock_size = (*bbox[3:6],)
                vmin = (nodeX[0], nodeY[0], nodeZ[0])
                vmax = (nodeX[-1], nodeY[-1], nodeZ[-1])

            dv = tuple(
                map(
                    lambda i: (vmax[i] - vmin[i]) / vblocks[i] / vblock_size[i],
                    range(0, 3),
                )
            )

            self.meshes[popname] = VMeshInfo(vblocks, vblock_size, vmin, vmax, dv)

            # Precipitation energy bins
            i = 0
            energybins = []
            binexists = True
            while binexists:
                binexists = self.has_parameter(
                    f"{popname}_PrecipitationCentreEnergy{i}"
                )
                if binexists:
                    binvalue = self.read_parameter(
                        f"{popname}_PrecipitationCentreEnergy{i}"
                    )
                    energybins.append(binvalue)
                i += 1
            if i > 1:
                self.precipitationenergybins[popname] = energybins

        self.variable = [
            node.attrib["name"] for node in self.xmlroot.findall("VARIABLE")
        ]

        cellid = self.read(mesh="SpatialGrid", name="CellID", tag="VARIABLE")
        self.cellindex = np.argsort(cellid)
        self.celldict = {cid: i for (i, cid) in enumerate(cellid)}

        self.maxamr = self.getmaxrefinement(cellid)

        self.nodecellwithVDF = self.xmlroot.findall("CELLSWITHBLOCKS")

        if len(self.nodecellwithVDF) == 0:
            self.hasvdf = False
        else:
            self.hasvdf = self.nodecellwithVDF[0].attrib["arraysize"] != "0"

    def __repr__(self) -> str:
        str = (
            f"File       : {self.name}\n"
            f"Time       : {self.time:.4f}\n"
            f"Dimension  : {self.ndims()}\n"
            f"Max AMR lvl: {self.maxamr}\n"
            f"Has VDF    : {self.hasvdf}\n"
            f"Variables  : {self.variable}\n"
        )
        return str

    def ndims(self) -> int:
        """Get the spatial dimension of data."""
        return sum(i > 1 for i in self.ncells)

    def _read_xml_footer(self):
        """Read the XML footer of the VLSV file."""
        fid = self.fid
        # first 8 bytes indicate endianness
        endianness_offset = 8
        fid.seek(endianness_offset)
        # offset of the XML file
        uint64_byte_amount = 8
        offset = int.from_bytes(fid.read(uint64_byte_amount), "little", signed=True)
        fid.seek(offset)
        xmlstring = fid.read()
        self.xmlroot = ElementTree.fromstring(xmlstring)

    def read(
        self,
        name: str = "",
        tag: str = "",
        mesh: str = "",
        cellids=-1,
    ) -> np.ndarray:
        """
        Read data of name, tag, and mesh from the vlsv file.

        This is the general reading function for all types of variables in VLSV files.

        Parameters
        ----------
        cellids : int or list of int
            If -1 then all data is read. If nonzero then only the vector for the specified
            cell id or cellids is read.
        Returns
        -------
        numpy.ndarray
        """
        if not tag and not name:
            raise ValueError()

        name = name.lower()

        fid = self.fid

        if "/" in name:
            popname, varname = name.split("/")
        else:
            popname, varname = "pop", name

        # TODO: add data reducers

        for child in self.xmlroot:
            if tag and child.tag != tag:
                continue
            if name and "name" in child.attrib and child.attrib["name"].lower() != name:
                continue
            if mesh and "mesh" in child.attrib and child.attrib["mesh"] != mesh:
                continue
            if child.tag == tag:
                vsize = int(child.attrib["vectorsize"])
                asize = int(child.attrib["arraysize"])
                dsize = int(child.attrib["datasize"])
                dtype = child.attrib["datatype"]
                variable_offset = int(child.text)

                # Select efficient method to read data based on number of cells
                if hasattr(cellids, "__len__"):
                    ncellids = len(cellids)
                    # Read multiple specified cells
                    # For reading a large amount of single cells, it'll be faster to
                    # read all data from the file and sort afterwards.
                    arraydata = []
                    if ncellids > 5000:
                        result_size = ncellids
                        read_size = asize
                        read_offsets = [0]
                    else:  # Read multiple cell ids one-by-one
                        result_size = ncellids
                        read_size = 1
                        read_offsets = [
                            self.celldict[cid] * dsize * vsize for cid in cellids
                        ]
                else:
                    if cellids < 0:  # all cells
                        result_size = asize
                        read_size = asize
                        read_offsets = [0]
                    else:  # parameter or single cell
                        result_size = 1
                        read_size = 1
                        read_offsets = [self.celldict[cellids] * dsize * vsize]

                for r_offset in read_offsets:
                    use_offset = int(variable_offset + r_offset)
                    fid.seek(use_offset)

                    if dtype == "float" and dsize == 4:
                        dtype = np.float32
                    elif dtype == "float" and dsize == 8:
                        dtype = np.float64
                    elif dtype == "int" and dsize == 4:
                        dtype = np.int32
                    elif dtype == "int" and dsize == 8:
                        dtype = np.int64
                    elif dtype == "uint" and dsize == 4:
                        dtype = np.uint32
                    elif dtype == "uint" and dsize == 8:
                        dtype = np.uint64

                    data = np.fromfile(fid, dtype, count=vsize * read_size)

                    if len(read_offsets) != 1:
                        arraydata.append(data)

                if len(read_offsets) == 1 and result_size < read_size:
                    # Many single cell IDs requested
                    # Pick the elements corresponding to the requested cells
                    for cid in cellids:
                        append_offset = self.celldict[cid] * vsize
                        arraydata.append(data[append_offset : append_offset + vsize])
                    data = np.squeeze(np.array(arraydata))
                elif len(read_offsets) != 1:
                    # Not so many single cell IDs requested
                    data = np.squeeze(np.array(arraydata))

                if vsize > 1:
                    data = data.reshape(result_size, vsize)

                if result_size == 1:
                    return data[0]
                else:
                    return data

        if name:
            raise NameError(
                name
                + "/"
                + tag
                + "/"
                + mesh
                + " not found in .vlsv file or in data reducers!"
            )

    def read_variable(self, name: str, cellids=-1, sorted: bool = True):
        """Read variables as numpy arrays from the open vlsv file."""

        if self.has_variable(name) and name.startswith("fg_"):
            if not cellids == -1:
                raise ValueError("CellID requests not supported for FSgrid.")
            return self.read_fg_variable(name=name)

        if self.has_variable(name) and name.startswith("ig_"):
            if not cellids == -1:
                raise ValueError("CellID requests not supported for ionosphere.")
            return self.read_ionosphere_variable(name=name)

        raw = self.read(
            mesh="SpatialGrid",
            name=name,
            tag="VARIABLE",
            cellids=cellids,
        )
        if sorted:
            if raw.ndim == 1:
                v = raw[self.cellindex]
            else:
                v = raw[self.cellindex, :]
            if v.dtype == np.float64:  # 32-bit is enough for analysis
                v = np.float32(v)
        else:
            v = raw

        return v

    def read_fg_variable(self, name: str):
        raw = self.read(
            mesh="fsgrid",
            name=name,
            tag="VARIABLE",
        )

        bbox = tuple(ncell * 2**self.maxamr for ncell in self.ncells)

        # Determine fsgrid domain decomposition
        nIORanks = self.read_parameter("numWritingRanks")  # Int32

        if raw.ndim > 1:
            dataOrdered = np.empty((*bbox, raw.shape[-1]), dtype=np.float32)
        else:
            dataOrdered = np.empty(bbox, dtype=np.float32)

        def getDomainDecomposition(globalsize, nproc: int) -> list[int]:
            """Obtain decomposition of this grid over the given number of processors.
            Reference: fsgrid.hpp
            """
            domainDecomp = (1, 1, 1)
            minValue = 1e20
            for i in range(1, min(nproc, globalsize[0]) + 1):
                iBox = max(globalsize[0] / i, 1.0)
                for j in range(1, min(nproc, globalsize[1]) + 1):
                    if i * j > nproc:
                        break
                    jBox = max(globalsize[1] / j, 1.0)
                    for k in range(1, min(nproc, globalsize[2]) + 1):
                        if i * j * k > nproc:
                            continue
                        kBox = max(globalsize[2] / k, 1.0)
                        v = (
                            10 * iBox * jBox * kBox
                            + ((jBox * kBox) if i > 1 else 0)
                            + ((iBox * kBox) if j > 1 else 0)
                            + ((iBox * jBox) if k > 1 else 0)
                        )
                        if i * j * k == nproc and v < minValue:
                            minValue = v
                            domainDecomp = (i, j, k)

            return domainDecomp

        def calcLocalStart(globalCells, nprocs: int, lcells: int) -> int:
            ncells = globalCells // nprocs
            remainder = globalCells % nprocs
            lstart = (
                lcells * (ncells + 1)
                if lcells < remainder
                else lcells * ncells + remainder
            )

            return lstart

        def calcLocalSize(globalCells, nprocs: int, lcells: int) -> int:
            ncells = globalCells // nprocs
            remainder = globalCells % nprocs
            lsize = ncells + 1 if lcells < remainder else ncells

            return lsize

        fgDecomposition = getDomainDecomposition(bbox, nIORanks)

        offsetnow = 0

        for i in range(nIORanks):
            xyz = (
                i // fgDecomposition[2] // fgDecomposition[1],
                i // fgDecomposition[2] % fgDecomposition[1],
                i % fgDecomposition[2],
            )

            lsize = tuple(
                map(
                    lambda i: calcLocalSize(bbox[i], fgDecomposition[i], xyz[i]),
                    range(0, 3),
                )
            )

            lstart = tuple(
                map(
                    lambda i: calcLocalStart(bbox[i], fgDecomposition[i], xyz[i]),
                    range(0, 3),
                )
            )

            offsetnext = offsetnow + np.prod(lsize)
            lend = tuple(st + si for st, si in zip(lstart, lsize))

            # Reorder data
            if raw.ndim > 1:
                ldata = raw[offsetnow:offsetnext, :].reshape(*lsize, raw.shape[-1])
                dataOrdered[
                    lstart[0] : lend[0], lstart[1] : lend[1], lstart[2] : lend[2], :
                ] = ldata
            else:
                ldata = raw[offsetnow:offsetnext].reshape(*lsize)
                dataOrdered[
                    lstart[0] : lend[0], lstart[1] : lend[1], lstart[2] : lend[2]
                ] = ldata

            offsetnow = offsetnext

        v = np.squeeze(dataOrdered)

        return v

    def read_variable_meta(self, var: str):
        unit, unitLaTeX, variableLaTeX, unitConversion = "", "", "", ""

        if var in units_predefined:
            unit, variableLaTeX, unitLaTeX = units_predefined[var]
        elif self.has_variable(var):  # For Vlasiator 5 files, MetaVLSV is included
            for child in self.xmlroot:
                if "name" in child.attrib and child.attrib["name"] == var:
                    if not "unit" in child.attrib:
                        break
                    else:
                        unit = child.attrib["unit"]
                        unitLaTeX = child.attrib["unitLaTeX"]
                        variableLaTeX = child.attrib["variableLaTeX"]
                        unitConversion = child.attrib["unitConversion"]

        return VarInfo(unit, unitLaTeX, variableLaTeX, unitConversion)

    def read_parameter(self, name: str):
        return self.read(name=name, tag="PARAMETER")

    def read_vcells(self, cellid: int, species: str = "proton"):
        """Read raw velocity block data.

        Parameters
        ----------
        cellid :
            Cell ID of the cell whose velocity blocks are read.
        species : str
            Population required.

        Returns
        -------
        numpy.ndarray
            A numpy array with block ids and their data.
        """

        fid = self.fid
        mesh = self.meshes[species]
        vblock_size = mesh.vblock_size

        if not np.any(mesh.cellwithVDF):
            for node in self.nodecellwithVDF:
                if node.attrib["name"] == species:
                    asize = int(node.attrib["arraysize"])
                    offset = int(node.text)
                    fid.seek(offset)
                    cellwithVDF = np.fromfile(fid, dtype=np.uint64, count=asize)
                    mesh.cellwithVDF = cellwithVDF
                    break

            for node in self.xmlroot.findall("BLOCKSPERCELL"):
                if node.attrib["name"] == species:
                    asize = int(node.attrib["arraysize"])
                    dsize = int(node.attrib["datasize"])
                    offset = int(node.text)
                    fid.seek(offset)
                    T = np.int32 if dsize == 4 else np.int64
                    nblock_C = np.fromfile(fid, dtype=T, count=asize).astype(np.int64)
                    mesh.nblock_C = nblock_C
                    break

        # Check that cells have VDF stored
        try:
            cellWithVDFIndex = np.where(mesh.cellwithVDF == cellid)[0][0]
            nblocks = mesh.nblock_C[cellWithVDFIndex]
            if nblocks == 0:
                raise ValueError(f"Cell ID {cellid} does not store VDF!")
        except:
            raise ValueError(f"Cell ID {cellid} does not store VDF!")
        # Offset position to vcell storage
        offset_v = np.sum(mesh.nblock_C[0 : cellWithVDFIndex - 1], initial=0).item()

        # Read raw VDF
        for node in self.xmlroot.findall("BLOCKVARIABLE"):
            if node.attrib["name"] == species:
                dsize = int(node.attrib["datasize"])
                offset = int(node.text)
            break

        bsize = np.prod(vblock_size).item()
        fid.seek(offset_v * bsize * dsize + offset)
        T = np.float32 if dsize == 4 else np.float64
        data = np.fromfile(
            fid,
            dtype=T,
            count=bsize * nblocks,
        ).reshape(nblocks, bsize)

        # Read block IDs
        for node in self.xmlroot.findall("BLOCKIDS"):
            if node.attrib["name"] == species:
                dsize = int(node.attrib["datasize"])
                offset = int(node.text)
            break

        fid.seek(offset_v * dsize + offset)
        T = np.int32 if dsize == 4 else np.int64
        blockIDs = np.fromfile(fid, dtype=T, count=nblocks)

        # Velocity cell IDs and distributions (ordered by blocks)
        vcellids = np.empty(bsize * nblocks, dtype=np.int32)
        vcellf = np.empty(bsize * nblocks, dtype=np.float32)

        for i, bid in enumerate(blockIDs):
            for j in range(bsize):
                index_ = i * bsize + j
                vcellids[index_] = j + bsize * bid
                vcellf[index_] = data[i, j]

        return vcellids, vcellf

    def _has_attribute(self, attribute: str, name: str) -> bool:
        """Check if a given attribute exists in the xml."""
        for child in self.xmlroot:
            if child.tag == attribute and "name" in child.attrib:
                if child.attrib["name"].lower() == name.lower():
                    return True
        return False

    def has_variable(self, name: str) -> bool:
        return self._has_attribute("VARIABLE", name)

    def has_parameter(self, name: str) -> bool:
        return self._has_attribute("PARAMETER", name)

    def getmaxrefinement(self, cellid: np.ndarray):
        """Get the maximum spatial refinement level."""
        ncell = np.prod(self.ncells)
        maxamr, cid = 0, ncell
        while cid < max(cellid):
            maxamr += 1
            cid += ncell * 8**maxamr

        return maxamr

    def getvcellcoordinates(self, vcellids: np.ndarray, species: str = "proton"):
        mesh = self.meshes[species]
        vblocks = mesh.vblocks
        vblock_size = mesh.vblock_size
        dv = mesh.dv
        vmin = mesh.vmin

        bsize = np.prod(vblock_size).item()
        blockid = tuple(cid // bsize for cid in vcellids)
        # Get block coordinates
        blockInd = [
            (
                bid.item() % vblocks[0].item(),
                bid.item() // vblocks[0].item() % vblocks[1].item(),
                bid.item() // (vblocks[0].item() * vblocks[1].item()),
            )
            for bid in blockid
        ]
        blockCoord = [
            (
                bInd[0] * dv[0].item() * vblock_size[0].item() + vmin[0].item(),
                bInd[1] * dv[1].item() * vblock_size[1].item() + vmin[1].item(),
                bInd[2] * dv[2].item() * vblock_size[2].item() + vmin[2].item(),
            )
            for bInd in blockInd
        ]
        # Get cell indices
        vcellblockids = tuple(vid % bsize for vid in vcellids)
        cellidxyz = [
            (
                cid % vblock_size[0],
                cid // vblock_size[0] % vblock_size[1],
                cid // (vblock_size[0] * vblock_size[1]),
            )
            for cid in vcellblockids
        ]
        # Get cell coordinates
        cellCoords = [
            tuple(
                blockCoord[i][j] + (cellidxyz[i][j].item() + 0.5) * dv[j].item()
                for j in range(3)
            )
            for i in range(len(vcellids))
        ]

        return cellCoords


def _getdim2d(ncells: tuple, maxamr: int, normal: str):
    ratio = 2**maxamr
    if normal == "x":
        i1, i2 = 1, 2
    elif normal == "y":
        i1, i2 = 0, 2
    elif normal == "z":
        i1, i2 = 0, 1
    dims = (ncells[i1] * ratio, ncells[i2] * ratio)

    return dims
