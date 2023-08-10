from xml.etree import ElementTree
import numpy as np
import os
from collections import namedtuple
from pyvlasiator.vlsv.variables import units_predefined


class VMeshInfo:
    def __init__(
        self,
        vblocks: np.ndarray,
        vblock_size: np.ndarray,
        vmin: np.ndarray,
        vmax: np.ndarray,
        dv: np.ndarray,
    ) -> None:
        self.vblocks = vblocks
        self.vblock_size = vblock_size
        self.vmin = vmin
        self.vmax = vmax
        self.dv = dv
        self.cellwithVDF = np.empty(0, dtype=np.uint64)
        self.nblock_C = np.empty(0, dtype=np.int64)

        self.vblocks.flags.writeable = False
        self.vblock_size.flags.writeable = False
        self.vmin.flags.writeable = False
        self.vmax.flags.writeable = False
        self.dv.flags.writeable = False


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
                self.ncells = np.array(
                    (
                        self.read_parameter("xcells_ini"),
                        self.read_parameter("ycells_ini"),
                        self.read_parameter("zcells_ini"),
                    ),
                    dtype=int,
                )
                self.block_size = np.array((1, 1, 1), dtype=int)
                self.coordmin = np.array(
                    (
                        self.read_parameter("xmin"),
                        self.read_parameter("ymin"),
                        self.read_parameter("zmin"),
                    ),
                    dtype=float,
                )
                self.coordmax = np.array(
                    (
                        self.read_parameter("xmax"),
                        self.read_parameter("ymax"),
                        self.read_parameter("zmax"),
                    ),
                    dtype=float,
                )
            except:  # dummy values
                self.ncells = np.array((1, 1, 1), dtype=int)
                self.block_size = np.array((1, 1, 1), dtype=int)
                self.coordmin = np.array((0.0, 0.0, 0.0), dtype=float)
                self.coordmax = np.array((1.0, 1.0, 1.0), dtype=float)

        else:
            # Vlasiator 5+ file
            nodeX = self.read(tag="MESH_NODE_CRDS_X", mesh=meshName)
            nodeY = self.read(tag="MESH_NODE_CRDS_Y", mesh=meshName)
            nodeZ = self.read(tag="MESH_NODE_CRDS_Z", mesh=meshName)
            self.ncells = np.fromiter((i for i in bbox[0:3]), dtype=int)
            self.block_size = np.fromiter((i for i in bbox[3:6]), dtype=int)
            self.coordmin = np.array((nodeX[0], nodeY[0], nodeZ[0]), dtype=float)
            self.coordmax = np.array((nodeX[-1], nodeY[-1], nodeZ[-1]), dtype=float)

        self.dcoord = np.fromiter(
            ((self.coordmax[i] - self.coordmin[i]) / self.ncells[i] for i in range(3)),
            dtype=float,
        )

        self.ncells.flags.writeable = False
        self.block_size.flags.writeable = False
        self.coordmin.flags.writeable = False
        self.coordmax.flags.writeable = False
        self.dcoord.flags.writeable = False

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
                    vblocks = np.array(
                        (
                            self.read_parameter("vxblocks_ini"),
                            self.read_parameter("vyblocks_ini"),
                            self.read_parameter("vzblocks_ini"),
                        ),
                        dtype=int,
                    )
                    vblock_size = np.array((4, 4, 4), dtype=int)
                    vmin = np.array(
                        (
                            self.read_parameter("vxmin"),
                            self.read_parameter("vymin"),
                            self.read_parameter("vzmin"),
                        ),
                        dtype=float,
                    )
                    vmax = np.array(
                        (
                            self.read_parameter("vxmax"),
                            self.read_parameter("vymax"),
                            self.read_parameter("vzmax"),
                        ),
                        dtype=float,
                    )

                else:  # no velocity space
                    vblocks = np.array((0, 0, 0), dtype=int)
                    vblock_size = np.array((4, 4, 4), dtype=int)
                    vmin = np.array((0.0, 0.0, 0.0), dtype=float)
                    vmax = np.array((0.0, 0.0, 0.0), dtype=float)
                    dv = np.array((1.0, 1.0, 1.0), dtype=float)

            else:  # Vlasiator 5+ file with bounding box
                nodeX = self.read(tag="MESH_NODE_CRDS_X", mesh=popname)
                nodeY = self.read(tag="MESH_NODE_CRDS_Y", mesh=popname)
                nodeZ = self.read(tag="MESH_NODE_CRDS_Z", mesh=popname)

                vblocks = np.array((*bbox[0:3],), dtype=int)
                vblock_size = np.array((*bbox[3:6],), dtype=int)
                vmin = np.array((nodeX[0], nodeY[0], nodeZ[0]), dtype=float)
                vmax = np.array((nodeX[-1], nodeY[-1], nodeZ[-1]), dtype=float)

            dv = np.fromiter(
                ((vmax[i] - vmin[i]) / vblocks[i] / vblock_size[i] for i in range(3)),
                dtype=float,
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

    def _read_xml_footer(self) -> None:
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

    def read_variable(
        self, name: str, cellids: int | list[int] | np.ndarray = -1, sorted: bool = True
    ) -> np.ndarray:
        """
        Read variables as numpy arrays from the open vlsv file.

        Parameters
        ----------
        cellids : int or list[int] or np.ndarray
            If -1 then all data is read. If nonzero then only the vector for the specified
            cell id or cellids is read.
        sorted : bool
            If the returned array is sorted by cell IDs. Only applied for full arrays.
        Returns
        -------
        numpy.ndarray
        """

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
        if hasattr(cellids, "__len__"):  # part of cells requested
            return np.float32(raw)

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

        self.init_cellswithVDF(species)

        # Check that cells have VDF stored
        try:
            cellWithVDFIndex = np.where(mesh.cellwithVDF == cellid)[0][0]
            nblocks = mesh.nblock_C[cellWithVDFIndex]
        except:
            raise ValueError(f"Cell ID {cellid} does not store VDF!")
        # Offset position to vcell storage
        offset_v = np.sum(mesh.nblock_C[0:cellWithVDFIndex], initial=0)

        # Read raw VDF
        for node in self.xmlroot.findall("BLOCKVARIABLE"):
            if node.attrib["name"] == species:
                dsize = int(node.attrib["datasize"])
                offset = int(node.text)
            break

        bsize = np.prod(vblock_size)
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

    def init_cellswithVDF(self, species: str = "proton") -> None:
        fid = self.fid
        mesh = self.meshes[species]
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

            mesh.cellwithVDF = np.delete(mesh.cellwithVDF, np.where(mesh.nblock_C == 0))

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

    def getcell(self, loc: np.ndarray | tuple[int, ...] | list[int]):
        coordmin, coordmax = self.coordmin, self.coordmax
        dcoord = self.dcoord
        ncells = self.ncells
        celldict = self.celldict
        maxamr = self.maxamr

        for i in range(3):
            if not coordmin[i] < loc[i] < coordmax[i]:
                raise ValueError(f"{i} coordinate out of bound!")

        indices = np.fromiter(
            ((loc[i] - coordmin[i]) // dcoord[i] for i in range(3)), dtype=int
        )

        cid = (
            indices[0] + indices[1] * ncells[0] + indices[2] * ncells[0] * ncells[1] + 1
        )

        ncells_lowerlevel = 0
        ncell = np.prod(ncells)

        for ilevel in range(maxamr):
            if cid in celldict:
                break
            ncells_lowerlevel += (8**ilevel) * ncell
            ratio = 2 ** (ilevel + 1)
            indices = np.fromiter(
                (
                    np.floor((loc[i] - coordmin[i]) / dcoord[i] * ratio)
                    for i in range(3)
                ),
                dtype=int,
            )
            cid = (
                ncells_lowerlevel
                + indices[0]
                + ratio * ncells[0] * indices[1]
                + ratio**2 * ncells[0] * ncells[1] * indices[2]
                + 1
            )

        return cid

    def getvcellcoordinates(
        self, vcellids: np.ndarray, species: str = "proton"
    ) -> np.ndarray:
        mesh = self.meshes[species]
        vblocks = mesh.vblocks
        vblock_size = mesh.vblock_size
        dv = mesh.dv
        vmin = mesh.vmin

        bsize = np.prod(vblock_size)
        blockid = np.fromiter((cid // bsize for cid in vcellids), dtype=int)
        # Get block coordinates
        blockInd = [
            np.array(
                (
                    bid % vblocks[0],
                    bid // vblocks[0] % vblocks[1],
                    bid // (vblocks[0] * vblocks[1]),
                ),
                dtype=int,
            )
            for bid in blockid
        ]
        blockCoord = [
            np.array(
                (
                    bInd[0] * dv[0] * vblock_size[0] + vmin[0],
                    bInd[1] * dv[1] * vblock_size[1] + vmin[1],
                    bInd[2] * dv[2] * vblock_size[2] + vmin[2],
                ),
                dtype=float,
            )
            for bInd in blockInd
        ]
        # Get cell indices
        vcellblockids = np.fromiter((vid % bsize for vid in vcellids), dtype=int)
        cellidxyz = np.array(
            [
                np.array(
                    (
                        cid % vblock_size[0],
                        cid // vblock_size[0] % vblock_size[1],
                        cid // (vblock_size[0] * vblock_size[1]),
                    ),
                    dtype=int,
                )
                for cid in vcellblockids
            ]
        )
        # Get cell coordinates
        cellCoords = np.array(
            [
                np.fromiter(
                    (
                        blockCoord[i][j] + (cellidxyz[i][j] + 0.5) * dv[j]
                        for j in range(3)
                    ),
                    dtype=float,
                )
                for i in range(len(vcellids))
            ]
        )

        return cellCoords

    def getnearestcellwithvdf(self, id: int, species: str = "proton"):
        self.init_cellswithVDF(species)
        cells = self.meshes[species].cellwithVDF
        if not np.any(cells):
            raise ValueError(f"No distribution saved in {self.name}")
        coords_orig = self.getcellcoordinates(id)
        coords = [self.getcellcoordinates(cid) for cid in cells]
        min_ = np.argmin(np.sum(np.square(coords - coords_orig), axis=1))

        return cells[min_]

    def getcellcoordinates(self, cid: int):
        ncells = self.ncells
        coordmin, coordmax = self.coordmin, self.coordmax
        cid -= 1  # for easy divisions

        ncells_refmax = list(ncells)
        reflevel = 0
        subtraction = np.prod(ncells) * (2**reflevel) ** 3
        # sizes on the finest level
        while cid >= subtraction:
            cid -= subtraction
            reflevel += 1
            subtraction *= 8
            ncells_refmax[0] *= 2
            ncells_refmax[1] *= 2
            ncells_refmax[2] *= 2

        indices = np.array(
            (
                cid % ncells_refmax[0],
                cid // ncells_refmax[0] % ncells_refmax[1],
                cid // (ncells_refmax[0] * ncells_refmax[1]),
            ),
            dtype=int,
        )

        coords = np.fromiter(
            (
                coordmin[i]
                + (indices[i] + 0.5) * (coordmax[i] - coordmin[i]) / ncells_refmax[i]
                for i in range(3)
            ),
            dtype=float,
        )

        return coords

    def getslicecell(
        self, sliceoffset: float, dir: int, minCoord: float, maxCoord: float
    ):
        if not dir in (0, 1, 2):
            raise ValueError(f"Unknown slice direction {dir}")

        ncells, maxamr, celldict = self.ncells, self.maxamr, self.celldict
        nsize = ncells[dir]
        sliceratio = sliceoffset / (maxCoord - minCoord)
        if not (0.0 <= sliceratio <= 1.0):
            raise ValueError("slice plane index out of bound!")

        # Find the ids
        nlen = 0
        ncell = np.prod(ncells)
        # number of cells up to each refinement level
        lvlC = np.fromiter((ncell * 8**ilvl for ilvl in range(maxamr + 1)), dtype=int)
        lvlAccum = np.add.accumulate(lvlC)
        nStart = np.insert(lvlAccum, 0, 0)

        indexlist = np.empty(0, dtype=int)
        idlist = np.empty(0, dtype=int)

        cellidsorted = np.fromiter(celldict.keys(), dtype=int)
        cellidsorted.sort()

        for ilvl in range(maxamr + 1):
            nLow, nHigh = nStart[ilvl], nStart[ilvl + 1]
            idfirst_ = np.searchsorted(cellidsorted, nLow + 1)
            idlast_ = np.searchsorted(cellidsorted, nHigh, side="right")

            ids = cellidsorted[idfirst_:idlast_]

            ix, iy, iz = getindexes(ilvl, ncells[0], ncells[1], nLow, ids)

            if dir == 0:
                coords = ix
            elif dir == 1:
                coords = iy
            else:
                coords = iz

            # Find the cut plane index for each refinement level (0-based)
            depth = int(np.floor(sliceratio * nsize * 2**ilvl))
            # Find the needed elements to create the cut and save the results
            elements = coords == depth
            indexlist = np.append(indexlist, np.arange(nlen, nlen + len(ids))[elements])
            idlist = np.append(idlist, ids[elements])

            nlen += len(ids)

        return idlist, indexlist

    def refineslice(self, idlist: np.ndarray, data: np.ndarray, normal: int):
        ncells, maxamr = self.ncells, self.maxamr

        dims = _getdim2d(ncells, maxamr, normal)
        # meshgrid-like 2D input for matplotlib
        dpoints = np.empty((dims[1], dims[0]), dtype=data.dtype)

        # Create the plot grid
        ncell = np.prod(ncells)
        nHigh, nLow = ncell, 0

        for i in range(maxamr + 1):
            idfirst_ = np.searchsorted(idlist, nLow + 1)
            idlast_ = np.searchsorted(idlist, nHigh, side="right")

            ids = idlist[idfirst_:idlast_]
            d = data[idfirst_:idlast_]

            ix, iy, iz = getindexes(i, ncells[0], ncells[1], nLow, ids)

            # Get the correct coordinate values and the widths for the plot
            if normal == 0:
                a, b = iy, iz
            elif normal == 1:
                a, b = ix, iz
            elif normal == 2:
                a, b = ix, iy

            # Insert the data values into dpoints
            refineRatio = 2 ** (maxamr - i)
            iRange = range(refineRatio)
            X, Y = np.meshgrid(iRange, iRange, indexing="ij")
            coords = np.empty((len(a), 2 ** (2 * (maxamr - i)), 2), dtype=int)

            for ic, (ac, bc) in enumerate(zip(a, b)):
                for ir in range(2 ** (2 * (maxamr - i))):
                    index_ = np.unravel_index(ir, (refineRatio, refineRatio))
                    coords[ic, ir] = [
                        ac * refineRatio + X[index_],
                        bc * refineRatio + Y[index_],
                    ]

            for ic, dc in enumerate(d):
                for ir in range(2 ** (2 * (maxamr - i))):
                    dpoints[coords[ic, ir, 1], coords[ic, ir, 0]] = dc

            nLow = nHigh
            nHigh += ncell * 8 ** (i + 1)

        return dpoints


def _getdim2d(ncells: tuple, maxamr: int, normal: int):
    ratio = 2**maxamr
    if normal == 0:
        i1, i2 = 1, 2
    elif normal == 1:
        i1, i2 = 0, 2
    elif normal == 2:
        i1, i2 = 0, 1
    dims = (ncells[i1] * ratio, ncells[i2] * ratio)

    return dims


def getindexes(
    ilevel: int, xcells: int, ycells: int, nCellUptoLowerLvl: int, ids: np.ndarray
):
    ratio = 2**ilevel
    slicesize = xcells * ycells * ratio**2

    iz = (ids - nCellUptoLowerLvl - 1) // slicesize
    iy = np.zeros_like(iz)
    ix = np.zeros_like(iz)

    # number of ids up to the coordinate z in the refinement level ilevel
    idUpToZ = iz * slicesize + nCellUptoLowerLvl
    iy = (ids - idUpToZ - 1) // (xcells * ratio)
    ix = ids - idUpToZ - iy * xcells * ratio - 1

    return ix, iy, iz
