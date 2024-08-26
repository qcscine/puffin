#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Tuple
import numpy as np
import os
import unittest

try:
    import scine_database as db
    import scine_molassembler as masm
    import scine_utilities as utils
    from scine_puffin.utilities.masm_helper import add_masm_info, get_molecules_result, _modify_based_on_distances
except ImportError:
    pass
else:
    class MasmHelperTests(unittest.TestCase):
        def __init__(self, *args, **kwargs) -> None:
            super(MasmHelperTests, self).__init__(*args, **kwargs)
            self.settings = {
                "sub_based_on_distance_connectivity": True,
                "add_based_on_distance_connectivity": True,
            }

        class MockStructure(object):
            atoms: utils.AtomCollection
            graph_dict: dict = {}
            model: db.Model = db.Model("", "", "")

            def __init__(self, atoms: utils.AtomCollection) -> None:
                self.atoms = atoms

            def get_atoms(self) -> utils.AtomCollection:
                return self.atoms

            def set_graph(self, key, value) -> None:
                self.graph_dict[key] = value

            def get_model(self) -> db.Model:
                return self.model

            def set_model(self, model: db.Model) -> None:
                self.model = model

        def read_fake_files(self,
                            content: str,
                            filename: str) -> Tuple[utils.AtomCollection,
                                                    utils.BondOrderCollection]:
            with open(filename, "w") as f:
                f.write(content)
            ac, bo = utils.io.read(filename)
            os.unlink(filename)
            return ac, bo

        def test_add_masm_info(self):
            molfile_content = """CH4O4
APtclcactv11132009123D 0   0.00000     0.00000

  9  6  0  0  0  0  0  0  0  0999 V2000
  -13.3670   -0.1500    0.0435 O   0  0  0  0  0  0  0  0  0  0  0  0
  -13.6338    0.7157   -0.2949 H   0  0  0  0  0  0  0  0  0  0  0  0
  -12.5296   -0.4722   -0.3171 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.3999    0.0736    0.0262 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.6667    0.9393   -0.3122 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5624   -0.2486   -0.3343 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.8376    0.0064    0.0035 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0152    0.8911    0.0200 O   0  0  0  0  0  0  0  0  0  0  0  0
    7.6600   -0.8783   -0.0130 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  4  5  1  0  0  0  0
  4  6  1  0  0  0  0
  7  8  2  0  0  0  0
  7  9  2  0  0  0  0
M  END
$$$$
            """
            import ast

            ac, bo = self.read_fake_files(molfile_content, "add_masm_info.mol")

            # check if bo are unchanged if both options are False
            distance_bo = utils.BondDetector.detect_bonds(ac)
            unchanged_bo = _modify_based_on_distances(ac, distance_bo, bo, self.settings)
            assert unchanged_bo == bo

            structure = self.MockStructure(ac)
            add_masm_info(structure, bo, self.settings)

            assert "masm_decision_list" in structure.graph_dict
            assert "masm_cbor_graph" in structure.graph_dict
            assert "masm_idx_map" in structure.graph_dict

            # The molfile above should yield two water molecules and a co2
            water = masm.io.experimental.from_smiles("O")
            co2 = masm.io.experimental.from_smiles("O=C=O")

            def set_single_bonds_only(mol: masm.Molecule) -> None:
                for b in mol.graph.bonds():
                    if mol.graph.bond_type(b) == masm.BondType.Eta:
                        continue
                    mol.set_bond_type(b[0], b[1], masm.BondType.Single)

            set_single_bonds_only(water)
            set_single_bonds_only(co2)

            def revert_cbor(cbor_str: str) -> masm.Molecule:
                serializer = masm.JsonSerialization
                cbor_binary = serializer.base_64_decode(cbor_str)
                cbor_format = serializer.BinaryFormat.CBOR
                serialization = serializer(cbor_binary, cbor_format)
                return serialization.to_molecule()

            joint_cbor_str = structure.graph_dict["masm_cbor_graph"]
            molecules = [revert_cbor(m_str) for m_str in joint_cbor_str.split(";")]
            assert len(molecules) == 3
            assert molecules.count(water) == 2
            assert molecules.count(co2) == 1

            # Check that the index map is correct
            index_map = ast.literal_eval(structure.graph_dict["masm_idx_map"])
            assert all([ac.elements[i] == molecules[c].graph.element_type(j) for i, (c, j) in enumerate(index_map)])
            assert len(index_map) == len(set(index_map))

        def test_surface_structure(self):
            xyz_file_content = """13

Cu     0.0000000000    1.4783740208   16.7258927240
Cu     0.0000000000    0.0000000000   20.9073659051
Cu     1.2803094583    0.7391870104   18.8166293146
Cu    -1.2803094583    3.6959350521   16.7258927240
Cu    -1.2803094583    2.2175610312   20.9073659051
Cu     0.0000000000    2.9567480416   18.8166293146
Cu     2.5606189167    1.4783740208   16.7258927240
Cu     2.5606189167    0.0000000000   20.9073659051
Cu     3.8409283750    0.7391870104   18.8166293146
Cu     1.2803094583    3.6959350521   16.7258927240
Cu     1.2803094583    2.2175610312   20.9073659051
Cu     2.5606189167    2.9567480416   18.8166293146
H      1.2803094572    2.2175610336   21.7011317394
            """
            pbc_matrix = np.array(
                [
                    [9.67773693e00, 0.00000000e00, -1.55629880e-15],
                    [-4.83886846e00, 8.38116603e00, 5.92590478e-16],
                    [0.00000000e00, 0.00000000e00, 1.42233104e02],
                ]
            )
            pbc = utils.PeriodicBoundaries(pbc_matrix)
            ac, bo = self.read_fake_files(xyz_file_content, "test_surface_pruning.xyz")
            structure = self.MockStructure(ac)
            model = db.Model("FAKE", "", "")
            model.periodic_boundaries = str(pbc)
            structure.set_model(model)
            surface_indices = list(range(len(ac) - 1))  # last atom is adsorbant
            bo = utils.SolidStateBondDetector.detect_bonds(ac, pbc, set(surface_indices))
            add_masm_info(structure, bo, self.settings, surface_indices)
            assert "masm_decision_list" in structure.graph_dict
            assert "masm_cbor_graph" in structure.graph_dict
            assert "masm_idx_map" in structure.graph_dict

        def test_split_surface_structure(self):
            xyz_file_content = """81

Cu     1.0719532500    1.0719532500   35.0171395000
Cu     1.0719532500    5.3597662500   35.0171395000
Cu     5.3597662500    1.0719532500   35.0171395000
Cu     5.3597662500    5.3597662500   35.0171395000
Cu     3.2158597500    1.0719532500   32.8732330000
Cu     3.2158597500    5.3597662500   32.8732330000
Cu     7.5036727500    1.0719532500   32.8732330000
Cu     7.5036727500    5.3597662500   32.8732330000
Cu     1.0719532500    3.2158597500   32.8732330000
Cu     1.0719532500    7.5036727500   32.8732330000
Cu     5.3597662500    3.2158597500   32.8732330000
Cu     5.3597662500    7.5036727500   32.8732330000
Cu     3.2158597500    3.2158597500   35.0171395000
Cu     3.2158597500    7.5036727500   35.0171395000
Cu     7.5036727500    3.2158597500   35.0171395000
Cu     7.5036727500    7.5036727500   35.0171395000
O      2.1439065000    2.1439065000   33.9451862500
O      2.1439065000    6.4317195000   33.9451862500
O      6.4317195000    2.1439065000   33.9451862500
O      6.4317195000    6.4317195000   33.9451862500
O      0.0000000000    0.0000000000   36.0890927500
O      0.0000000000    4.2878130000   36.0890927500
O      4.2878130000    0.0000000000   36.0890927500
O      4.2878130000    4.2878130000   36.0890927500
Cu     1.0719532500    1.0719532500   39.3049525000
Cu     1.0719532500    5.3597662500   39.3049525000
Cu     5.3597662500    1.0719532500   39.3049525000
Cu     5.3597662500    5.3597662500   39.3049525000
Cu     3.2158597500    1.0719532500   37.1610460000
Cu     3.2158597500    5.3597662500   37.1610460000
Cu     7.5036727500    1.0719532500   37.1610460000
Cu     7.5036727500    5.3597662500   37.1610460000
Cu     1.0719532500    3.2158597500   37.1610460000
Cu     1.0719532500    7.5036727500   37.1610460000
Cu     5.3597662500    3.2158597500   37.1610460000
Cu     5.3597662500    7.5036727500   37.1610460000
Cu     3.2158597500    3.2158597500   39.3049525000
Cu     3.2158597500    7.5036727500   39.3049525000
Cu     7.5036727500    3.2158597500   39.3049525000
Cu     7.5036727500    7.5036727500   39.3049525000
O      2.1439065000    2.1439065000   38.2329992500
O      2.1439065000    6.4317195000   38.2329992500
O      6.4317195000    2.1439065000   38.2329992500
O      6.4317195000    6.4317195000   38.2329992500
O      0.0000000000    0.0000000000   40.3769057500
O      0.0000000000    4.2878130000   40.3769057500
O      4.2878130000    0.0000000000   40.3769057500
O      4.2878130000    4.2878130000   40.3769057500
Cu     1.0719532500    1.0719532500   43.5927655000
Cu     1.0719532500    5.3597662500   43.5927655000
Cu     5.3597662500    1.0719532500   43.5927655000
Cu     5.3597662500    5.3597662500   43.5927655000
Cu     3.2158597500    1.0719532500   41.4488590000
Cu     3.2158597500    5.3597662500   41.4488590000
Cu     7.5036727500    1.0719532500   41.4488590000
Cu     7.5036727500    5.3597662500   41.4488590000
Cu     1.0719532500    3.2158597500   41.4488590000
Cu     1.0719532500    7.5036727500   41.4488590000
Cu     5.3597662500    3.2158597500   41.4488590000
Cu     5.3597662500    7.5036727500   41.4488590000
Cu     3.2158597500    3.2158597500   43.5927655000
Cu     3.2158597500    7.5036727500   43.5927655000
Cu     7.5036727500    3.2158597500   43.5927655000
Cu     7.5036727500    7.5036727500   43.5927655000
O      2.1439065000    2.1439065000   42.5208122500
O      2.1439065000    6.4317195000   42.5208122500
O      6.4317195000    2.1439065000   42.5208122500
O      6.4317195000    6.4317195000   42.5208122500
O      0.0000000000    0.0000000000   44.6647187500
O      0.0000000000    4.2878130000   44.6647187500
O      4.2878130000    0.0000000000   44.6647187500
O      4.2878130000    4.2878130000   44.6647187500
C      7.8374544210    6.0029380451   47.2231590235
C      6.3856957891    6.3093735268   47.1847260085
C      5.8568798291    7.5383134573   46.9965352261
H      8.1063386427    5.4638831305   48.1468080782
H      8.1035125720    5.3095234924   46.3964106616
H      8.4554510178    6.9063900636   47.1308283651
H      5.7020734826    5.4626932681   47.3117261503
H      4.7827542159    7.7098942445   46.9898841604
H      6.5007646914    8.4087043392   46.8529728506
            """
            pbc = utils.PeriodicBoundaries("16.205584494355413,16.205584494355413,145.85026044919871,90.0,90.0,90.0")
            ac, bo = self.read_fake_files(xyz_file_content, "test_surface_split.xyz")
            surface_indices = set(range(72))
            bo = utils.SolidStateBondDetector.detect_bonds(ac, pbc, surface_indices)
            structure = self.MockStructure(ac)
            model = db.Model("FAKE", "", "")
            model.periodic_boundaries = str(pbc)
            structure.set_model(model)
            mol_results = get_molecules_result(ac, bo, self.settings, str(pbc), surface_indices)
            assert len(mol_results.molecules) == 2
            add_masm_info(structure, bo, self.settings, surface_indices)
            assert "masm_decision_list" in structure.graph_dict
            assert "masm_cbor_graph" in structure.graph_dict
            assert "masm_idx_map" in structure.graph_dict
            cbor = structure.graph_dict['masm_cbor_graph']
            assert cbor.count(';') == 1
