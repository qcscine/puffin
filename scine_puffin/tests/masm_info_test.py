#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
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
    from scine_puffin.utilities.masm_helper import add_masm_info, _modify_based_on_distances
except ImportError:
    pass
else:
    class MasmHelperTests(unittest.TestCase):
        def __init__(self, *args, **kwargs):
            super(MasmHelperTests, self).__init__(*args, **kwargs)
            self.settings = {
                "sub_based_on_distance_connectivity": False,
                "add_based_on_distance_connectivity": False,
            }

        class MockStructure(object):
            atoms: utils.AtomCollection
            graph_dict: dict = {}
            model: db.Model = db.Model("", "", "")

            def __init__(self, atoms: utils.AtomCollection):
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
