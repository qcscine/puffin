#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import unittest
import os
import pathlib

from scine_puffin.config import Configuration
from ..testcases import skip_without


class Cp2kTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Cp2kTests, self).__init__(*args, **kwargs)
        self.resources_directory = os.path.join(pathlib.Path(__file__).parent.resolve(), "files")

    @skip_without('database')
    def test_setup_for_single_point_calculations(self):
        import scine_database as db
        import pytest
        from ..db_setup import get_clean_db
        from scine_puffin.utilities.program_helper import Cp2kHelper

        manager = get_clean_db("puffin_unittests_cp2k")
        config = Configuration()
        config.load()
        config["daemon"]["mode"] = "debug"

        calculations = manager.get_collection("calculations")
        properties = manager.get_collection("properties")
        structures = manager.get_collection("structures")

        # The test molecule is the Si unit cell
        model = db.Model("DFT", "PADE", "SZV-MOLOPT-GTH")
        model.program = "cp2k"
        model.periodic_boundaries = "5.4306975, 5.4306975, 5.4306975, 90, 90, 90, xyz"
        job = db.Job("scine_single_point")
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(self.resources_directory, "system.xyz"), 0, 1)
        calculation = db.Calculation()
        calculation.link(calculations)
        calculation.create(model, job, [structure.id()])
        settings = calculation.get_settings()
        cutoff = 350.0
        settings["plane_wave_cutoff"] = cutoff
        calculation.set_settings(settings)

        with pytest.raises(RuntimeError) as excinfo:
            _ = Cp2kHelper(manager, structure, calculation)
        assert "Only specified one of the two cutoff" in str(excinfo.value)

        rel_cutoff = 70.0
        settings["relative_multi_grid_cutoff"] = rel_cutoff
        calculation.set_settings(settings)
        cp2k_helper = Cp2kHelper(manager, structure, calculation)

        # test transfer to new structure
        new_structure = db.Structure()
        new_structure.link(structures)
        new_structure.create(os.path.join(self.resources_directory, "system.xyz"), 0, 1)
        assert not cp2k_helper.structure_has_cutoff_properties(structure)
        assert not cp2k_helper.structure_has_cutoff_properties(new_structure)

        prop1 = db.NumberProperty()
        prop1.link(properties)
        id1 = prop1.create(model, cp2k_helper.cutoff_name, structure.id(), calculation.id(), cutoff)
        prop2 = db.NumberProperty()
        prop2.link(properties)
        id2 = prop2.create(
            model,
            cp2k_helper.rel_cutoff_name,
            structure.id(),
            calculation.id(),
            rel_cutoff,
        )
        structure.set_property(cp2k_helper.cutoff_name, id1)
        structure.set_property(cp2k_helper.rel_cutoff_name, id2)

        # now finds properties
        assert cp2k_helper.structure_has_cutoff_properties(structure)
        cp2k_helper.calculation_postprocessing(calculation, structure, new_structure)
        # has saved properties for new structures
        assert new_structure.has_property(cp2k_helper.cutoff_name)
        assert new_structure.has_property(cp2k_helper.rel_cutoff_name)
        # properties are linked instead of created new
        assert new_structure.get_property(cp2k_helper.cutoff_name) == id1
        assert new_structure.get_property(cp2k_helper.rel_cutoff_name) == id2

        manager.wipe()
        return manager, structure, calculation, config

    @skip_without('cp2k', 'database', 'readuct')
    def test_cp2k_calculation(self):
        from scine_puffin.utilities import scine_helper
        from scine_puffin.utilities.program_helper import Cp2kHelper
        import scine_readuct as readuct

        manager, structure, calculation, config = self.test_setup_for_single_point_calculations()
        cp2k_helper = Cp2kHelper(manager, structure, calculation)
        # test calculation
        settings_manager = scine_helper.SettingsManager(calculation.get_model().method_family, "cp2k")
        systems, keys = settings_manager.prepare_readuct_task(structure, calculation,
                                                              calculation.get_settings(), config["resources"])
        cp2k_helper.calculation_preprocessing(systems[keys[0]], calculation.get_settings())
        systems, success = readuct.run_sp_task(systems, keys, **settings_manager.task_settings)
        assert success
