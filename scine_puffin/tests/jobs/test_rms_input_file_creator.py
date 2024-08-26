# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING
import os

from ..testcases import (
    JobTestCase,
)
from ..db_setup import (
    add_compound_and_structure,
    add_reaction,
)
from scine_puffin.tests.testcases import skip_without

from scine_puffin.utilities.imports import module_exists, MissingDependency
if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class RMSInputFileCreatorTest(JobTestCase):

    @skip_without('utilities')
    def test_phase_entry(self):
        """
        Idea of the test: Check if the created dictionaries have the expected format.
        """
        import scine_utilities as utils
        from ...utilities.rms_input_file_creator import create_rms_phase_entry
        aggregate_list = ["63d12175fc016ecdcf53d4e9", "63d12175fc016ecdcf53d4e1"]

        phase_list = create_rms_phase_entry(aggregate_list, [3.0, 2.0], [0.0, 1.0], "Some-Solvent")
        r = utils.MOLAR_GAS_CONSTANT
        reference = [{'Species': [
            {'name': '63d12175fc016ecdcf53d4e9',
             'radicalelectrons': 0,
             'thermo': {
                 'polys': [{
                     'Tmax': 5000.0,
                     'Tmin': 1.0,
                     'coefs': [0.0, 0.0, 0.0, 0.0, 0.0, 3.0 / r, 0.0 / r],
                     'type': 'NASApolynomial'}],
                 'type': 'NASA'},
             'type': 'Species'},
            {'name': '63d12175fc016ecdcf53d4e1',
             'radicalelectrons': 0,
             'thermo': {
                 'polys': [{'Tmax': 5000.0,
                            'Tmin': 1.0,
                            'coefs': [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 / r, 1.0 / r],
                            'type': 'NASApolynomial'}],
                 'type': 'NASA'}, 'type': 'Species'},
            {'name': 'Some-Solvent',
             'radicalelectrons': 0,
             'thermo': {
                 'polys': [{'Tmax': 5000.0,
                            'Tmin': 1.0,
                            'coefs': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            'type': 'NASApolynomial'}],
                 'type': 'NASA'}, 'type': 'Species'}],
            'name': 'phase'}]
        assert reference == phase_list

    @skip_without('database')
    def test_create_rms_reaction_entry(self):
        """
        Idea of the test: Check if the created dictionaries have the expected format.
        """
        from ...utilities.rms_input_file_creator import create_rms_reaction_entry
        n_compounds = 2
        all_compounds = [add_compound_and_structure(self.manager) for _ in range(n_compounds)]
        reactions = self.manager.get_collection("reactions")
        c_ids = [c.id() for c in all_compounds]
        all_reaction_ids = [
            add_reaction(self.manager, [c_ids[0]], [c_ids[1]]).id(),
        ]
        ea = [0.00]
        a = [0.1]
        n = [0]

        reactants = []
        for r_id in all_reaction_ids:
            r = db.Reaction(r_id, reactions)
            reactants.append(([a_id.string() for a_id in r.get_reactants(db.Side.BOTH)[0]],
                              [a_id.string() for a_id in r.get_reactants(db.Side.BOTH)[1]]))
        reaction_list = create_rms_reaction_entry(a, n, ea, reactants)
        reference = [{
            'kinetics': {
                'A': a[0],
                'Ea': ea[0],
                'n': n[0],
                'type': 'Arrhenius'},
            'products': [c_ids[1].string()],
            'reactants': [c_ids[0].string()],
            'type': 'ElementaryReaction'}]
        assert reaction_list == reference

    @skip_without('database')
    def test_create_rms_yml_file(self):
        """
        Idea of the test: Check if the rms yaml input file can be created without problems.
        """
        from ...utilities.rms_input_file_creator import create_rms_yml_file
        n_compounds = 2
        all_compounds = [add_compound_and_structure(self.manager) for _ in range(n_compounds)]
        reactions = self.manager.get_collection("reactions")
        c_ids = [c.id() for c in all_compounds]
        all_reaction_ids = [
            add_reaction(self.manager, [c_ids[0]], [c_ids[1]]).id(),
        ]
        ea = [0.00]
        a = [0.1]
        n = [0]
        h = [9.0, 38.0]
        s = [0.03, 0.69]

        reactants = []
        for r_id in all_reaction_ids:
            r = db.Reaction(r_id, reactions)
            reactants.append(([a_id.string() for a_id in r.get_reactants(db.Side.BOTH)[0]],
                              [a_id.string() for a_id in r.get_reactants(db.Side.BOTH)[1]]))
        file_name = "puffin_test_chem.rms"
        create_rms_yml_file([c.id().string() for c in all_compounds], h, s, a, n, ea, reactants, file_name,
                            "Some-Solvent", 0.5, None)
        os.remove(file_name)
