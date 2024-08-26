# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import sys
from typing import TYPE_CHECKING, Dict, Optional

from scine_puffin.config import Configuration
from .templates.job import breakable, calculation_context, job_configuration_wrapper
from ..utilities.task_to_readuct_call import SubTaskToReaductCall
from .scine_dissociation_cut import ScineDissociationCut
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_database")
if module_exists("scine_molassembler") or TYPE_CHECKING:
    import scine_molassembler as masm
else:
    masm = MissingDependency("scine_database")


class ScineDissociationCutWithOptimization(ScineDissociationCut):
    """
    Identical to :py:class:`ScineDissociationCut`, but does not assume that the given structure is a minimum,
    e.g., because this job is carried out with a different model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine React Job optimization and bond cutting"

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        # Everything that calls SCINE is enclosed in a try/except block
        with (breakable(calculation_context(self))):
            """ sanity checks """
            if len(calculation.get_structures()) != 1:
                self.raise_named_exception(f"{self.name} is only implemented for single molecule system.")

            # preprocessing of structure
            self.ref_structure = db.Structure(calculation.get_structures()[0], self._structures)
            settings_manager, program_helper = self.create_helpers(self.ref_structure)

            # Separate the calculation settings from the database into the task and calculator settings
            # This overwrites any default settings by user settings
            settings_manager.separate_settings(self._calculation.get_settings())
            settings_manager.correct_non_applicable_settings()
            settings_manager.calculator_settings[utils.settings_names.molecular_charge] = \
                self.ref_structure.get_charge()
            settings_manager.calculator_settings[utils.settings_names.spin_multiplicity] = \
                self.ref_structure.get_multiplicity()
            self.sort_settings(settings_manager.task_settings)
            """ Setup calculator """
            self.systems: Dict[str, Optional[utils.core.Calculator]] = {}
            utils.io.write("system.xyz", self.ref_structure.get_atoms())
            self.systems[self.rc_key] = utils.core.load_system_into_calculator(
                "system.xyz",
                self._calculation.get_model().method_family,
                **settings_manager.calculator_settings,
            )
            if program_helper is not None:
                program_helper.calculation_preprocessing(
                    self.get_system(self.rc_key), self._calculation.get_settings())

            # Calculate bond orders and graph of reactive complex and compare to database graph of start structures
            self.start_graph, self.systems = self.make_graph_from_calc(self.systems, self.rc_key)

            # optimize
            self.systems = self.observed_readuct_call_with_throw(SubTaskToReaductCall.OPT, self.systems,
                                                                 [self.rc_key], ['energy'],
                                                                 "Optimization of initial structure failed.",
                                                                 **self.settings[self.rc_opt_system_name])
            opt_graph, self.systems = self.make_graph_from_calc(self.systems, self.rc_opt_system_name)
            if ";" in opt_graph:
                self.raise_named_exception("Structure decomposed in optimization.")
            calc = self.get_system(self.rc_opt_system_name)
            self.ref_structure = db.Structure.make(calc.structure,
                                                   self.get_charge(calc),
                                                   self.get_multiplicity(calc),
                                                   self._calculation.get_model(), self.ref_structure.get_label(),
                                                   self._structures)
            self.ref_structure.add_calculation(self._calculation.get_job().order, self._calculation.id())
            db_results = self._calculation.get_results()
            db_results.clear()
            db_results.add_structure(self.ref_structure.id())
            self._calculation.set_results(db_results)
            if not masm.JsonSerialization.equal_molecules(self.start_graph, opt_graph):
                sys.stderr.write("Warning: Graphs of optimized and initial structure differ.\n")
                self.start_graph = opt_graph

            self._dissociation_impl(settings_manager, program_helper)

        return self.postprocess_calculation_context()
