# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import sys
from typing import TYPE_CHECKING

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_optimization_job import OptimizationJob
from .templates.scine_propensity_job import ScinePropensityJob
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ScineGeometryOptimization(OptimizationJob, ScinePropensityJob):
    __doc__ = ("""
    A job optimizing the geometry of a given structure, in search of a local
    minimum on the potential energy surface.
    Optimizing a given structure's geometry, generating a new minimum energy
    structure, if successful.

    **Order Name**
      ``scine_geometry_optimization``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      All settings recognized by ReaDuct's geometry optimization task, which must be prepended by ``opt`` prefix.

      Common examples are:

      opt_optimizer : str
         The name of the optimizer to be used, e.g. 'bfgs', 'lbfgs', 'nr' or
         'sd'.
      opt_convergence_max_iterations : int
         The maximum number of geometry optimization cycles.
      opt_convergence_delta_value : float
         The convergence criterion for the electronic energy difference between
         two steps.
      opt_convergence_gradient_max_coefficient : float
         The convergence criterion for the maximum absolute gradient.
         contribution.
      opt_convergence_step_rms : float
         The convergence criterion for root mean square of the geometric
         gradient.
      opt_convergence_step_max_coefficient : float
         The convergence criterion for the maximum absolute coefficient in the
         last step taken in the geometry optimization.
      opt_convergence_gradient_rms : float
         The convergence criterion for root mean square of the last step taken
         in the geometry optimization.

      For a complete list see the
      `ReaDuct manual <https://scine.ethz.ch/download/readuct>`_

    """ + "\n"
               + OptimizationJob.optional_settings_doc() + "\n" + ScinePropensityJob.optional_settings_doc() + "\n"
               + ScinePropensityJob.general_calculator_settings_docstring() + "\n"
               + ScinePropensityJob.generated_data_docstring() + "\n" +
               """

      If successful the following data will be generated and added to the
      database:

      Structures
        A new minimum energy structure.
      Properties
        The ``electronic_energy`` associated with the new structure.
    """
               + ScinePropensityJob.required_packages_docstring()
               )

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine Geometry Optimization"
        self.settings = {
            **self.settings,
            self.propensity_key: {
                **self.settings[self.propensity_key],
                "check_for_unimolecular_reaction": True,
                "energy_range_to_save": 100.0,
                "optimize_all": False,
                "energy_range_to_optimize": 250.0,
                "check": 0,
            }
        }

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:
        from scine_utilities import settings_names as sn

        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)

        with calculation_context(self):
            """ preparation """
            if len(calculation.get_structures()) > 1:
                raise RuntimeError(self.name + " is only meant for a single structure!")
            settings_manager.separate_settings(self._calculation.get_settings())
            self.sort_settings(settings_manager.task_settings)

            systems, keys = settings_manager.prepare_readuct_task(
                structure, calculation, calculation.get_settings(), config["resources"]
            )
            if program_helper is not None:
                program_helper.calculation_preprocessing(self.get_calc(keys[0], systems), calculation.get_settings())

            optimize_cell: bool = "unitcelloptimizer" in self.settings[self.opt_key] \
                                  and len(self.settings[self.opt_key]["unitcelloptimizer"]) > 0
            opt_names, systems = self.optimize_structures(
                "system",
                systems,
                [structure.get_atoms()],
                [structure.get_charge()],
                [structure.get_multiplicity()],
                settings_manager.calculator_settings,
            )
            if len(opt_names) != 1:
                self.raise_named_exception("Optimization of the structure yielded multiple structures, "
                                           "which is not expected")
            lowest_name, _ = self._get_propensity_names_within_range(
                opt_names[0], systems, self.settings[self.propensity_key]["energy_range_to_optimize"]
            )
            if lowest_name is None:
                self.raise_named_exception("No optimization was successful.")
                raise RuntimeError("Unreachable")
            if lowest_name != opt_names[0]:
                sys.stderr.write(f"Warning: Specified the spin multiplicity '{structure.get_multiplicity()}', but "
                                 f"the system reached a lower energy with the spin multiplicity "
                                 f"'{self.get_multiplicity(self.get_calc(lowest_name, systems))}'.\n"
                                 f"Continuing with the latter.\n")
                opt_names[0] = lowest_name

            opt_calc = self.get_calc(opt_names[0], systems)
            if optimize_cell:
                # require to change the calculator settings, to avoid model completion failure
                model = calculation.get_model()
                old_pbc = model.periodic_boundaries
                new_pbc = opt_calc.settings[sn.periodic_boundaries]
                opt_calc.settings[sn.periodic_boundaries] = old_pbc

            # Graph generation
            graph, systems = self.make_graph_from_calc(systems, opt_names[0])
            old_label = structure.get_label()
            new_label = self.determine_new_label(old_label, graph, structure.has_property("surface_atom_indices"))

            new_structure = self.optimization_postprocessing(
                True, systems, opt_names, structure, new_label, program_helper
            )
            if graph:
                new_structure.set_graph("masm_cbor_graph", graph)

            if optimize_cell:
                assert isinstance(new_pbc, str)
                # update model of new structure to match the optimized unit cell
                model = new_structure.get_model()
                model.periodic_boundaries = new_pbc
                new_structure.set_model(model)

        return self.postprocess_calculation_context()
