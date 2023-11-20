# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_optimization_job import OptimizationJob
from .templates.scine_connectivity_job import ConnectivityJob


class ScineGeometryOptimization(OptimizationJob, ConnectivityJob):
    """
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

      All settings recognized by ReaDuct's geometry optimization task.

      Common examples are:

      optimizer :: str
         The name of the optimizer to be used, e.g. 'bfgs', 'lbfgs', 'nr' or
         'sd'.
      convergence_max_iterations :: int
         The maximum number of geometry optimization cycles.
      convergence_delta_value :: float
         The convergence criterion for the electronic energy difference between
         two steps.
      convergence_gradient_max_coefficient :: float
         The convergence criterion for the maximum absolute gradient.
         contribution.
      convergence_step_rms :: float
         The convergence criterion for root mean square of the geometric
         gradient.
      convergence_step_max_coefficient :: float
         The convergence criterion for the maximum absolute coefficient in the
         last step taken in the geometry optimization.
      convergence_gradient_rms :: float
         The convergence criterion for root mean square of the last step taken
         in the geometry optimization.

      For a complete list see the
      `ReaDuct manual <https://scine.ethz.ch/static/download/readuct_manual.pdf>`_

      All settings that are recognized by the SCF program chosen.

      Common examples are:

      max_scf_iterations :: int
         The number of allowed SCF cycles until convergence.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Readuct (present by default)
      - SCINE: Utils (present by default)
      - A program implementing the SCINE Calculator interface, e.g. Sparrow

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Structures
        A new minimum energy structure.
      Properties
        The ``electronic_energy`` associated with the new structure.
    """

    def __init__(self):
        super().__init__()
        self.name = "Scine Geometry Optimization"

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        self.run_geometry_optimization(calculation, config)
        return self.postprocess_calculation_context()

    def run_geometry_optimization(self, calculation, config):
        import scine_database as db
        import scine_readuct as readuct
        from scine_utilities import settings_names as sn

        # preprocessing of structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)

        # actual calculation
        with calculation_context(self):
            systems, keys = settings_manager.prepare_readuct_task(
                structure, calculation, calculation.get_settings(), config["resources"]
            )
            if program_helper is not None:
                program_helper.calculation_preprocessing(systems[keys[0]], calculation.get_settings())
            optimize_cell: bool = "unitcelloptimizer" in settings_manager.task_settings \
                                  and settings_manager.task_settings["unitcelloptimizer"]
            systems, success = readuct.run_opt_task(systems, keys, **settings_manager.task_settings)

            if optimize_cell:
                # require to change the calculator settings, to avoid model completion failure
                model = calculation.get_model()
                old_pbc = model.periodic_boundaries
                new_pbc = systems[keys[0]].settings[sn.periodic_boundaries]
                systems[keys[0]].settings[sn.periodic_boundaries] = old_pbc

            # Graph generation
            if success:
                graph, systems = self.make_graph_from_calc(systems, keys[0])
                new_label = self.determine_new_label(structure, graph)
            else:
                new_label = db.Label.IRRELEVANT

            if graph:
                structure.set_graph("masm_cbor_graph", graph)

            t = self.optimization_postprocessing(
                success, systems, keys, structure, new_label, program_helper
            )

            if optimize_cell:
                # update model of new structure to match the optimized unit cell
                new_structure = db.Structure(calculation.get_results().structure_ids[0], self._structures)
                model = new_structure.get_model()
                model.periodic_boundaries = new_pbc
                new_structure.set_model(model)
            return t
