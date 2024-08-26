# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import TYPE_CHECKING

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_react_job import ReactJob
from scine_puffin.utilities.imports import module_exists, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")


class ScineAfir(ReactJob):
    __doc__ = ("""A job optimizing a structure while applying an artificial force.

    **Order Name**
      ``scine_afir``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:
      All settings recognized by ReaDuct's AFIR task. For a complete list see
      the
      `ReaDuct manual <https://scine.ethz.ch/download/readuct>`_
      The most common AFIR related ones being:

      afir_afir_weak_forces : bool
        This activates an additional, weakly attractiveforce applied to all atom
        pairs. By default set to ``False``.
      afir_afir_attractive : bool
        Specifies whether the artificial force is attractive or repulsive. By
        default set to ``True``, which means that the force is attractive.
      afir_afir_rhs_list : List[int]
        This specifies list of indices of atoms to be artificially forced onto
        or away from those in the LHS list (see below). By default, this list is
        empty. Note that the first atom has the index zero.
      afir_afir_lhs_list : List[int]
        This specifies list of indices of atoms to be artificially forced onto
        or away from those in the RHS list (see above). By default, this list is
        empty. Note that the first atom has the index zero.
      afir_afir_energy_allowance : float
        The maximum amount of energy to be added by the artificial force, in
        kJ/mol. By default set to 1000 kJ/mol.
      afir_afir_phase_in : int
        The number of steps over which the full force is gradually applied.
        By default set to 100.

    """ + "\n"
               + ReactJob.optional_settings_doc() + "\n"
               + ReactJob.general_calculator_settings_docstring() + "\n"
               + ReactJob.generated_data_docstring() + "\n" +
               """
      If successful the following data will be generated and added to the
      database:

      Structures
        A new structure, the optimized (inc. artificial force) structure.
      Properties
        The ``electronic_energy`` associated with the new structure.
    """

               + ReactJob.required_packages_docstring()
               )

    def __init__(self) -> None:
        super().__init__()
        self.name = "Scine AFIR Job"
        afir_defaults = {
            "output": ["afir"],
            "convergence_max_iterations": 500,
        }
        opt_defaults = {
            "output": ["opt"],
            "convergence_max_iterations": 500,
        }
        self.settings = {
            **self.settings,
            "afir": afir_defaults,
            self.opt_key: opt_defaults,
        }

    @job_configuration_wrapper
    def run(self, manager: db.Manager, calculation: db.Calculation, config: Configuration) -> bool:

        import scine_readuct as readuct

        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)

        with calculation_context(self):
            """ preparation """
            if len(calculation.get_structures()) > 1:
                raise RuntimeError(self.name + " is only meant for a single structure!")
            settings_manager.separate_settings(self._calculation.get_settings())
            self.sort_settings(settings_manager.task_settings)

            self.systems, keys = settings_manager.prepare_readuct_task(
                structure, calculation, calculation.get_settings(), config["resources"]
            )
            if program_helper is not None:
                program_helper.calculation_preprocessing(self.get_system(keys[0]), calculation.get_settings())

            """ AFIR Optimization """
            print("Afir Settings:")
            print(self.settings["afir"], "\n")
            self.systems, success = readuct.run_afir_task(self.systems, keys, **self.settings['afir'])
            self.throw_if_not_successful(success, self.systems, keys, ["energy"], "AFIR optimization failed:\n")

            """ Endpoint Optimization """
            product_names, self.systems = self.optimize_structures(
                "product", self.systems,
                [self.get_system(self.output('afir')[0]).structure],
                [structure.get_charge()],
                [structure.get_multiplicity()],
                settings_manager.calculator_settings
            )
            if len(product_names) != 1:
                self.raise_named_exception("Optimization of the product yielded multiple structures, "
                                           "which is not expected")
            lowest_name, names_within_range = self._get_propensity_names_within_range(
                product_names[0], self.systems, self.settings[self.propensity_key]["energy_range_to_optimize"]
            )
            if lowest_name is None:
                self.raise_named_exception("Product optimization was not successful")
                raise RuntimeError("Unreachable")

            old_label = structure.get_label()
            db_results = calculation.get_results()
            for product in [lowest_name] + names_within_range:
                graph, self.systems = self.make_graph_from_calc(self.systems, product)
                new_label = self.determine_new_label(old_label, graph, structure.has_property("surface_atom_indices"))

                new_structure = self.optimization_postprocessing(success, self.systems, [product], structure,
                                                                 new_label, program_helper, ['energy', 'bond_orders'])
                bond_orders = self.get_system(product_names[0]).get_results().bond_orders
                assert bond_orders is not None

                self.store_property(
                    self._properties,
                    "bond_orders",
                    "SparseMatrixProperty",
                    bond_orders.matrix,
                    self._calculation.get_model(),
                    self._calculation,
                    new_structure,
                )
                self.add_graph(new_structure, bond_orders)
                db_results += calculation.get_results()

            calculation.set_results(db_results)
        return self.postprocess_calculation_context()
