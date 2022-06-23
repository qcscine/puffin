# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_react_job import ReactJob


class ScineAfir(ReactJob):
    """
    A job optimizing a structure while applying an artificial force.

    **Order Name**
      ``scine_afir``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:
      All settings recognized by ReaDuct's AFIR task. For a complete list see
      the
      `ReaDuct manual <https://scine.ethz.ch/static/download/readuct_manual.pdf>`_
      The most common AFIR related ones being:

      afir_afir_weak_forces :: bool
        This activates an additional, weakly attractiveforce applied to all atom
        pairs. By default set to ``False``.
      afir_afir_attractive :: bool
        Specifies whether the artificial force is attractive or repulsive. By
        default set to ``True``, which means that the force is attractive.
      afir_afir_rhs_list :: List[int]
        This specifies list of indices of atoms to be artificially forced onto
        or away from those in the LHS list (see below). By default, this list is
        empty. Note that the first atom has the index zero.
      afir_afir_lhs_list :: List[int]
        This specifies list of indices of atoms to be artificially forced onto
        or away from those in the RHS list (see above). By default, this list is
        empty. Note that the first atom has the index zero.
      afir_afir_energy_allowance :: float
        The maximum amount of energy to be added by the artificial force, in
        kJ/mol. By default set to 1000 kJ/mol.
      afir_afir_phase_in :: int
        The number of steps over which the full force is gradually applied.
        By default set to 100.

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
        A new structure, the optimized (inc. artificial force) structure.
      Properties
        The ``electronic_energy`` associated with the new structure.
    """

    def __init__(self):
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
            "opt": opt_defaults,
        }

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:

        import scine_database as db
        import scine_readuct as readuct

        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)

        with calculation_context(self):
            """ preparation """
            if len(calculation.get_structures()) > 1:
                raise RuntimeError(self.name + " is only meant for a single structure!")
            settings_manager.separate_settings(self._calculation.get_settings())
            self.sort_settings(settings_manager.task_settings)
            new_label = self.determine_new_label(structure)

            self.systems, keys = settings_manager.prepare_readuct_task(
                structure, calculation, calculation.get_settings(), config["resources"]
            )
            if program_helper is not None:
                program_helper.calculation_preprocessing(self.systems[keys[0]], calculation.get_settings())

            """ AFIR Optimization """
            print("Afir Settings:")
            print(self.settings["afir"], "\n")
            self.systems, success = readuct.run_afir_task(self.systems, keys, **self.settings['afir'])
            self.throw_if_not_successful(success, self.systems, keys, ["energy"], "AFIR optimization failed:\n")

            """ Endpoint Optimization """
            product_names = self.optimize_structures(
                "product",
                [self.systems[self.output('afir')[0]].structure],
                [structure.get_charge()],
                [structure.get_multiplicity()],
                settings_manager.calculator_settings
            )
            new_structure = self.optimization_postprocessing(success, self.systems, product_names, structure,
                                                             new_label, program_helper, ['energy', 'bond_orders'])
            self.store_property(
                self._properties,
                "bond_orders",
                "SparseMatrixProperty",
                self.systems[product_names[0]].get_results().bond_orders.matrix,
                self._calculation.get_model(),
                self._calculation,
                new_structure,
            )
            self.add_graph(new_structure, self.systems[product_names[0]].get_results().bond_orders)

        return self.postprocess_calculation_context()
