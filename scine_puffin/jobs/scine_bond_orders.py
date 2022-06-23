# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from scine_puffin.config import Configuration
from .templates.job import calculation_context, job_configuration_wrapper
from .templates.scine_connectivity_job import ConnectivityJob


class ScineBondOrders(ConnectivityJob):
    """
    A job calculating bond orders.

    **Order Name**
      ``scine_bond_orders``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      only_distance_connectivity :: bool
        Whether the bond orders shall be constructed via distance information only (True)
        or from an electronic structure calculation (False). (default: False)
      add_based_on_distance_connectivity :: bool
        If ``True``, the structure's connectivity is derived from interatomic
        distances via the utils.BondDetector: The bond orders used for
        interpretation are set to the maximum between those given by an electronic structure
        calculation and 1.0, whereever the utils.BondDetector
        detects a bond. (default: True)
      sub_based_on_distance_connectivity :: bool
        If ``True``, the structure's connectivity is derived from interatomic
        distances via the utils.BondDetector: The bond orders used given by an electronic structure
        calculation are removed, whereever the utils.BondDetector does not
        detect a bond. (default: True)
      add_graph :: bool
        Whether to add a molassembler graph and decision list to the structure
        based on the determined bond orders. (default: True)

      All settings that are recognized by the SCF program chosen.

      Common examples are:

      max_scf_iterations :: int
        The number of allowed SCF cycles until convergence.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: molassembler (present by default)
      - SCINE: Readuct (present by default)
      - SCINE: Utils (present by default)
      - A program implementing the SCINE Calculator interface, e.g. Sparrow

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Properties
        The ``bond_orders`` (``SparseMatrixProperty``) are added.
        Optionally the ``electronic_energy`` associated with the structure if it
        is present in the results of provided by the calculator interface.
      Other
        If a graph is requested, graph representations of the structure will be
        added to the structures ``graphs`` field. The added representations are:
        A representation of the graph ``masm_cbor_graph``, and the decision
        representations of the existing stereopermutators using a nearest
        neighbour fit ``masm_decision_list``.
        Any previous graph representations of the structure will be overwritten.
    """

    def __init__(self):
        super().__init__()
        self.name = "Scine Bond Order Job"

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:

        import scine_database as db
        import scine_readuct as readuct

        # Get structure
        structure = db.Structure(calculation.get_structures()[0], self._structures)
        settings_manager, program_helper = self.create_helpers(structure)
        settings_manager.separate_settings(calculation.get_settings())

        # Get connectivity settings and remove them from settings passed to readuct
        self.extract_connectivity_settings_from_dict(settings_manager.task_settings)
        add_graph = settings_manager.task_settings.pop("add_graph", True)

        # actual calculation
        success = False  # success might not be set if something throws in context -> ensure it exists in scope
        with calculation_context(self):
            # Distance based bond orders
            if self.connectivity_settings["only_distance_connectivity"]:
                bond_orders = self.distance_bond_orders(structure)
            # Bond order calculation with readuct
            else:
                systems, keys = settings_manager.prepare_readuct_task(
                    structure,
                    calculation,
                    settings_manager.task_settings,
                    config["resources"],
                )
                if program_helper is not None:
                    program_helper.calculation_preprocessing(
                        systems[keys[0]], settings_manager.task_settings
                    )
                systems, success = readuct.run_single_point_task(
                    systems, keys, require_bond_orders=True, **settings_manager.task_settings
                )

                self.throw_if_not_successful(success, systems, keys)
                bond_orders = systems[keys[0]].get_results().bond_orders

            # Graph generation
            if add_graph:
                self.add_graph(structure, bond_orders)

            # There are no results if the bond orders are purely distance based
            if self.connectivity_settings["only_distance_connectivity"]:
                self.verify_connection()
                self.capture_raw_output()
            else:
                self.sp_postprocessing(
                    success, systems, keys, structure, program_helper
                )

            # Store bond orders
            self.store_property(
                self._properties,
                "bond_orders",
                "SparseMatrixProperty",
                bond_orders.matrix,
                calculation.get_model(),
                calculation,
                structure,
            )

        return self.postprocess_calculation_context()
