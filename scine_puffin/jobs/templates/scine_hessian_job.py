# -*- coding: utf-8 -*-
from __future__ import annotations
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
from abc import ABC
from typing import TYPE_CHECKING, List

import numpy as np

from .scine_job import ScineJob
from .job import is_configured
from scine_puffin.utilities.imports import module_exists, requires, MissingDependency

if module_exists("scine_database") or TYPE_CHECKING:
    import scine_database as db
else:
    db = MissingDependency("scine_database")
if module_exists("scine_utilities") or TYPE_CHECKING:
    import scine_utilities as utils
else:
    utils = MissingDependency("scine_utilities")


class HessianJob(ScineJob, ABC):
    """
    A common interface for all jobs in Puffin that use the Scine::Core::Calculator interface to calculate a Hessian
    and carry out a Thermochemistry analysis.
    """

    own_expected_results = ["energy", "hessian", "thermochemistry"]

    def __init__(self) -> None:
        super().__init__()
        self.name = "HessianJob"

    @staticmethod
    def required_programs() -> List[str]:
        return ["database", "readuct", "utils"]

    @is_configured
    @requires("utilities")
    def store_hessian_data(self, system: utils.core.Calculator, structure: db.Structure) -> None:
        """
        Stores results from a Hessian calculation and Thermochemistry for the specified structure based on the given
        calculator. Does not perform checks.

        Notes
        -----
        * Requires run configuration

        Parameters
        ----------
        system : utils.core.Calculator (Scine::Core::Calculator)
            A Scine calculator holding a results object with energy, Hessian, and Thermochemistry properties.
        structure : db.Structure (Scine::Database::Structure)
            A structure for which the property is saved.
        """
        results = system.get_results()
        if results.energy is None:
            self.raise_named_exception(f"{system.name()} is missing energy result")
            return  # unreachable only for linter
        if not structure.has_property("electronic_energy"):
            self.store_energy(system, structure)
        if results.hessian is None and results.partial_hessian is None:
            self.raise_named_exception(f"{system.name()} is missing Hessian result")
            return  # unreachable only for linter

        model = self.get_model()
        if results.partial_hessian is not None:
            tmp_hessian_for_db: np.ndarray = results.partial_hessian.matrix
            # NOTE: Requires Int vector as property
            self.store_property(
                self._properties,
                "qm_atoms",
                "VectorProperty",
                results.partial_hessian.indices,
                model,
                self._calculation,
                structure,
            )
        else:
            tmp_hessian_for_db = results.hessian  # type: ignore

        self.store_property(
            self._properties,
            "hessian",
            "DenseMatrixProperty",
            tmp_hessian_for_db,
            model,
            self._calculation,
            structure,
        )

        # NOTE: only do normal modes and thermochemistry if Hessian is available
        # do not store for Partial Hessian, gets too big and crashes DB, should be sparse
        if results.hessian is not None:
            # Get normal modes and frequencies
            atoms = structure.get_atoms()
            modes_container = utils.normal_modes.calculate(results.hessian, atoms.elements, atoms.positions)
            # Wavenumbers in cm-1
            wavenumbers = modes_container.get_wave_numbers()
            # Frequencies in atomic units
            frequencies = np.array(wavenumbers) * utils.HARTREE_PER_INVERSE_CENTIMETER / (2 * utils.PI)
            # Get normal modes: Flattened mass-weighted eigenvectors/normal modes as matrix columns
            # lengths are in a.u. and masses in u
            modes = np.column_stack([modes_container.get_mode(i).flatten() for i in range(modes_container.size())])

            # store properties
            self.store_property(
                self._properties,
                "normal_modes",
                "DenseMatrixProperty",
                modes,
                model,
                self._calculation,
                structure,
            )
            self.store_property(
                self._properties,
                "frequencies",
                "VectorProperty",
                frequencies,
                model,
                self._calculation,
                structure,
            )

            thermo_container = results.thermochemistry
            # NOTE: could avoid thermochemistry with `and results.partial_hessian is None`, if one does not want it
            if thermo_container is None:
                thermo_calculator = utils.ThermochemistryCalculator(results.hessian, atoms,
                                                                    structure.get_multiplicity(), results.energy)
                thermo_calculator.set_temperature(float(model.temperature))
                thermo_calculator.set_pressure(float(model.pressure))
                thermo_container = thermo_calculator.calculate()
            self.store_property(
                self._properties,
                "gibbs_free_energy",
                "NumberProperty",
                thermo_container.overall.gibbs_free_energy,
                model,
                self._calculation,
                structure,
            )
            self.store_property(
                self._properties,
                "gibbs_energy_correction",
                "NumberProperty",
                thermo_container.overall.gibbs_free_energy - results.energy,
                model,
                self._calculation,
                structure,
            )
