# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import numpy as np

import scine_database as db
import scine_utilities as utils


from .job import job_configuration_wrapper
from .scine_job import ScineJob
from scine_puffin.config import Configuration


class HessianJob(ScineJob):
    """
    A common interface for all jobs in Puffin that use the Scine::Core::Calculator interface to calculate a Hessian
    and carry out a Thermochemistry analysis.
    """

    def __init__(self):
        super().__init__()
        self.name = "HessianJob"
        self.own_expected_results = ["energy", "hessian", "thermochemistry"]

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        """See Job.run()"""
        raise NotImplementedError

    @staticmethod
    def required_programs():
        return ["database", "readuct", "utils"]

    def store_hessian_data(self, system: utils.core.Calculator, structure: db.Structure) -> None:
        """
        Stores results from a Hessian calculation and Thermochemistry for the specified structure based on the given
        calculator. Does not perform checks.

        Notes
        -----
        * Requires run configuration

        Parameters
        ----------
        system :: core.calculator (Scine::Core::Calculator)
            A Scine calculator holding a results object with energy, Hessian, and Thermochemistry properties.
        structure :: db.Structure (Scine::Database::Structure)
            A structure for which the property is saved.
        """
        results = system.get_results()
        if results.energy is None:
            self.raise_named_exception(f"{system.name()} is missing energy result")
            return  # unreachable only for linter
        if not structure.has_property("electronic_energy"):
            self.store_energy(system, structure)
        if results.hessian is None:
            self.raise_named_exception(f"{system.name()} is missing Hessian result")
            return  # unreachable only for linter
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
        model = self._calculation.get_model()

        # store properties
        self.store_property(
            self._properties,
            "hessian",
            "DenseMatrixProperty",
            results.hessian,
            model,
            self._calculation,
            structure,
        )
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
        if thermo_container is None:
            thermo_calculator = utils.ThermochemistryCalculator(results.hessian, atoms, structure.get_multiplicity(),
                                                                results.energy)
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
