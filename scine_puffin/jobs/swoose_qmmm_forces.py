# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import os
from scine_puffin.config import Configuration
from .templates.job import Job, calculation_context, job_configuration_wrapper


class SwooseQmmmForces(Job):
    """
    A job calculating the forces for a given structure with the QM/MM method
    provided by the Scine Swoose Module.

    **Order Name**
      ``swoose_qmmm_forces``

    **Optional Settings**
      Optional settings are read from the ``settings`` field, which is part of
      any ``Calculation`` stored in a SCINE Database.
      Possible settings for this job are:

      All settings that are recognized by the Swoose binary for the QM/MM
      method. This includes the pure MM and the pure QM settings.

      Common examples are:

      max_scf_iterations :: int
         The number of allowed SCF cycles until convergence.

    **Required Packages**
      - SCINE: Database (present by default)
      - SCINE: Utils (present by default)
      - SCINE: Swoose
      - A program that implements a QM calculator, e.g., Sparrow or Orca.

    **Generated Data**
      If successful the following data will be generated and added to the
      database:

      Properties
        The ``electronic_energy`` associated with the given structure.
        The ``atomic_forces`` associated with the given structure.
    """

    def write_parameter_and_connectivity_file(self, parameter_file: str, connectivity_file: str,
                                              settings, structure, properties):
        """
        This function needs to be called inside the calculation context, because it writes files to
        strange places otherwise.
        """

        import scine_database as db
        try:
            parameters = db.StringProperty(structure.get_property('sfam_parameters'))
            bond_orders = db.SparseMatrixProperty(structure.get_property('bond_orders'))
        except RuntimeError as e:
            raise RuntimeError('Parameters or bond orders are missing as properties of the structure.') from e
        parameters.link(properties)
        bond_orders.link(properties)
        n_atoms = len(structure.get_atoms())
        bo_matrix = bond_orders.get_data().toarray()
        if bo_matrix.shape != (n_atoms, n_atoms):
            raise RuntimeError('The dimensions of the provided bond orders are incompatible with the structure.')
        with open(parameter_file, 'w') as p_file:
            p_file.write(parameters.get_data())
        with open(connectivity_file, 'w') as c_file:
            for i in range(n_atoms):
                neighbors = ''
                for j in range(n_atoms):
                    if i == j or bo_matrix[i, j] < 0.5:
                        continue
                    neighbors += str(j) + ' '
                c_file.write(neighbors + '\n')
        settings.update({'mm_parameter_file': parameter_file, 'mm_connectivity_file': connectivity_file})
        return settings

    def parse_energy(self, output: str):
        try:
            lines = output.split('\n')
            for line in lines:
                if line.startswith('# Energy'):
                    energy = float(line.split()[3])
                    return energy
            return None
        except (IndexError, ValueError):
            return None

    def parse_forces(self, output: str, n_atoms: int):
        import scine_utilities as utils
        try:
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('# Gradients'):
                    forces = []
                    for j in range(i + 2, i + 2 + n_atoms):
                        force = [-float(val) * utils.HARTREE_PER_KCALPERMOL for val in lines[j].split()[1:]]
                        forces.append(force)
                    return forces
            return None
        except (IndexError, ValueError):
            return None

    def process_output(self, calculation, atoms):
        import scine_database as db

        # Capture raw output and parse the results from it
        stdout_path = os.path.join(self.work_dir, 'output')
        stderr_path = os.path.join(self.work_dir, 'errors')
        with open(stdout_path, 'r') as stdout, open(stderr_path, 'r') as stderr:
            calculation.set_raw_output(stdout.read() + '\n' + stderr.read())

        # Parse energy
        energy = self.parse_energy(calculation.get_raw_output())

        # Parse forces
        forces = self.parse_forces(calculation.get_raw_output(), len(atoms))

        # Check whether everything worked
        if energy is None or forces is None:
            calculation.set_status(db.Status.FAILED)
            raise RuntimeError("Something went wrong while parsing energies or forces.")

        dimension_is_correct = len(forces) == len(atoms)
        for force in forces:
            if len(force) != 3:
                dimension_is_correct = False
                break

        if not dimension_is_correct:
            calculation.set_status(db.Status.FAILED)
            raise RuntimeError("The obtained forces have wrong dimension.")

        return energy, forces

    def manage_settings(self, settings: dict, job, resources):
        import scine_utilities as utils
        program = settings['qm_module']
        method_family = 'dft' if program.lower() == 'orca' or program.lower() == 'turbomole' else settings['qm_model']
        available_qm_settings = utils.core.get_available_settings(method_family, program)
        if job.cores > 1 and 'external_program_nprocs' in available_qm_settings:
            settings['external_program_nprocs'] = job.cores
        if 'external_program_memory' in available_qm_settings:
            settings['external_program_memory'] = int(resources['memory'] * 1024)

    @job_configuration_wrapper
    def run(self, manager, calculation, config: Configuration) -> bool:
        import scine_database as db
        import scine_swoose as swoose
        import scine_utilities as utils

        # Gather all required collections
        structures = manager.get_collection('structures')
        calculations = manager.get_collection('calculations')
        properties = manager.get_collection('properties')

        # Link calculation if needed
        if not calculation.has_link():
            calculation.link(calculations)

        # Get structure
        structure = db.Structure(calculation.get_structures()[0])
        structure.link(structures)

        # Get model
        model = calculation.get_model()
        # Get job
        job = calculation.get_job()

        debug = bool(config['daemon']['mode'] == 'debug')
        if debug:
            print('Warning: The Swoose QM/MM job will result in a failed calculation in debug mode.\n')

        with calculation_context(self):
            xyz_file = 'system.xyz'
            settings = self.write_parameter_and_connectivity_file('Parameters.dat', 'Connectivity.dat',
                                                                  calculation.get_settings(), structure, properties)
            self.manage_settings(settings, job, config['resources'])
            utils.io.write(xyz_file, structure.get_atoms())
            swoose.calculate_qmmm(xyz_file, **settings)
            energy, forces = self.process_output(calculation, structure.get_atoms())

        # After the calculation, verify that the connection to the database still exists
        self.verify_connection()

        # Set the program
        model.program = 'swoose'

        # Store energy
        self.store_property(properties, 'electronic_energy', 'NumberProperty',
                            energy, model, calculation, structure)

        # Store forces
        self.store_property(properties, 'atomic_forces', 'DenseMatrixProperty',
                            forces, model, calculation, structure)

        calculation.set_executor(config['daemon']['uuid'])
        calculation.set_status(db.Status.COMPLETE)
        return True

    @staticmethod
    def required_programs():
        return ['database', 'utils', 'swoose']
