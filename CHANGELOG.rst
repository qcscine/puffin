Changelog
=========

Release 2.1.0
-------------

New Jobs:
    - Job that tries to find a new reaction with active involvement of solvent molecules with the help of QM/MM.
    - Hamiltonian generation for Huzinaga-type DFT embedding (including QM/MM support).

New Features:
    - DFTB+ is available as a program
    - The `scine_dissociation_cut` has a new setting (`additional_nt_run_dissociation_energy_limit`) that lets one set a dissociation energy limit below which every dissociation that cannot be confirmed to be a barrierless reaction is additionally searched for a transition state by an associative NT2 reaction search protocol started from a reactive complex assembled from the dissociated fragments.
    - Allow different settings for the IRC sanity check in our `scine_react` jobs.
    - Add a Swoose helper to combine SFAM parameters and determine sphere-based QM regions for non-covalently bound systems.
    - The `rms_kinetic_modeling` job now accepts a list of booleans `reversible_reactions` in its settings to signal which reactions are reversible. If not given, all reactions are assumed to be reversible.
    - Parameters are not analyzed for their sensitivity indices during kinetic modeling if their uncertainty is zero.
    - Allows jobs to require additional programs based on the given settings.
    - Atomic charges and atom types can be forwarded to Swoose for QM/MM calculations.
    - Partial energies and partial gradients are now stored if available after running a `scine_single_point` job.
    - The energy criterion for saving weakly interacting complexes in `scine_react_complex` jobs is now a setting.

Technical changes:
    - Removed versioneer

Release 2.0.0
-------------

New Jobs:
    - Automatic QM region selection for QM/MM.
    - Job that tries to find a new reaction starting from a transition state guess only.
    - Job that carries out a fast dissociation reaction trial protocol after optimizing the input structure.
    - Remove `scine_step_refinement` job.

New Settings:
    - Add the option `allow_exhaustive_product_decomposition` to jobs that carry out reaction trials, which allows products to decompose and re-optimize them until no further decomposition is observed.
    - Add the option `always_add_barrierless_step_for_reactive_complex` to jobs that carry out reaction trials, which enables that a barrierless elementary step is added for the reactive complex formation of a bimolecular reaction irrespective of the complexation energy.
    - Add option `store_structures_with_frequency` and `store_structures_with_fraction` that allow storing a portion of structures per sub-task.
    - Add option `spin_propensity_ts_check` to `scine_react_ts_guess` job to specify a spin multiplicity range to be checked for the transition state different to the spin multiplicity range to be checked for the reactants and products (`spin_propensity_check`)

Technical changes:
    - Ensure that the calculated spin states are considered in the calculation of the dissociation energy in the `DissociationCut` jobs and in structure optimizing jobs (`GeometryOptimization` and `ReactTsGuess`)
    - Write normalized modes in database.
    - Improve dependency handling and add more typehints.
    - Optimize only unique structures of the endpoints of an IRC calculation.
    - Deduplicate code for analyzing both sides of the IRC part in the `scine_react_job`.
    - Stricter conditions to distribute charges if `expect_charge_separation` is set to `True` by prohibiting changing
      already changed charge.
    - Comply docstrings with numpy styling.
    - Introduce Enums for ReaDuct calls.

Release 1.3.0
-------------

New Features:
    - Store found elementary step even if none of the endpoints corresponds to the initial starting structures
    - Add restart information with valid TS for jobs trying to find new elementary steps, where the IRC failed to produce different endpoints
    - Consider potential surface structures for label determination of new structures
    - Logic to transfer indices information and other complex properties from reactants to products
    - Save close lying spin multiplicities and allow to manipulate exact spin propensity
      check behavior with added settings

New Jobs:
    - Microkinetic modeling with the program Reaction Mechanism Simulator.

New interfaced programs
    - AMS via SCINE AMS Wrapper
    - MRCC (release version March 2022)

Bug fixes:
    - Ensure that `only_distance_connectivity` is adhered in all reaction jobs

Other:
    - Update address in license

Release 1.2.0
-------------

New Features:
    - Add a mechanism to stop multiple Puffins
    - Generate PID based on UUID, allowing to run multiple Puffins on the same filesystem

New Jobs:
    - Double ended reaction step refinement.

Further changes:
    - Various bugfixes and improvements

Release 1.1.0
-------------

New Features:
 - Support for stable intermediate complexes and barrier-less reactions
    - Strongly interacting complexes containing multiple structures
      are now saved in the database.
    - Spontaneous barrier-less associations detected during reaction probing
      are now considered barrier-less reactions.
    - Uphill barrier-less dissociations may be probed.
    - All structures visited during reaction probing may be saved in the
      database if required.

New Jobs:
 - Open source (SCINE-based) jobs
    - Elementary step refinement starting from a previously optimized transition state
    - Conceptual DFT property calculation
    - QM/MM force calculation
    - Barrier-less dissociation probing

 - New interfaced programs
    - SCINE Swoose


Release 1.0.0
-------------

Initial Features:
 - Runs as a daemon
    - With possible graceful timeout/shutdown after a user-defined time
    - With automatic cleaning of failed jobs
    - With a tolerance for database disconnects at the end of jobs
 - Provides a containerized version
    - Usable with Docker, Podman, and Singularity
    - Includes/installs all open source programs

Initial Jobs:
 - Open source (SCINE-based) jobs
    - Conformer generation
    - Artificial force induced reactions (AFIR) optimization
    - Bond order generation
    - Geometry optimization
    - Hessian generation incl. thermo chemistry
    - IRC scan
    - Reactive complex reaction probing (using AFIR, NT1, NT2)
    - Single point calculations
    - Transition state optimization

 - Specialized jobs:
    - Gaussian: partial charges - charge model 5 (CM5)
    - Orca: geometry optimization
    - Turbomole: geometry optimization
    - Turbomole: single point
    - Turbomole: Hessian 
    - RDKit: conformer generation

Initially interfaced programs used in calculations:
 - SCINE Molassembler
 - SCINE Readuct
 - SCINE Sparrow
 - Serenity (v1.4, via SCINE Serenity Wrapper)
 - XTB (v6.4.1, via SCINE XTB Wrapper)
 - Orca (v4.1.X, v4.2.X)
 - Turbomole (v7.x.x)
 - Gaussian (g09 Rev. D.01)


