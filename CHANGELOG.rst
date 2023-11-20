Changelog
=========

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


