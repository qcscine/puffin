Changelog
=========

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


