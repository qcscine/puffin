.. image:: docs/source/res/puffin_header.png
   :alt: SCINE Puffin

.. inclusion-marker-do-not-remove

Introduction
------------

SCINE Puffin is a calculation handler for SCINE Chemoton. Thus, it bridges the gap between high level exploration jobs for reaction networks and  basic quantum chemical calculations. Making use of the abstractions defined the SCINE Core and SCINE Utilities  modules it provides the means to define and execute jobs that generate new data for reaction networks. SCINE Puffin is designed as an instanced runner that, once bootstrapped, continuously processes requested jobs. It is built to work in containerized environments (Docker, Apptainer/Singularity).

License and Copyright Information
---------------------------------

For license and copyright information, see the file ``LICENSE.txt`` in this
directory.

Installation
------------

Prerequisites
.............

Basic Requirements
``````````````````

Thus far Puffin is expected to run on Linux systems and possibly on OSX systems,
Windows-based architectures are not specifically supported.
Using containers and the integrated subsystem feature of Windows 10 it should
be possible to toy around with it nonetheless. This is, however, not a case
we would recommend for inexperienced users.

Puffin collects a variety of programs to run calculations. In order to get a
minimal working version of Puffin to run on a given system, the following
programs/packages are required:

- Python3
- CMake >= v3.12
- Git
- GCC >= v7.0.0, CLang >= v6.0.0 (Unix), ICC >= v2017.01
- Boost >= v1.64.0
- A MongoDB including the C++ bindings

Puffin Python Package
`````````````````````

The main Python package Puffin requires the several additional packages, which will be
automatically installed when installing Puffin using ``pip``. These packages are
listed in the file ``requirements.txt``.

For development work, a few extra packages are needed; these are listed in the file
``requirements-dev.txt``.

Puffin Instance
```````````````

A usable Puffin instance will require at least a minimal set of additional C++
components. These components do not need to be compiled by the user, but instead
they will be compiled when bootstrapping the instance. These components are:

- SCINE Database
- SCINE Molassembler
- SCINE ReaDuct
- SCINE Utilities

Installation
............

Puffin can be installed using pip (pip3) once the repository has been cloned:

.. code-block:: bash

   git clone <puffin-repo>
   pip install ./puffin

A non super user can install the package using a virtual environment, or
the ``--user`` flag.

It is also possible to run/install a dockerized version of Puffin. For more
details on this please see the ``Usage->Docker`` section of this documentation.

The documentation can be found online, or it can be built using:

.. code-block:: bash

   cd puffin
   make -C docs html

It is then available at:

.. code-block:: bash

   <browser name> docs/build/html/index.html

In order to build the documentation, you need a few extra Python packages wich
are not installed automatically together with Puffin. In order to install them,
run

.. code-block:: bash

   cd puffin
   pip install -r requirements-dev.txt

Usage
------

Basic Usage
...........

In order to use Puffin, at least one Puffin instance has to be generated.
Each instance will then separately accept and run jobs later on.
In order to setup a first instance, prepare a folder:

.. code-block:: bash

   mkdir puffin_instance
   cd puffin_instance

and generate the default configuration file for an instance inside it:

.. code-block:: bash

   python3 -m scine_puffin configure
   vi puffin.yaml

Afterwards it is possible to configure the instance to your liking, by editing
the generated ``puffin.yaml`` file.

It is also possible to enter configurations via environment variables.
The environment variables override the variables given in the configure file.
Any configuration option can be given via the environment as:

.. code-block:: bash

   export PUFFIN_<key1>_<key2>_<key3>=<value>

or as a concrete example:

.. code-block:: bash

   export PUFFIN_DATABASE_PORT=27019

The settings for each program are explained in the documentation of that
particular program. The general settings pertaining to the daemon and Puffin in
general are explained separately in a section below.

After (if desired) editing the settings it is possible to bootstrap Puffin and
thus install all programs that have to be installed in order to run the
instance.

.. code-block:: bash

   python3 -m scine_puffin -c puffin.yaml bootstrap

All programs that will be installed are marked in the configuration as
``available`` and have a ``source`` path given.
Bootstrapping the instance will generate a source file called ``puffin.sh``. It
contains all environment variables that have to be set in order to make the
instance find all installed programs.
Hence, source the installed programs using:

.. code-block:: bash

   source puffin.sh

The Puffin instance is now ready to be used. Start and stop the actual daemon by
using

.. code-block:: bash

   python3 -m scine_puffin -c puffin.yaml start

and

.. code-block:: bash

   python3 -m scine_puffin -c puffin.yaml stop

If the ``puffin.yaml`` is not given, the default options will be used. Once again,
any environment variables precede the loaded file.

Docker
......

It is possible to generate a containerized version of Puffin using Docker or
Podman. The ``Dockerfile`` is present in the directory ``container/docker`` of this repository.
The generated Docker image includes all packages required to run Puffin with the
default set of programs. When running the image it is possible to configure the
options of the Puffin instance. We recommend mounting a scratch directory e.g.
``/scratch/puffin`` into the Docker.

Furthermore, it is possible to dry-run (without a database) a version of the
image which will still execute the bootstrap stage of Puffin.
It is then possible to copy the clean ``/scratch/puffin`` folder and then
shorten (skip) the bootstrap step in other instances of the image by
pasting this copied folder into the ``/scratch`` folder mounted into the
other instances. However, for this to properly work we advise caution when
bootstrapping with ``-march=native`` flags for the programs in the initial
instance.

Command Line Setup
``````````````````

In order to build the image, execute

.. code-block:: bash

   sudo docker build -t <image name>

The image will bootstrap and start running a Puffin instance as soon as
the image is run with:

.. code-block:: bash

   docker run -it --mount src=<path to local folder>,target=/scratch,type=bind <image name>

Any configuration can be done by adding environment variables in the run command.

Docker Compose Setup
````````````````````

A minimal version of docker-compose input starting an instance of Puffin is
deposited in the ``docker`` folder located in the top directory of this
repository.

Basic Settings and Configuration
................................

The full list of all options to be set is given in the online documentation in
the ``Settings`` section. Furthermore, this documentation can be read directly
from the doc-string of the ``scine_puffin.config.Configuration`` class.
The settings of each program can be read from the documentation of the
respective class interfacing the program, e.g.
``scine_puffin.programs.sparrow.Sparrow``.

How to Cite
-----------

When publishing results obtained with Puffin, please cite the corresponding
release as archived on `Zenodo <https://doi.org/10.5281/zenodo.6695461>`_ (DOI
10.5281/zenodo.6695461; please use the DOI of the respective release).

In addition, we kindly request you to cite the following article when using Puffin:
J. P. Unsleber, S. A. Grimmel, M. Reiher,
"Chemoton 2.0: Autonomous Exploration of Chemical Reaction Networks",
*J. Chem. Theory Comput.*, **2022**, *18*, 5393.

Furthermore, when publishing results obtained with any SCINE module, please cite the following paper:

T. Weymuth, J. P. Unsleber, P. L. Türtscher, M. Steiner, J.-G. Sobez, C. H. Müller, M. Mörchen,
V. Klasovita, S. A. Grimmel, M. Eckhoff, K.-S. Csizi, F. Bosia, M. Bensberg, M. Reiher,
"SCINE—Software for chemical interaction networks", *J. Chem. Phys.*, **2024**, *160*, 222501
(DOI `10.1063/5.0206974 <https://doi.org/10.1063/5.0206974>`_).


Support and Contact
-------------------

In case you should encounter problems or bugs, please write a short message
to scine@phys.chem.ethz.ch.
