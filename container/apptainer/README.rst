Basic Container Usage
---------------------

In essence, these two commands should be enough to run a basic ``Puffin``
instance inside an Apptainer (formerly Singularity) image.
One to build the container:

.. code-block:: bash

   apptainer build puffin.sif puffin.def

and one to run the generated image:

.. code-block:: bash

   apptainer run puffin.sif

The build command may require slight alterations to the ``.def`` file first, as
the file may reference a user's SSH keys in order to allow the cloning of some
``SCINE`` repositories.
For some users the ``--fakeroot`` option may be more comfortable than using a
``sudo`` preface, but keep in mind that this option is not guaranteed to work,
depending on the subordinate UID mapping configured for you on the machine you're working on.
Also, it may be required to make Apptainer build the image in
a custom temporary directory if the default ``/tmp`` is not big enough.
A slightly different build command may thus look like this:

.. code-block:: bash

   APPTAINER_TMPDIR=/scratch/tmp apptainer build --fakeroot puffin.sif puffin.def

The run command will have to be changed depending on the use case.
It may be required to change some of the settings of the Puffin
instance running inside the container. To this end, environment variables for
Puffin can be set. These variables need to be present inside the container.
It is possible to use the ``--env`` or ``--env-file`` argument to the run command.
However, the more general way is setting ``APPTAINERENV_`` variables on the host
machine.

Furthermore it is required to mount two folders into the image, both are scratch
direcories, one for ``.log`` and ``.pid`` files (``/socket``) and one for the
actual job scratch (``/jobs``).

A more complete run could thus look like this:

.. code-block:: bash

   apptainer build puffin.sif puffin.def
   export APPTAINERENV_PUFFIN_DATABASE_NAME=ath_dft
   export APPTAINERENV_PUFFIN_DATABASE_PORT=27001
   export APPTAINERENV_PUFFIN_DATABASE_IP=129.132.118.83
   apptainer run --bind /scratch/puffin:/socket \
                   --bind /scratch/puffin/jobs:/jobs \
                   puffin.sif

