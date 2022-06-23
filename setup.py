# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from os import path
from setuptools import setup, find_packages
import sys

min_version = (3, 6)
if sys.version_info < min_version:
    error = """
SCINE Puffin does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(
        *(sys.version_info[:2] + min_version)
    )
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt")) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines() if not line.startswith("#")]

with open(path.join(here, "README.rst"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open('scine_puffin/_version.py') as f:
    exec(f.read())

# Define the setup
setup(
    name="scine_puffin",
    version=__version__,
    author="ETH Zurich, Laboratory of Physical Chemistry, Reiher Group",
    author_email="scine@phys.chem.ethz.ch",
    description="Calculation handler for SCINE Chemoton",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://www.scine.ethz.ch",
    python_requires=">={}".format(".".join(str(n) for n in min_version)),
    packages=find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    package_data={
        "scine_puffin": [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    license="BSD (3-clause)",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: C++",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    entry_points={
        'console_scripts': [
            'scine_puffin = scine_puffin.__main__:main',
            'scine_puffin_bootstrap = scine_puffin.__main__:main_bootstrap',
            'scine_puffin_configure = scine_puffin.__main__:configure',
            'scine_puffin_container = scine_puffin.__main__:container',
            'scine_puffin_kill = scine_puffin.__main__:kill',
            'scine_puffin_start = scine_puffin.__main__:start',
            'scine_puffin_stop = scine_puffin.__main__:stop',
        ],
    },
    install_requires=requirements,
    zip_safe=False,
    test_suite="pytest",
    tests_require=["pytest"],
)
