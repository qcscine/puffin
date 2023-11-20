#!/bin/bash

# This code is licensed under the 3-clause BSD license.
# Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
# See LICENSE.txt for details.

# This script creates a new Python environment (python-3.7) and installs julia and
# the reaction mechanism simulator. Note that the script requires an existing conda
# installation and the path to this installation in the variable conda_home (vide infra).

# Assumes that conda exists at the following path
export conda_home=The-path-to-the-anaconda-installation
export rms_path=$PWD/ReactionMechanismSimulator
export rms_so_path=$rms_path/rms.so
export path_to_conda_env=$PWD/rms_conda_env

# Download RMS development version
git clone https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl.git $rms_path
cd $rms_path
# This script was tested with the following commit (Aug. 8. 2023, 19:32). The source code has likely changed since then and may include additional bug
# fixes and improvements.
git checkout 8fa7f60e2ef62050ba365503ddb95c16933752c3
cd ..
# export conda commands + create environment
. $conda_home/etc/profile.d/conda.sh
conda env create --file $rms_path/environment.yml --prefix $path_to_conda_env
conda activate $path_to_conda_env

# install julia and RMS
conda install -c rmg "julia>=1.8.5,!=1.9.0" "pyjulia>=0.6"

export JULIA_NUM_THREADS=8
export path_to_python=`which python`
echo 'import Pkg' > install_rms.jl
echo 'Pkg.add("PyCall")' >> install_rms.jl
echo 'ENV["CONDA_JL_HOME"] = "'$conda_home'"' >>  install_rms.jl
echo 'Pkg.build("Conda")' >>  install_rms.jl
echo 'ENV["PYTHON"] = "'$path_to_python'"' >> install_rms.jl
echo 'Pkg.build("PyCall")' >> install_rms.jl
echo 'Pkg.add("DiffEqBase")' >> install_rms.jl
echo 'Pkg.build("DiffEqBase")' >> install_rms.jl
echo 'Pkg.add("DifferentialEquations")' >> install_rms.jl
echo 'Pkg.build("DifferentialEquations")' >> install_rms.jl
echo 'Pkg.develop(Pkg.PackageSpec(path="'$rms_path'"))' >> install_rms.jl
echo 'Pkg.build("ReactionMechanismSimulator")' >> install_rms.jl
echo "Julia install script"
julia ./install_rms.jl

# create system image
python -m julia.sysimage $rms_so_path
# We need diffeqpy for the python bindings of SciML.
pip install diffeqpy==1.2.0

# Check installation
python -c 'from julia import Julia; jl = Julia(sysimage="'$rms_so_path'"); from diffeqpy import de;'
python -c 'from julia import Julia; jl = Julia(sysimage="'$rms_so_path'"); from julia import ReactionMechanismSimulator'

