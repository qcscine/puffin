# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from enum import Enum


class SubTaskToReaductCall(Enum):
    OPT = "run_opt_task"
    RCOPT = "run_opt_task"
    IRCOPT = "run_opt_task"
    IRC = "run_irc_task"
    TSOPT = "run_tsopt_task"
    NT = "run_nt_task"
    NT2 = "run_nt2_task"
    AFIR = "run_afir_task"
    SP = "run_sp_task"
