#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import pytest

from ..utilities.imports import MissingDependency, MissingDependencyError, requires


@requires("arbitrary_dependency")
def _failing(_some_argument, _another_argument):
    pass


@requires("pytest")
def _working(_some_argument, _another_argument):
    pass


def test_requires():
    _working(1, 2)
    with pytest.raises(MissingDependencyError):
        _failing(1, 2)


def test_missing_dependency():
    missing_dependency = MissingDependency("arbitrary_dependency")
    with pytest.raises(NameError):
        _ = missing_dependency.arbitrary_attribute_name
    with pytest.raises(MissingDependencyError):
        _ = missing_dependency()
    with pytest.raises(NameError):
        _ = missing_dependency.some_method(42)
