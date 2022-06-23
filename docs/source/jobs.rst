Jobs
====

Basic Jobs
----------

All Jobs prepended with ``Scine`` indicate that they are making use of the C++
``module`` interface defined in SCINE. Hence, adding a personal ``module`` with
a custom ``Calculator`` will automatically enable the use of these tasks with
the custom ``module``/``Calculator``.

For more information  about this interface please see the documentation of
SCINE: Core and possibly also SCINE: Utils.

Conformer Generation
````````````````````
.. autoclass:: scine_puffin.jobs.conformers.Conformers

SCINE: AFIR
```````````
.. autoclass:: scine_puffin.jobs.scine_afir.ScineAfir

SCINE: Bond Orders
``````````````````
.. autoclass:: scine_puffin.jobs.scine_bond_orders.ScineBondOrders

SCINE: Geometry Optimization
````````````````````````````
.. autoclass:: scine_puffin.jobs.scine_geometry_optimization.ScineGeometryOptimization

SCINE: Hessian
``````````````````
.. autoclass:: scine_puffin.jobs.scine_hessian.ScineHessian

SCINE: IRC Scan
``````````````````
.. autoclass:: scine_puffin.jobs.scine_irc_scan.ScineIrcScan

SCINE: Artificial Force Induced Reaction Probe
``````````````````````````````````````````````
.. autoclass:: scine_puffin.jobs.scine_react_complex_afir.ScineReactComplexAfir

SCINE: Newton Trajectory Reaction Probe
```````````````````````````````````````
.. autoclass:: scine_puffin.jobs.scine_react_complex_nt.ScineReactComplexNt

SCINE: Newton Trajectory 2 Reaction Probe
`````````````````````````````````````````
.. autoclass:: scine_puffin.jobs.scine_react_complex_nt2.ScineReactComplexNt2

SCINE: Single Point
```````````````````
.. autoclass:: scine_puffin.jobs.scine_single_point.ScineSinglePoint

SCINE: Transition State Optimization
````````````````````````````````````
.. autoclass:: scine_puffin.jobs.scine_ts_optimization.ScineTsOptimization

Specialized Jobs
----------------

Gaussian: Partial Charges - Charge Model 5
``````````````````````````````````````````
.. autoclass:: scine_puffin.jobs.gaussian_charge_model_5.GaussianChargeModel5

Orca: Geometry Optimization
```````````````````````````
.. autoclass:: scine_puffin.jobs.orca_geometry_optimization.OrcaGeometryOptimization

Turbomole: Geometry Optimization
`````````````````````````````````
.. autoclass:: scine_puffin.jobs.turbomole_geometry_optimization.TurbomoleGeometryOptimization

Turbomole: Single Point
````````````````````````
.. autoclass:: scine_puffin.jobs.turbomole_single_point.TurbomoleSinglePoint

Turbomole: Hessian
```````````````````
.. autoclass:: scine_puffin.jobs.turbomole_hessian.TurbomoleHessian

Turbomole: Bond Orders
```````````````````````
.. autoclass:: scine_puffin.jobs.turbomole_bond_orders.TurbomoleBondOrders

Debugging: Sleep
````````````````
.. autoclass:: scine_puffin.jobs.sleep.Sleep
