# Contactbench

C++ implementation of various contact models and solvers used for robotics simulation.


### The contact problem

The contact problem is formulated as a Non-linear Complementarity Problem (NCP):
$$c = G \lambda + g \\
\mathcal{K}_\mu \ni \lambda \perp c + \Gamma_\mu(c) \in \mathcal{K}^*_\mu$$

Several algorithms solving this problem or a relaxation are implemented and empirically evaluated.
We also benchmark various algorithms for gradients computation: automatic differentiation and implicit differentiation.


### Requirements
We use [pinocchio3x](https://github.com/stack-of-tasks/pinocchio) for free dynamics and Delassus computation. Collision detection and its gradients are performed with [hppfcl](https://github.com/humanoid-path-planner/hpp-fcl). The QP solver [ProxSuite](https://github.com/Simple-Robotics/proxsuite) is used in some contact models and to compute derivatives. Automatic differentiation is implemented via [CppAD](https://github.com/coin-or/CppAD).


### Citing this work
If you find this work useful, please cite the related paper:
```
@article{lelidec2023contact,
  title={Contact Models in Robotics: a Comparative Analysis},
  author={Le Lidec, Quentin and Jallet, Wilson and Montaut, Louis and Laptev, Ivan and Schmid, Cordelia and Carpentier, Justin}
}
```
