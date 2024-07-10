# Contactbench

C++ implementation of various contact models and solvers used for robotics simulation.

### The contact problem

The contact problem is formulated as a Non-linear Complementarity Problem (NCP):

$$
\begin{align}
&c = G \lambda + g \\
&\mathcal{K} \ni \lambda \perp c + {\Phi}(c) \in \mathcal{K} \\
\end{align}
$$

Several algorithms solving this problem or a relaxation of it are implemented and empirically evaluated.
We also benchmark various algorithms for gradient computation: automatic and implicit differentiation.

### Requirements
We use [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for free dynamics and Delassus computation. Collision detection and its gradients are performed with [hppfcl](https://github.com/humanoid-path-planner/hpp-fcl). The QP solver [ProxSuite](https://github.com/Simple-Robotics/proxsuite) is used in some contact models and to compute derivatives. Automatic differentiation is implemented via [CppAD](https://github.com/coin-or/CppAD).


### Citing this work
If you find this helpful work, please cite the related paper:
```
@article{lelidec2023contact,
  title={Contact Models in Robotics: a Comparative Analysis},
  author={Le Lidec, Quentin and Jallet, Wilson and Montaut, Louis and Laptev, Ivan and Schmid, Cordelia and Carpentier, Justin}
}
```

### Credits

The following people have been involved in the development of **contactbench**:

-   [Quentin Le Lidec](https://quentinll.github.io) (Inria): main developer and manager of the project
-   [Justin Carpentier](https://jcarpent.github.io) (Inria): manager of the project
-   [Wilson Jallet](https://github.com/ManifoldFR) (LAAS-CNRS/Inria): developer of the project
