# Computing Abs Linearizations using Jax

Given a function implmented via Jax, this module computes the piecewise lienarization as introduced by Griewank in https://doi.org/10.1080/10556788.2013.796683

This is done by performing the nessesary modifications on a symbolic representation of the function, which is provided by Jax. Given a function f and an argument x, you run

```python
ALF = AbsLinearForm(f, x)
```
This only performs symbolic computations, however, the argument is required to determine the shapes of the switching variables. Afterwards, you can evaluate the Abs-linearization using
```python
y, z, a, b, Z, L = ALF(x)
```
Here, y is the value of the function f at x and z are the corresponding switching variables.

In case x is not a Jax array but a tupel or a list, the code flattens and concatenates x before computing the Abs-linearization. To obtain the original structure of x, run
```python
jax.tree_util.tree_unflatten(ALF.treedef, JaxAbsLinearization.split_by_shapes(x_flat, ALF.xshapes))
```