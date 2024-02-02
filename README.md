# ODE Solver Comparison

This project implements and compares two numerical methods for solving Ordinary Differential Equations (ODEs): the Runge-Kutta method (specifically RK45) and Euler's method. The goal is to evaluate these methods in terms of accuracy and performance for a given ODE.

## Code Packages Used

- **NumPy**: Used for numerical operations, particularly for handling arrays and mathematical functions.
- **SciPy**: Specifically, the `RK45` solver from the `scipy.integrate` module is used for applying the Runge-Kutta method.
- **time**: This module is used to measure the performance in terms of computation time for both methods.

## Approach to Implementation

### ODE Definition

The ODE used for comparison is defined as:

```python
def odeFunction(x, y):
    return -y + np.log(x)
```

