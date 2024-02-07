# ODE Solver Comparison

This project implements and compares two numerical methods for solving Ordinary Differential Equations (ODEs): the Runge-Kutta method (specifically RK45) and Euler's method. The goal is to evaluate these methods in terms of accuracy and performance for a given ODE.

## Code Packages Used

- **NumPy**: Used for numerical operations, particularly for handling arrays and mathematical functions.
- **SciPy**: Specifically, the `RK45` solver from the `scipy.integrate` module is used for applying the Runge-Kutta method.
- **time**: This module is used to measure the performance in terms of computation time for both methods.


## How to Run
1. Ensure Python is installed on your system.
2. Install the required packages: numpy and scipy.
3. Run the provided Python script to observe the output and performance of each method.

## Approach to Implementation

### ODE Definition

The ODE used for comparison is defined as:

```python
def odeFunction(x, y):
    return -y + np.log(x)
```

The differential equation is given by:

$$
f(x, y) = -y + \ln(x)
$$

### Defining the Differential Equation

I started by defining the differential equation as a Python function, which takes \(x\) and \(y\) as inputs and computes the derivative \(f(x, y)\).

### Choosing Numerical Methods

- **RK45 Method**: I chose the RK45 method for its adaptive step size feature, offering a balance between accuracy and efficiency, which is crucial for achieving precise solutions.
- **Euler's Method**: As a contrast, I also implemented Euler's method, known for its simplicity and speed, though it typically offers less accuracy than RK45.

### Implementing the RK45 Method

Utilizing the `scipy.integrate.RK45` class, I implemented this method by specifying the differential equation, initial conditions, step size, and integration bounds.

### Implementing Manually using Euler's Method

I manually coded Euler's method, iteratively updating the solution based on the equation \(y_{n+1} = y_n + h \cdot f(x_n, y_n)\), where \(h\) is the predefined step size.

### Measuring and Comparing Performance

To evaluate both methods, I measured their performance in terms of computational time and solution accuracy. This involved running each method under identical initial conditions and comparing their outputs and execution times.

### Analyzing Results

My analysis focused on the trade-offs between these methods, considering the accuracy of their solutions, the computational effort involved, and the ease of implementation.


