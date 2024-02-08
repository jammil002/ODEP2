from scipy.integrate import RK45
import matplotlib.pyplot as plt
from numpy import exp
import numpy as np

# Define the ODE function
def odeFunction(t, y):
    return y / (exp(t) - 1)

# Manually implemented RK4 solver
def solveOdeRK4(odeFunction, initialTime, initialSolution, endTime, stepSize):
    timeSteps = np.arange(initialTime, endTime, stepSize)
    solutionValues = np.zeros(len(timeSteps))
    solutionValues[0] = initialSolution
    for i in range(1, len(timeSteps)):
        t = timeSteps[i-1]
        y = solutionValues[i-1]
        k1 = odeFunction(t, y)
        k2 = odeFunction(t + stepSize / 2, y + stepSize / 2 * k1)
        k3 = odeFunction(t + stepSize / 2, y + stepSize / 2 * k2)
        k4 = odeFunction(t + stepSize, y + stepSize * k3)
        solutionValues[i] = y + (stepSize / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return timeSteps, solutionValues

# Solve the ODE using RK45 from SciPy
def solveOdeSciPy(odeFunction, initialTime, initialSolution, endTime, stepSize):
    solver = RK45(odeFunction, initialTime, [initialSolution], endTime, max_step=stepSize)
    timeSteps, solutionValues = [], []
    while solver.status == 'running':
        solver.step()
        timeSteps.append(solver.t)
        solutionValues.append(solver.y[0])
    return np.array(timeSteps), np.array(solutionValues)

# Initial conditions
initialTime = 1.0
initialSolution = 5.0
stepSize = 0.02
endTime = initialTime + 2000 * stepSize  # Ensuring at least 2000 steps

# Solve the ODE with both methods
timeStepsRK4, solutionValuesRK4 = solveOdeRK4(odeFunction, initialTime, initialSolution, endTime, stepSize)
timeStepsSciPy, solutionValuesSciPy = solveOdeSciPy(odeFunction, initialTime, initialSolution, endTime, stepSize)

# Plot the results for comparison
plt.figure(figsize=(10, 6))
plt.plot(timeStepsRK4, solutionValuesRK4, label='RK4 Manual', color='red', linestyle='--')
plt.plot(timeStepsSciPy, solutionValuesSciPy, label='RK45 SciPy', color='blue', linestyle='-')
plt.title('Comparison of ODE Solutions: RK4 Manual vs RK45 SciPy')
plt.xlabel('Time t')
plt.ylabel('Solution y')
plt.legend()
plt.grid(True)
plt.show()
