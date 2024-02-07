import numpy as np
from scipy.integrate import RK45
import time
import matplotlib.pyplot as plt

# ODE Function
def odeFunction(t, y):
    return y / (np.exp(t) - 1)

# Euler's Method
def eulerMethod(f, x0, y0, h, steps):
    x, y = x0, y0
    for _ in range(steps):
        y += h * f(x, y)
        x += h
    return x, y

# Generic ODE Solver with Timing
def solveOde(method, odeFunc, initialT, initialY, tEnd, stepSize, stepsEuler=None):
    startTime = time.time()
    if method == "RK45":
        solver = RK45(odeFunc, initialT, np.array([initialY]), tEnd, max_step=stepSize)
        steps = 0
        while solver.status == 'running':
            solver.step()
            steps += 1
            if steps >= 2000:  # Prevent excessive computation
                break
        tFinal, yFinal = solver.t, solver.y[0]
    else:  # Euler
        steps = stepsEuler or 2000
        tFinal, yFinal = eulerMethod(odeFunc, initialT, initialY, stepSize, steps)
    endTime = time.time()
    elapsedTime = endTime - startTime
    return tFinal, yFinal, steps, elapsedTime

# Error Analysis
def calculateAbsoluteError(yTrue, yApprox):
    return abs(yTrue - yApprox)

def calculateRelativeError(yTrue, yApprox):
    return abs((yTrue - yApprox) / yTrue) if yTrue != 0 else float('inf')

def performErrorAnalysis(odeFunc, initialT, initialY, tEnd, stepSizes):
    errors = []
    for stepSize in stepSizes:
        # RK45 Solution
        xRk45, yRk45, stepsRk45, timeRk45 = solveOde("RK45", odeFunc, initialT, initialY, tEnd, stepSize)
        # Euler Solution
        stepsEuler = int((tEnd - initialT) / stepSize)
        xEuler, yEuler, stepsEulerActual, timeEuler = solveOde("Euler", odeFunc, initialT, initialY, tEnd, stepSize, stepsEuler)
        
        # Errors
        absError = calculateAbsoluteError(yRk45, yEuler)
        relError = calculateRelativeError(yRk45, yEuler)
        errors.append((stepSize, absError, relError, timeRk45, timeEuler))

        print(f"Step Size: {stepSize}, Absolute Error: {absError}, Relative Error: {relError}, RK45 Time: {timeRk45:.5f}s, Euler Time: {timeEuler:.5f}s")

    return errors

# Plotting Function
def plotErrors(errors):
    stepSizes, absErrors, relErrors, _, _ = zip(*errors)
    plt.figure(figsize=(14, 6))

    # Absolute Error Plot
    plt.subplot(1, 2, 1)
    plt.plot(stepSizes, absErrors, 'o-', label='Absolute Error')
    plt.xlabel('Step Size')
    plt.ylabel('Error')
    plt.title('Absolute Error vs Step Size')
    plt.grid(True)
    plt.legend()

    # Relative Error Plot
    plt.subplot(1, 2, 2)
    plt.plot(stepSizes, relErrors, 'o-', color='red', label='Relative Error')
    plt.xlabel('Step Size')
    plt.ylabel('Error')
    plt.title('Relative Error vs Step Size')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == "__main__":
    stepSizes = [0.02, 0.1, 0.5]  # Define step sizes for testing
    errors = performErrorAnalysis(odeFunction, 1, 5, 10, stepSizes)
    plotErrors(errors)
