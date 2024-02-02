#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:27:03 2024

@author: jamesmiller
@packages: scipy, numpy, time
"""

import numpy as np
import time
from scipy.integrate import RK45

# Define the ODE function
def odeFunction(x, y):
    return -y + np.log(x)

# RK45 solver function
def solveOdeWithRK45(odeFunc, initialX, initialY, targetStep):
    startTime = time.time()
    solver = RK45(odeFunc, initialX, np.array([initialY]), t_bound=1e6)
    stepCount = 0
    while stepCount < targetStep and solver.status == 'running':
        solver.step()
        stepCount += 1
    endTime = time.time()
    return solver.t, solver.y[0], stepCount, endTime - startTime

# Euler's method is a simple, first-order numerical procedure for solving ODEs with a given initial value.
def eulerMethod(odeFunc, initialX, initialY, stepSize, numSteps):
    xValues, yValues = [initialX], [initialY]
    for _ in range(numSteps):
        yNext = yValues[-1] + stepSize * odeFunc(xValues[-1], yValues[-1])
        xNext = xValues[-1] + stepSize
        xValues.append(xNext)
        yValues.append(yNext)
    return xValues, yValues

# Initial conditions for the original problem
initialX, initialY = 2, 1

# Solve using RK45 for 1000 and 2000 steps
x1000RK45, y1000RK45, steps1000RK45, timeRK451000 = solveOdeWithRK45(odeFunction, initialX, initialY, 1000)
x2000RK45, y2000RK45, steps2000RK45, timeRK452000 = solveOdeWithRK45(odeFunction, initialX, initialY, 2000)

# Solve using Euler's method for 1000 and 2000 steps
stepSizeEuler = 0.3
numSteps1000Euler = int((x1000RK45 - initialX) / stepSizeEuler)
numSteps2000Euler = int((x2000RK45 - initialX) / stepSizeEuler)

startTimeEuler = time.time()
x1000Euler, y1000Euler = eulerMethod(odeFunction, initialX, initialY, stepSizeEuler, numSteps1000Euler)
timeEuler1000 = time.time() - startTimeEuler

startTimeEuler = time.time()
x2000Euler, y2000Euler = eulerMethod(odeFunction, initialX, initialY, stepSizeEuler, numSteps2000Euler)
timeEuler2000 = time.time() - startTimeEuler

# Print results
print(f"RK45 1000 Steps: x={x1000RK45}, y={y1000RK45}, Time={timeRK451000} seconds")
print(f"RK45 2000 Steps: x={x2000RK45}, y={y2000RK45}, Time={timeRK452000} seconds")
print(f"Euler 1000 Steps: x={x1000Euler[-1]}, y={y1000Euler[-1]}, Time={timeEuler1000} seconds")
print(f"Euler 2000 Steps: x={x2000Euler[-1]}, y={y2000Euler[-1]}, Time={timeEuler2000} seconds")
