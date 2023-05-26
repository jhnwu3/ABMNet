import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import roadrunner

time_points = 500
num_parameters = 1000
num_init_cond = 5000

rr = roadrunner.RoadRunner("3pro_sbml.xml")
initial_conditions_arr = np.loadtxt("3linX0.csv", delimiter=",", dtype=float)
rr.k1 = 0.276782
rr.k2 = 0.837081
rr.k3 = 0.443217
rr.k4 = 0.0424412
rr.k5 = 0.304645

# EXAMPLE CODE:
# x_mat = np.zeros((5000, 3))
# # for x in range(num_init_cond):
# rr.model.setFloatingSpeciesInitConcentrations(initial_conditions_arr[0])
# rr.setIntegrator("gillespie")
# result = rr.simulate(0, 100, 100)
# print(result)

times = np.linspace(0.0, 5.0, num=time_points+1)

mean_arr = np.zeros((time_points, 3))
var_arr = np.zeros((time_points, 3))
cov_arr = np.zeros((time_points, 3))

mean_arr[0] = np.mean(initial_conditions_arr, axis=0)
var_arr[0] = np.var(initial_conditions_arr, axis=0)
c = np.cov(initial_conditions_arr, rowvar=False)
n = 0
for i in range(c.shape[0] - 1):
    for j in range(1, c.shape[1]):
        if(i != j):
            cov_arr[0][n] = c[i,j]
            n += 1

for t in range(1, time_points):
    x_mat = np.zeros((5000, 3))
    for x in range(num_init_cond):
        rr.model.setFloatingSpeciesInitConcentrations(initial_conditions_arr[x])
        rr.setIntegrator("gillespie")
        result = rr.simulate(0, times[t], 2)
        x_mat[x][0] = result[1][1]
        x_mat[x][1] = result[1][2]
        x_mat[x][2] = result[1][3]
    mean_arr[t] = np.mean(x_mat, axis=0)
    var_arr[t] = np.var(x_mat, axis=0)
    c = np.cov(x_mat, rowvar=False)
    n = 0
    for i in range(c.shape[0] - 1):
        for j in range(1, c.shape[1]):
            if(i != j):
                cov_arr[t][n] = c[i,j]
                n += 1

times = times[:time_points].reshape((time_points, 1))

total_mat = np.hstack((mean_arr, var_arr, cov_arr, times))
np.savetxt("3linyt2.csv", total_mat, delimiter=",")