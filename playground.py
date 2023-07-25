

from modules.data.temporal import *
from modules.data.mixed import *
from modules.utils.pso import *
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import numpy as np
import torch 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# file1 = "data/time_series/indrani_zeta_ca_no_zeroes_2500.pickle"
# file2 = "data/time_series/indrani_zeta_ca_no_zeroes.pickle"
# path = "data/time_series/indrani_zeta_ca_no_zeroes_3500.pickle"
# combined = combine_temporal_pickles(file1, file2, save=True, path=path)
# print(len(combined["outputs"]))
# print(len(combined["rates"]))

# seq_data = TemporalDataset(path,min_max_scale=False, standardize_inputs=False)
# # find 4 time points
# time_points = [10,250,750,1750]
# # process these into a new dataset, and then train
# for t in time_points:
#     data = generate_static_dataset(seq_data,t)
#     data.write_to_csv("data/static/indrani/indrani_zeta_ca_t" +str(t) + ".csv")
    
    



def indrani_costss(x, surrogate, y, dataset, standardize=True, normalize=True):
    # costs = []
    if len(x.shape) < 2:
        thetaCopy = x
    else:
        thetaCopy = x.reshape(x.shape[0], -1)
    # print("ThetaCopy:", thetaCopy.shape)
    # print(thetaCopy)
    if standardize:
        thetaCopy = (thetaCopy - dataset.input_means) / dataset.input_stds
    surrogate.eval()
    input = tc.from_numpy(thetaCopy)
    if next(surrogate.parameters()).is_cuda:
        input = input.to(tc.device("cuda"))

    # if batch:
    #     input = input.unsqueeze(dim=0)
    # print(" IM ALIVE")
    output = None
    with tc.no_grad():
        output = surrogate(input).squeeze().cpu().numpy()

    if normalize:
        scale_factor = dataset.output_maxes - dataset.output_mins
        output = (output * scale_factor) + dataset.output_mins

    cost = (output[:, 1] - y) * (output[:, 1] - y)
    
   
    # print(cost.shape)
    return cost

def evaluate_MLP_surrogate(x, surrogate, dataset, standardize=True, normalize=True, batch=False):
    if len(x.shape) < 2:
        thetaCopy = x
    else:
        thetaCopy = x.reshape(x.shape[0], -1)
    # print("ThetaCopy:", thetaCopy.shape)
    # print(thetaCopy)
    if standardize:
        thetaCopy = (thetaCopy - dataset.input_means) / dataset.input_stds
    surrogate.eval()
    input = tc.from_numpy(thetaCopy)
    if next(surrogate.parameters()).is_cuda:
        input = input.to(tc.device("cuda"))

    if batch:
        input = input.unsqueeze(dim=0)
    # print(" IM ALIVE")
    output = None
    with tc.no_grad():
        output = surrogate(input).squeeze().cpu().numpy()

    if normalize:
        scale_factor = dataset.output_maxes - dataset.output_mins
        output = (output * scale_factor) + dataset.output_mins
    return output


def plot_confidence_intervals(x, confidence=0.95, title=None, labels=None):
    n_features = x.shape[1]
    n_samples = x.shape[0]
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z_value = np.abs(np.random.standard_normal(x.shape)) * std
    intervals = np.percentile(x, [(1-confidence)*50, (1+confidence)*50], axis=0)

    plt.figure(figsize=(10, 6))
    for i in range(n_features):
        plt.plot([i, i], intervals[:, i], 'r-', lw=2)
        plt.plot(i, mean[i], 'bo', markersize=6)
    plt.xticks(range(n_features), labels)
    plt.xlabel('Surrogates')
    plt.ylabel('Ca')
    plt.title(title)
    plt.grid(True)
    plt.savefig(title + ".png")

def plot_confidence_intervals_log(x, confidence=0.95, title=None, labels=None):
    n_features = x.shape[1]
    n_samples = x.shape[0]
    
    # Convert features to log scale
    x_log = np.log(x)
    
    mean = np.mean(x_log, axis=0)
    std = np.std(x_log, axis=0)
    z_value = np.abs(np.random.standard_normal(x_log.shape)) * std
    intervals = np.percentile(x_log, [(1-confidence)*50, (1+confidence)*50], axis=0)

    plt.figure(figsize=(10, 6))
    for i in range(n_features):
        plt.plot([i, i], intervals[:, i], 'r-', lw=2)
        plt.plot(i, mean[i], 'bo', markersize=6)
    plt.xticks(range(n_features), labels)
    plt.xlabel('Features')
    plt.ylabel('Values (Log Scale)')
    plt.title(title)
    plt.grid(True)
    plt.savefig(title + ".png")


def find_closest_index(arr, value):
    closest_index = np.abs(arr - value).argmin()
    return closest_index

# observed data is in the format of  (t,y)
def get_corresponding_y(observed_data, chosen_times):
    y = []
    for t in chosen_times:
        tdx = find_closest_index(observed_data[:,0], t)
        y.append(observed_data[tdx, 1])
    return y 

def multi_indrani_cost_fxn(x, surrogates, ys, datasets):
    # Reshape the input to match the expected shape
     
    mse = 0
    for i in range(len(surrogates)):
        surrogate = surrogates[i]
        dataset = datasets[i]
        y = ys[i]
        mse += indrani_costss(x, surrogate, y, dataset=dataset, batch=False)
    # print(mse.shape)
    return mse

def evaluate_pso_estimate(estimate, surrogate, y, dataset, standardize=True, normalize=True):
    # get surrogate predictions

    if standardize:
        thetaCopy = (estimate - dataset.input_means) / dataset.input_stds

    input = tc.from_numpy(thetaCopy)
    if next(surrogate.parameters()).is_cuda:
        input = input.to(tc.device("cuda"))

    # if batch:
    input = input.unsqueeze(dim=0)
    # print(" IM ALIVE")
    output = None
    with tc.no_grad():
        output = surrogate(input).squeeze().cpu().numpy()

    if normalize:
        scale_factor = dataset.output_maxes - dataset.output_mins
        output = (output * scale_factor) + dataset.output_mins
    

    print("Evaluating PSO Estimate:", estimate )
    print("Surrogate Prediction:", output[1])
    print("Observed Y:", y)
    return output
    
       
    
    
# Get the actual experimental data vectors
# full_seq_dataset = TemporalDataset("data/time_series/indrani_zeta_ca_no_zeroes.pickle", min_max_scale=True)
full_seq_dataset = TemporalDataset("data/time_series/indrani_zeta_ca_h_no_zeroes_3704.pickle", standardize_inputs=False, min_max_scale=False)
# triangle_data = generate_static_with_temporal_features_dataset(full_seq_dataset)
# triangle_data.write_to_csv("data/static/indrani_triangle_features.csv")

data_46L_50F_53V_85k = np.loadtxt("data/John_Indrani_data/zeta_Ca_signal/test_data_experiments/46L_50F_53V_85k.dat")
data_46L_50F_100k = np.loadtxt("data/John_Indrani_data/zeta_Ca_signal/test_data_experiments/46L_50F_100k.dat")
data_46L_53V_80k = np.loadtxt("data/John_Indrani_data/zeta_Ca_signal/test_data_experiments/46L_53V_80k.dat")
data_CD3z_46L_50k = np.loadtxt("data/John_Indrani_data/zeta_Ca_signal/test_data_experiments/CD3z_46L_50k.dat")
data_CD3z_mouse = np.loadtxt("data/John_Indrani_data/zeta_Ca_signal/test_data_experiments/CD3z_mouse_WT_33k.dat")
data_CD3z_human = np.loadtxt("data/John_Indrani_data/zeta_Ca_signal/test_data_experiments/human_cd3z_mean.dat")
t_max = 277
Y = [data_46L_50F_53V_85k[:t_max], data_46L_50F_100k[:t_max], data_46L_53V_80k[:t_max], data_CD3z_46L_50k[:t_max], data_CD3z_mouse[:t_max], data_CD3z_human[:t_max]] # big variable with all of the data in list format 
labels = ["46L_50F_53V", "46L_50F", "46L_53V", "CD3z_46L", "CD3z_mouse", "CD3z_human"] # for titles so we can plot them later with the predicted rate constants and outputs.
Y_dict = {}
for i in range(len(labels)):
    Y_dict[labels[i]] = Y[i]


# get surrogates
time_pts = [10, 250, 750, 1750]
# time_pts = [250]
# get their corresponding list of actual time-values
actual_times = full_seq_dataset.times[time_pts]
print("--- Times: -----")
print(actual_times)
surrogates = []
prefix = "model/ixr3k_zeta_ca_h_t"
suffix = "_res_batch.pt"

dataset_prefix = "data/static/indrani/indrani_zeta_ca_h_t"
dataset_suffix = ".csv"
datasets = []
for t in time_pts: 
    surrogates.append(tc.load(prefix + str(t) + suffix))
    datasets.append(ABMDataset(dataset_prefix + str(t) + dataset_suffix, root_dir="data/",standardize=True, norm_out=True))
# perform PSO across multiple time points (i.e each datasets y's) and surrogates.


# Define the bounds for a 5-dimensional problem
ixr_data = ABMDataset("data/static/indrani/indrani_zeta_ca_t750.csv", root_dir="data/", standardize=True, norm_out=True)
lower_bound = np.zeros(5)  # Lower bound array with all zeros
upper_bound = ixr_data.input_means + 6*ixr_data.input_stds  # Upper bound array with all 10,000s
bounds = (lower_bound, upper_bound)
# print(bounds)
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
num_particles = 2500  # Number of particles in the swarm
n_iters = 600
dim = 5  # Dimensionality of the problem

n_runs = 5

# generate contour plots across the 5 different parameters.
# kon=random.uniform(1e-7,1e-2)
# koff=random.uniform(1,10)

varyingX = 2
varyingY = 3
kon = 0.000026
koff = 9.23215
C1 = 4499.4346#random.uniform(5e3,1e4)
C2 = 1.28873#random.uniform(1,10)
# C1 = random.uniform(5e3,1e4)
# C2 = random.uniform(1,10)
g= 0.0032684#random.uniform(1e-4,1e-2)
# between k1 and k2 
N = 100000
surr_input = np.zeros((N,5))
labels = ["k1", "k2", "C1", "C2", "g"]
indraniTruth = np.array([kon, koff, C1, C2, g])
# for i in range(N):
#     # surr_input[i] = np.array([random.uniform(1e-7,1e-2),random.uniform(1,10), C1, C2, g])
#     # surr_input[i] = np.array([kon,koff, random.uniform(5e3,1e4), random.uniform(1,10), g])
#     # surr_input[i] = indraniTruth
#     randomVector = np.array([random.uniform(1e-7,1e-2), random.uniform(1,10), random.uniform(5e3,1e4), random.uniform(1,10), random.uniform(1e-4,1e-2)])
#     # surr_input[i,varyingX] = randomVector[varyingX]
#     # surr_input[i, varyingY] = randomVector[varyingY]
#     surr_input[i] = randomVector
    

ys = get_corresponding_y(Y_dict["CD3z_46L"], actual_times)
# for i in range(len(surrogates)):
#     evaluations = indrani_costss(surr_input, surrogates[i], ys[i], datasets[i], batch=False)  
#     print(evaluations.shape)
#     print(evaluations.max())
#     print(evaluations.min())
#     plt.figure()
    
#     # norm = mcolors.Normalize(vmin=np.min(evaluations), vmax=np.max(evaluations))
#     heat = np.log(evaluations)
#     plt.title("Surrogate for t="+ str(actual_times[i]))
#     plt.xlabel(labels[varyingX])
#     plt.ylabel(labels[varyingY])
#     plt.scatter(surr_input[:,varyingX], surr_input[:,varyingY] , c=heat,s=4)
#     plt.scatter(indraniTruth[varyingX], indraniTruth[varyingY], c="r", s=50, label="Indrani's Estimate")
#     colorbar = plt.colorbar()
#     colorbar.set_ticks([0, 0.5 ,1.0])  # Tick positions
#     colorbar.set_ticklabels([str(int(heat.min())),"log(square error)", str(int(heat.max()))])  # Tick labels
#     plt.legend()
#     plt.savefig("Surrogate"+str(i)+ "_Contour_Full_random.png")
#     plt.close()

# evaluations = evaluate_MLP_surrogate(surr_input, surrogates[1], datasets[1])
# print("EVALUATIONS", evaluations.shape)
# fig, axs = plt.subplots(5,5, figsize=(75,25))
# for j in range(5):
#     for i in range(5):
#         axs[j, i].scatter(surr_input[:,j], surr_input[:,i], c=evaluations[:,1], s=1)
#         fig.colorbar(axs[j, i].collections[0], ax=axs[j, i])
# plt.savefig("surr_eval_full_t250.png")


minEstimates = []
minCosts = []
quad_costs = []
varyingX = 0
varyingY = 1
soi = tc.load("model/ixr_biased_c20_ib_t750.pt")
for e in range(1):
    for i in range(N):
        randomVector = np.array([random.uniform(1e-7,1e-2), random.uniform(1,10), random.uniform(5e3,1e4), random.uniform(1,10), random.uniform(1e-4,1e-2)])
        surr_input[i] = randomVector
        
    quad_costs = indrani_costss(surr_input, surrogate=soi, y=ys[2], dataset=datasets[2])#multi_indrani_cost_fxn(surr_input, surrogates=surrogates, ys=ys, datasets=datasets)
    heat = np.log(quad_costs)
    
    minCosts.append(quad_costs.min())
    minEstimates.append(surr_input[np.argmin(quad_costs)])
    plt.title("Cost Space Across Time")
    plt.xlabel(labels[varyingX])
    plt.ylabel(labels[varyingY])
    plt.scatter(surr_input[:,varyingX], surr_input[:,varyingY] , c=heat, s=4)
    colorbar = plt.colorbar()
    colorbar.set_ticks([heat.min(), heat.max() / 2 ,heat.max()])  # Tick positions
    colorbar.set_ticklabels([str(int(heat.min())),"log(square error)", str(int(heat.max()))])  # Tick labels
    plt.legend()
    plt.savefig("Surrogates_cost_contour.png")
    plt.close()
    print("Found New Min Cost:")
    print(minCosts[e])
    print("Estimate:")
    print(minEstimates[e])

minCosts = np.array(minCosts)
minEstimates = np.array(minEstimates)

print(minEstimates.shape)
print("Final Min Cost:")
print(minCosts.min())
print("Final Min Estimate:")
print(minEstimates[np.argmin(minCosts)])


# 3.50749103e-04 3.81711042e+00 6.22638543e+03 2.45842800e+00
# 1.74688920e-03


# print("with cost:",quad_costs[np.argmin(quad_costs)])
# colorbar.set_ticks([0, 0.5 ,1.0])  # Tick positions
# colorbar.set_ticklabels([str(int(heat.min())),"log(square error)", str(int(heat.max()))])  # Tick labels
# plt.legend()
# heat = np.log(quad_costs)
# plt.scatter(surr_input[:,varyingX], surr_input[:,varyingY] , c=heat,s=1)
# colorbar = plt.colorbar()
# plt.savefig("ixr_Surrogate_MultiCost.png")

# load in and rescale indrani's estimates
# indrani_estimates = np.loadtxt("CD3z_46L_indrani_estimates.dat")
# print(indrani_estimates.shape)
# indrani_estimates = np.power(10, indrani_estimates[:15,:-1])
# print(indrani_estimates.shape)
# print(indrani_estimates.max())
# print(indrani_estimates.min())
# predictions = [] # get all the predictions and plot them across time
# for i in range(len(surrogates)):
#     predictions.append(evaluate_MLP_surrogate(indrani_estimates,surrogate=surrogates[i],dataset=datasets[i], standardize=True, normalize=True)[:,1])
# predictions = np.array(predictions)
# print(predictions[:,:5])
# print(predictions.shape)
# plt.figure()
# true = plt.plot(actual_times, ys, label="Observed Data",c="b", linewidth=10, alpha=0.5)
# for i in range(predictions.shape[1]):
#     plt.scatter(actual_times, predictions[:,i])
# plt.legend()
# plt.savefig("IndraniEstSurr.png")
# plot_confidence_intervals(predictions.transpose(), title="Indrani_CI_EstSurr")

# # check if anything in the dataset is outta bounds!
# for i in range(indrani_estimates.shape[0]):
#     for j in range(indrani_estimates.shape[1]):
#         if indrani_estimates[i,j] > upper_bound[j]:
#             print("TOO LARGE")
#             print("i:",i,"+","j:",j)
#         if indrani_estimates[i,j] < lower_bound[j]:
#             print("TOO SMALL")
#             print("i:",i,"+","j:",j)



# rate_labels = ["k1", "k2", "C1", "C2", "g"]
# results_dict = {}
# for label, observed_data in Y_dict.items(): 
#     # get the y observed values
#     ys = get_corresponding_y(observed_data, actual_times)
#     print(ys)
#     # exit(0)
#     # multiple n_runs
#     estimates = []
#     costs = []
#     for r in range(n_runs):
#     # optimize multi-cost function with all the surrogates and the datasets, also redeclare optimizer every time
#         optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=dim, bounds=bounds, options=options)
#         cost, pos = optimizer.optimize(multi_indrani_cost_fxn, iters=n_iters, surrogates=surrogates, ys=ys, datasets=datasets)
#         estimates.append(pos)
#         costs.append(cost)
#     estimates = np.array(estimates)
#     estimates = np.hstack((estimates, np.array(costs).reshape(-1,1)))
#     results_dict[label] = estimates
#     # results_dict[label + "_cost"] = costs
#     print(estimates.shape)
#     # plot_confidence_intervals(estimates, title=label,labels=rate_labels)

# Set the print options to display in decimal format
# np.set_printoptions(precision=6, suppress=True)
# for label, estimates in results_dict.items():
#     # print("MEAN ESTIMATES FOR " + label + ":")
#     # print(np.mean(estimates, axis=0))
#     # print("STANDARD DEVIATION:")
#     # print(np.std(estimates, axis= 0))
#     np.savetxt(label + ".csv", estimates, delimiter=",", fmt='%.5f')
#     ys = get_corresponding_y(Y_dict[label], actual_times)
#     # evaluate_pso_estimate(np.mean(estimates, axis=0), surrogates[0], ys[0], datasets[0])
#     print("---------------------------------------------------------------------------------------------") 
#     print(label)
#     # print("Estimates:")
#     # print(estimates)
#     for i in range(len(ys)):
#         evaluate_pso_estimate(np.mean(estimates[:,:dim], axis=0), surrogates[i], ys[i], datasets[i])
#     if label == "46L_50F_53V":
#         print("------------------------------- When Using Indrani's Estimates -------------------------------")
#         for i in range(len(ys)):
#             evaluate_pso_estimate(np.array([0.00075, 3.7, 7292.38, 1.07, 0.004432]), surrogates[i], ys[i], datasets[i])
#         print("---------------------------------------------------------------------------------------------")
#     if label == "CD3z_46L":
#         print("------------------------------- When Using Indrani's Estimates -------------------------------")
#         for i in range(len(ys)):
#             evaluate_pso_estimate(np.array([0.00026, 9.232, 4499.434, 1.289, 0.003268]), surrogates[i], ys[i], datasets[i])
        
#         print("---------------------------------------------------------------------------------------------")
    
#     if label == "46L_50F":
#         print("------------------------------- When Using Indrani's Estimates -------------------------------")
#         for i in range(len(ys)):
#             evaluate_pso_estimate(np.array([0.000593, 1.86284, 9578.257, 1.09531, 0.004975]), surrogates[i], ys[i], datasets[i])
#         print("---------------------------------------------------------------------------------------------")
    
#     if label == "46l_53V":
#         print("------------------------------- When Using Indrani's Estimates -------------------------------")
#         for i in range(len(ys)):
#             evaluate_pso_estimate(np.array([0.0006219, 7.04903588, 6757.33744, 1.06661, 0.00456079]), surrogates[i], ys[i], datasets[i])
#         print("---------------------------------------------------------------------------------------------")
    
#     if label == "CD3z_human":
#         print("------------------------------- When Using Indrani's Estimates -------------------------------")
#         for i in range(len(ys)):
#             evaluate_pso_estimate(np.array([0.0005028, 4.25944, 9742.070, 1.35489753, 0.00464946]), surrogates[i], ys[i], datasets[i])
#         print("---------------------------------------------------------------------------------------------")
    
#     if label == "CD3z_mouse":
#         print("------------------------------- When Using Indrani's Estimates -------------------------------")
#         for i in range(len(ys)):
#             evaluate_pso_estimate(np.array([0.02509417, 6.80673, 581.867, 2.811, 0.0003667823]), surrogates[i], ys[i], datasets[i])
#         print("---------------------------------------------------------------------------------------------")
        
# # using indranis 
# # labels = ["46L_50F_53V", "46L_50F", "46L_53V", "CD3z_46L", "CD3z_mouse", "CD3z_human"]