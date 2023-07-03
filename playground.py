

from modules.data.temporal import *
from modules.data.mixed import *
from modules.utils.pso import *
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import numpy as np
import torch 
import matplotlib.pyplot as plt
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
    
    



def indrani_costss(x, surrogate, y, dataset, standardize=True, normalize=True, batch=False):
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

    if batch:
        input = input.unsqueeze(dim=0)
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
    plt.xlabel('Features')
    plt.ylabel('Values')
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
full_seq_dataset = TemporalDataset("data/time_series/indrani_zeta_ca_no_zeroes.pickle", min_max_scale=True)

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
# get their corresponding list of actual time-values
actual_times = full_seq_dataset.times[time_pts]
print("--- Times: -----")
print(actual_times)
surrogates = []
prefix = "model/ixr3k_zeta_ca_t"
suffix = "_res_batch_full.pt"

dataset_prefix = "data/static/indrani/indrani_zeta_ca_t"
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
print(bounds)
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
num_particles = 2500  # Number of particles in the swarm
n_iters = 600
dim = 5  # Dimensionality of the problem

n_runs = 5
# MISSING KD!!!
rate_labels = ["k1", "k2", "C1", "C2", "g"]
results_dict = {}
for label, observed_data in Y_dict.items(): 
    # get the y observed values
    ys = get_corresponding_y(observed_data, actual_times)
    print(ys)
    # exit(0)
    # multiple n_runs
    estimates = []
    costs = []
    for r in range(n_runs):
    # optimize multi-cost function with all the surrogates and the datasets, also redeclare optimizer every time
        optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=dim, bounds=bounds, options=options)
        cost, pos = optimizer.optimize(multi_indrani_cost_fxn, iters=n_iters, surrogates=surrogates, ys=ys, datasets=datasets)
        estimates.append(pos)
        costs.append(cost)
    estimates = np.array(estimates)
    estimates = np.hstack((estimates, np.array(costs).reshape(-1,1)))
    results_dict[label] = estimates
    # results_dict[label + "_cost"] = costs
    print(estimates.shape)
    # plot_confidence_intervals(estimates, title=label,labels=rate_labels)

# Set the print options to display in decimal format
np.set_printoptions(precision=6, suppress=True)
for label, estimates in results_dict.items():
    # print("MEAN ESTIMATES FOR " + label + ":")
    # print(np.mean(estimates, axis=0))
    # print("STANDARD DEVIATION:")
    # print(np.std(estimates, axis= 0))
    np.savetxt(label + ".csv", estimates, delimiter=",", fmt='%.5f')
    ys = get_corresponding_y(Y_dict[label], actual_times)
    # evaluate_pso_estimate(np.mean(estimates, axis=0), surrogates[0], ys[0], datasets[0])
    print("---------------------------------------------------------------------------------------------") 
    print(label)
    # print("Estimates:")
    # print(estimates)
    for i in range(len(ys)):
        evaluate_pso_estimate(np.mean(estimates[:,:dim], axis=0), surrogates[i], ys[i], datasets[i])
    if label == "46L_50F_53V":
        print("------------------------------- When Using Indrani's Estimates -------------------------------")
        for i in range(len(ys)):
            evaluate_pso_estimate(np.array([0.00075, 3.7, 7292.38, 1.07, 0.004432]), surrogates[i], ys[i], datasets[i])
        print("---------------------------------------------------------------------------------------------")
    if label == "CD3z_46L":
        print("------------------------------- When Using Indrani's Estimates -------------------------------")
        for i in range(len(ys)):
            evaluate_pso_estimate(np.array([0.00026, 9.232, 4499.434, 1.289, 0.003268]), surrogates[i], ys[i], datasets[i])
        
        print("---------------------------------------------------------------------------------------------")
    
    if label == "46L_50F":
        print("------------------------------- When Using Indrani's Estimates -------------------------------")
        for i in range(len(ys)):
            evaluate_pso_estimate(np.array([0.000593, 1.86284, 9578.257, 1.09531, 0.004975]), surrogates[i], ys[i], datasets[i])
        print("---------------------------------------------------------------------------------------------")
    
    if label == "46l_53V":
        print("------------------------------- When Using Indrani's Estimates -------------------------------")
        for i in range(len(ys)):
            evaluate_pso_estimate(np.array([0.0006219, 7.04903588, 6757.33744, 1.06661, 0.00456079]), surrogates[i], ys[i], datasets[i])
        print("---------------------------------------------------------------------------------------------")
    
    if label == "CD3z_human":
        print("------------------------------- When Using Indrani's Estimates -------------------------------")
        for i in range(len(ys)):
            evaluate_pso_estimate(np.array([0.0005028, 4.25944, 9742.070, 1.35489753, 0.00464946]), surrogates[i], ys[i], datasets[i])
        print("---------------------------------------------------------------------------------------------")
    
    if label == "CD3z_mouse":
        print("------------------------------- When Using Indrani's Estimates -------------------------------")
        for i in range(len(ys)):
            evaluate_pso_estimate(np.array([0.02509417, 6.80673, 581.867, 2.811, 0.0003667823]), surrogates[i], ys[i], datasets[i])
        print("---------------------------------------------------------------------------------------------")
        
# using indranis 
# labels = ["46L_50F_53V", "46L_50F", "46L_53V", "CD3z_46L", "CD3z_mouse", "CD3z_human"]