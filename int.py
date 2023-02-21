import torch as tc
import numpy as np
import matplotlib.pyplot as plt 


# 
class DumbInterpreter:
    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.model = tc.load(modelPath)

    def plot(self, path, thetaStar, thetaFixed, nCols, nSteps):
        # load model that was saved.
        nIn = self.model.input_size
        nOut = self.model.output_size

        # create some data
        nSteps = 10
        thetas = np.zeros(nIn) + thetaFixed 
        thetas[thetaStar] = 0
        print("Interpreting Input Index:", thetaStar + 1, " from Model Path:", self.modelPath)
        # pick a rate constant to interpret
        # run model on step sizes of rate constants
        changing_inputs = []
        outputs = []
        for i in range(nSteps):
            changing_inputs.append(thetas[thetaStar])
            thetas[thetaStar] += (1.0 / nSteps)

            input = tc.from_numpy(thetas)
            if next(self.model.parameters()).is_cuda:
                input = input.to(tc.device("cuda"))
            output = self.model(input).cpu().detach().numpy()
            outputs.append(output)

        changing_inputs = np.array(changing_inputs)
        outputs = np.array(outputs)

        # plot change in outputs of mean counts (because that's easier to interpret)
        # subplotting 

        nRows = int(nOut / nCols)
        if nOut % nCols != 0:
            nRows+=1

        f, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(20,20))
        moment = 0

        for r in range(nRows):
            for c in range(nCols):
                if moment < nOut:
                    axs[r,c].plot(changing_inputs, outputs[:,moment], label='o' + 'moment')
                    moment+=1

        plt.xlabel("Theta" + str(thetaStar + 1))
        plt.ylabel("Output Value")
        # filename = "int_theta" + str(thetaStar + 1) + "_nl6"
        plt.savefig(path +"_theta" + str(thetaStar + 1) + '.png')







if __name__ == "__main__":
    nl6Int= DumbInterpreter(modelPath="model/nl6_poster.pt") 
    for i in range(nl6Int.model.input_size):
        nl6Int.plot(path="graphs/interpretability/nl6", thetaStar=i, thetaFixed=0.2, nCols=6, nSteps=10)
    
    l3Int = DumbInterpreter(modelPath="model/l3p_poster.pt") 
    l3Int.plot(path="graphs/interpretability/l3", thetaStar=0, thetaFixed=0.2, nCols=3, nSteps=10)
    l3Int.plot(path="graphs/interpretability/l3", thetaStar=1, thetaFixed=0.2, nCols=3, nSteps=10)
    l3Int.plot(path="graphs/interpretability/l3", thetaStar=2, thetaFixed=0.2, nCols=3, nSteps=10)
    
    gdag = DumbInterpreter(modelPath="model/gdag_poster.pt")
    for i in range(gdag.model.input_size):
        gdag.plot(path="graphs/interpretability/gdag", thetaStar=i, thetaFixed=0.2, nCols=3, nSteps=10)















# # load model that was saved.
# model = tc.load("model/nl6_poster.pt")
# model.eval()
# n_inputs = model.input_size
# n_outputs = model.output_size

# # create some data
# theta_star = 0 # what index we care about
# n_steps = 10
# thetas = (2*np.ones(n_inputs)) / 10.0 # convert to a set of 0.2's
# thetas[theta_star] = 0
# print("Interpreting Input:", theta_star + 1)
# # pick a rate constant to interpret
# # run model on step sizes of rate constants
# changing_inputs = []
# outputs = []
# for i in range(n_steps):
#     changing_inputs.append(thetas[theta_star])
#     thetas[theta_star] += (1.0 / n_steps)

#     input = tc.from_numpy(thetas)
#     if next(model.parameters()).is_cuda:
#         input = input.to(tc.device("cuda"))
#     output = model(input).cpu().detach().numpy()
#     outputs.append(output)

# changing_inputs = np.array(changing_inputs)
# outputs = np.array(outputs)

# # plot change in outputs of mean counts (because that's easier to interpret)
# # subplotting 
# n_cols = 6 # Use Four Columns For Width purposes
# n_rows = int(n_outputs / n_cols)
# if n_outputs % n_cols != 0:
#     n_rows+=1


# f, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20,20))

# moment = 0

# for r in range(n_rows):
#     for c in range(n_cols):
#         if moment < n_outputs:
#             axs[r,c].plot(changing_inputs, outputs[:,moment], label='o' + 'moment')
#             moment+=1

# plt.xlabel("Theta" + str(theta_star + 1))
# plt.ylabel("Output Value")
# filename = "int_theta" + str(theta_star + 1) + "_nl6"
# plt.savefig(filename + '.png')
