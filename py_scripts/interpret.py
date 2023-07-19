from modules.data.mixed import *
from modules.utils.graph import *
from modules.utils.interpret import *
from modules.utils.evaluate import *
import torch
if __name__ == "__main__":


    # nl6dataset = ABMDataset("data/static/NL6P_t05.csv", root_dir="data/", standardize=False, norm_out=True)
    # nl6Int= DumbInterpreter(modelPath="model/nl6_poster_default_inputs.pt", dataset=nl6dataset, normalize_out=True) 
    # wt = np.loadtxt("pso/gmm_weight/nl6p_t05.txt")
    # nl6Int.plot_contour(path="graphs/contour/nl6.png",nCols=3, groundTruthTheta = np.array([0.1, 0.1, 0.95, 0.17, 0.05, 0.18]), resolution=50, y=np.array([1.40012,1757.38,209.96,14.0588,121.369,90.9622,0.728361,273328,6695.53,201.283,6369.12,859.989,-258.233,53.35,7.06668,-18.5846,17.0746,-4524.28,-958.283,5467.64,-992.478,790.754,-1962.44,2254.3,-690.745,162.181,-220.746]))
    # nl6Int.plot_gmm_contour(path="graphs/contour/nl6gmm.png",nCols=3, groundTruthTheta = np.array([0.1, 0.1, 0.95, 0.17, 0.05, 0.18]), resolution=50, y=np.array([1.40012,1757.38,209.96,14.0588,121.369,90.9622,0.728361,273328,6695.53,201.283,6369.12,859.989,-258.233,53.35,7.06668,-18.5846,17.0746,-4524.28,-958.283,5467.64,-992.478,790.754,-1962.44,2254.3,-690.745,162.181,-220.746]), wt=wt)
    # nl6Int.plot_with_ground_truth(plotPath="graphs/interpretability/nl6_default_in", groundTruthPath="data/NL6_1k.csv",thetaStar=0, nCols=6)
    # nl6Int.plot(path="graphs/interpretability/nl6", thetaStar=0, thetaFixed=0.2, nCols=6, nSteps=10)
    
    baseName = "l3p_t"
    # models = []
    # wts = []
    # y = np.loadtxt("pso/truth/l3p_t123_mom.txt")
    # # magic number 3 for 3 tpts
    # for i in range(3):
    #     wts.append(np.loadtxt("pso/gmm_weight/" + baseName + str(i + 1) + ".txt"))
    #     models.append(tc.load("model/" + baseName + str(i+1) + ".pt"))

    # l3IntMulti = MultiInterpreter(models=models)
    # l3IntMulti.plot_mgmm_contour("graphs/contour/l3mgmm.png", nCols=3, groundTruthTheta = np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]), resolution=100, y=y, wts=wts, levels=40)
    
    
    # l3NormDataset = ABMDataset("data/static/l3p_10k_t3_5kss.csv", root_dir="data/", standardize=True, norm_out=True)
    l3Dataset100k = ABMDataset("data/static/l3p_100k.csv", root_dir="data/", standardize=False, norm_out=True)
    
    model = torch.load("model/l3p_100k_large_batch_normed.pt")
    print(model.parameter)
    # mse, t, predicted, truth = evaluate(model, l3Dataset100k, use_gpu=True, batch_size=512)
    # print("Time Taken to Evaluate 100k params:", t)
    # print("Overall Average MSE:", mse)
    # plot_scatter(truth, predicted, output="graphs/L3/l3p_100k_large_batched", nSpecies=3)
    # l3Int = DumbInterpreter(modelPath="model/l3p_10k_small_res_t3.pt", dataset=l3NormDataset, normalize_out=True, standardize_in=True)
    # l3Int100k = DumbInterpreter(modelPath="model/l3p_100k_small_t3.pt", dataset=l3Dataset100k, normalize_out=True, standardize_in=True)
    # wt = np.loadtxt("pso/gmm_weight/l3p_t3.txt")
    # l3Int.plot_contour(path="graphs/contour/l3_norm_t3.png",nCols=3, groundTruthTheta = np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]), resolution=100, y=np.array([12.4509,  6.9795, 9.06247, 93.9796, 31.9489, 84.5102, 53.8117, 72.7715, 47.3049]), levels=40)
    # l3Int.plot_gmm_contour(path="graphs/contour/l3_norm_gmm_t3.png",nCols=3, groundTruthTheta = np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]), resolution=100, y=np.array([12.4509,  6.9795, 9.06247, 93.9796, 31.9489, 84.5102, 53.8117, 72.7715, 47.3049]), wt=wt, levels = 40)
    # l3Int100k.plot_contour(path="graphs/contour/l3_norm_100k_t3.png",nCols=3, groundTruthTheta = np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]), resolution=100, y=np.array([12.4509,  6.9795, 9.06247, 93.9796, 31.9489, 84.5102, 53.8117, 72.7715, 47.3049]), levels=40)
    # l3Int100k.plot_gmm_contour(path="graphs/contour/l3_norm_gmm_100k_t3.png",nCols=3, groundTruthTheta = np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]), resolution=100, y=np.array([12.4509,  6.9795, 9.06247, 93.9796, 31.9489, 84.5102, 53.8117, 72.7715, 47.3049]), wt=wt, levels = 40)
    
    # l3tTestDataset = ABMDataset("data/time_series/l3p_unseen_data.csv", root_dir="data/time_series/")
    # l3tTrainDataset = ABMDataset("data/time_series/l3pt_i.csv", root_dir="data/")
    # l3T = DumbInterpreter(modelPath="model/l3p_i.pt", dataset=l3tTrainDataset)
    # mse, t, predicted, tested = l3T.evaluate(l3tTestDataset, use_gpu=True)
    # l3T.plot_scatter(tested, predicted, output='graphs/scatter/l3_t_test')
    
    # l3Int.plot_with_ground_truth(plotPath="graphs/interpretability/l3p_default_in", groundTruthPath="data/l3p_k1.csv",thetaStar=0, nCols=3)
    # l3Int.plot(path="graphs/interpretability/l3", thetaStar=0, thetaFixed=0.2, nCols=3, nSteps=10)
    # l3Int.plot(path="graphs/interpretability/l3", thetaStar=1, thetaFixed=0.2, nCols=3, nSteps=10)
    # l3Int.plot(path="graphs/interpretability/l3", thetaStar=2, thetaFixed=0.2, nCols=3, nSteps=10)
    # gdagdataset = ABMDataset("data/gdag1300sss_covs.csv", root_dir="data/", transform=False, standardize=False, norm_out=True)
    # gdag = DumbInterpreter(modelPath="model/gdag_default_input.pt")
    
    # gdag.plot(path="graphs/interpretability/gdag1300ss_default_in", thetaStar=0, thetaFixed=0.1, nCols=3, nSteps=20)
    # for i in range(gdag.model.input_size):
    #     gdag.plot(path="graphs/interpretability/gdag1300ss_default_in", thetaStar=i, thetaFixed=0.1, nCols=3, nSteps=10)
