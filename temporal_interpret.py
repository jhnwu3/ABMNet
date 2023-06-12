from modules.utils.interpret import *
import pickle 

path = "data/time_series/indrani_zeta.pickle"
data = pickle.load(open(path, "rb"))
interpret_temporal(data, threshold = 0.00001, path="graphs/temporal/zeta")