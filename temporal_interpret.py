from modules.utils.interpret import *
import pickle 

path = "data/time_series/indrani_zeta.pickle"
data = pickle.load(open(path, "rb"))
interpret_temporal(data, threshold = 0.00001, path="graphs/temporal/zeta")

# load a model in, and now let's see can it do it from scratch? 
# how many data points in the future can it predict?
# and how many data points does it need??

