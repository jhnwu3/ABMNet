import sys

def train_temporal():
    return '--temporal' in sys.argv


def get_depth():
    if '-d' in sys.argv:
        return int(sys.argv[sys.argv.index('-d') + 1])
    return -1

def get_data(): 
    return sys.argv[sys.argv.index('-i') + 1]

def get_epochs():
    if '--epochs' in sys.argv:
        return int(sys.argv[sys.argv.index('--epochs') + 1])
    return -1

def get_output(): 
    return sys.argv[sys.argv.index('-o') + 1]

def get_hidden():
    if '-h' in sys.argv:
        return int(sys.argv[sys.argv.index('-h') + 1])
    return -1

def use_gpu():
    return '--gpu' in sys.argv

def save_model():
    return '--save' in sys.argv

# def transform_data():
#     return '--transform' in sys.argv

def normalize_input():
    return '--normalize' in sys.argv
 
def normalize_output():
    return '--normalize_out' in sys.argv

def cross_validate():
    return '--cross' in sys.argv
 
def get_nn_type():
    if '--type' in sys.argv:
        return sys.argv[sys.argv.index('--type') + 1]
    else: 
        return -1 

def get_n_species():
    if '--type' in sys.argv:
        return sys.argv[sys.argv.index('--nSpecies') + 1]
    else: 
        return -1 