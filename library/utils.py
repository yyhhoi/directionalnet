import pickle

# [f1_begin, f1_end, f2_begin, f2_end]

AB_list = [
        [1, 0, 0, 1],  # Run from A to B
        [1, 0, 1, 1],  # Run from A+B to B
        [1, 1, 0, 1],  # Run from A to A+B
]
BA_list = [
        [0, 1, 1, 0],  # Run from B to A
        [0, 1, 1, 1],  # Run from B to A+B
        [1, 1, 1, 0],  # Run from A+B to A
]

def load_pickle(fp):
    with open(fp, "rb") as f:
        f_dict = pickle.load(f)
    return f_dict



def mat2direction(loc):

    if loc in AB_list:
        return 'A->B'
    elif loc in BA_list:
        return 'B->A'
    else:
        return loc
    
def check_iter(a):
    
    if hasattr(a, '__iter__'):
        return True
    else:
        return False


