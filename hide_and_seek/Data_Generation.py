"""
Generating Synthetic Data for Synthetic Examples

X ~ N(0,I) where d = 100

Y = 1/(1+logit)
""" 
#%% Necessary packages
import numpy as np 

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


#%% X Generation
def generate_X (n=10000, num_features=11):
    
    X = np.random.randn(n, num_features)
    
    return X

# ---- Base definitions ---- #
def syn1(X):
    logit = np.exp(X[:,0]*X[:,1])
    features = [0,1]
    return [logit], [features] #features expected to be in a list of lists by Label_Generation

def syn2(X):
    logit = np.exp(np.sum(X[:,2:6]**2, axis=1) - 4.0)
    features = [2,3,4,5]
    return [logit], [features]

def syn3(X):
    logit = np.exp(-10*np.sin(0.2*X[:,6]) + np.abs(X[:,7]) + X[:,8] + np.exp(-X[:,9]) - 2.4) #easier
    # logit = np.exp(-10*np.sin(2*X[:,6]) + np.abs(X[:,7]) + X[:,8] + np.exp(-X[:,9]) - 2.4) #harder
    #note: in their code, L2X and INVASE both used 0.2, despite writing otherwise in their papers
    #L2X: logit = np.exp(-100 * np.sin(0.2*X[:,0]) + abs(X[:,1]) + X[:,2] + np.exp(-X[:,3])  - 2.4)
    #INVASE: logit = np.exp(-10 * np.sin(0.2*x[:, 6]) + abs(x[:, 7]) + x[:, 8] + np.exp(-x[:, 9])  - 2.4)
    
    #this, below, is what INVASE write in their paper. L2X have the same but with "-100" instead of "-10" and with (arbitrarily) different features
    # logit = np.exp(-10*np.sin(2*X[:,6]) + 2*np.abs(X[:,7]) + X[:,8] + np.exp(-X[:,9]))
    features = [6,7,8,9]
    return [logit], [features]

def syn_2a(X):
    """Combines syn1 and syn2 style. But note syn2 is slightly different - this matche's invase's paper setting"""
    logit1 = np.exp(X[:,0]*X[:,1])
    logit2 = np.exp(np.sum(X[:,2:6]**2, axis=1) - 2.0)
    logit = 0.5 * logit1 + 0.5 * logit2

    features = [0,1,2,3,4,5]
    return [logit], [features]

# ---- Complex variant factory ---- #
def create_complex_syn(syn_functions, allocation_logic):
    """
    Creates a synthetic function combining multiple base functions with mandatory allocation logic.
    
    Args:
        syn_functions: List of base synthetic functions (e.g., [syn1, syn2])
        allocation_logic: Function that takes X and num_logits, returns array of indices (0 to num_logits-1)
    
    Returns:
        Function that returns list of logits and list of feature sets
    """
    def complex_syn(X):
        all_logits = []
        all_features = []
        for syn_func in syn_functions:
            logits, features = syn_func(X)
            all_logits.extend(logits)
            all_features.extend(features)
        
        if len(all_logits) < 2:
            raise ValueError("At least two synthetic functions are required for complex synthesis.")
        
        # Combine logits based on allocation
        idx, _ = allocation_logic(X, len(all_logits))
        combined_logit = np.zeros_like(all_logits[0])
        for i in range(len(all_logits)):
            combined_logit += all_logits[i] * (idx == i).astype(float)
        
        return [combined_logit], all_features #all_features is a list of lists
    
    return complex_syn

# ---- Allocation logic functions ---- #

# def single_allocation(X, num_logits):
#     if num_logits != 1:
#         raise ValueError("Single allocation requires exactly 1 logit")
#     idx = np.zeros(X.shape[0], dtype=int)  # All instances belong to the single logit
#     switch_features = {}

#     return idx, switch_features

def one_switch_x11(X, num_logits):
    """
    binary allocation:
    first logit for X[:,10] < 0, second for X[:,10] >= 0

    returns:
    - idx: an array of indices where each index corresponds to the logit that applies to that instance.
    - switch_features: a dictionary where key is the feature used in switching and value is the indices that considered the switch
    
    """
    if num_logits != 2:
        raise ValueError("Binary allocation requires exactly 2 logits")
    idx = np.zeros(X.shape[0], dtype=int)
    idx[X[:,10] >= 0] = 1
    switch_features = {10: np.arange(len(X))}
    return idx, switch_features

def quartile_allocation(X, num_logits):
    if num_logits != 4:
        raise ValueError("Quartile allocation requires exactly 4 logits")
    thresholds = np.percentile(X[:,10], [25, 50, 75])
    idx = np.zeros(X.shape[0], dtype=int)
    idx[X[:,10] > thresholds[0]] = 1
    idx[X[:,10] > thresholds[1]] = 2
    idx[X[:,10] > thresholds[2]] = 3
    switch_features = {} #to update if using
    return idx, switch_features

def two_switch_x1_x2(X, num_logits):
    if num_logits != 4:
        raise ValueError("Ternary allocation requires exactly 3 logits")
    
    idx = np.zeros(X.shape[0], dtype=int)
    idx[(X[:,0] < 0) & (X[:,1] < 0)] = 0
    idx[(X[:,0] < 0) & (X[:,1] >= 0)] = 1
    idx[(X[:,0] >= 0) & (X[:,1] < 0)] = 2
    idx[(X[:,0] >= 0) & (X[:,1] >= 0)] = 3

    switch_features = {0: np.arange(len(X)),
                       1: np.arange(len(X))
                       }
    return idx, switch_features

# ---- Registry ---- #
SYN_FUNCTIONS = {
    "Syn1": syn1,
    "Syn2": syn2,
    "Syn3": syn3,
    "Syn4": create_complex_syn([syn1, syn2], one_switch_x11),
    "Syn5": create_complex_syn([syn1, syn3], one_switch_x11),
    "Syn6": create_complex_syn([syn2, syn3], one_switch_x11),
    "Syn7": create_complex_syn([syn1, syn2, syn3, syn_2a], two_switch_x1_x2)
}

# ---- Unified generator ---- #
def Label_Generation(X, data_type):
    n, d = X.shape
    if data_type not in SYN_FUNCTIONS:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    logit, features = SYN_FUNCTIONS[data_type](X)
    logit = logit[0] #there is only one

    # ---- Probabilities ---- #
    prob_1 = (1 / (1 + logit)).reshape(n, 1)
    prob_0 = (logit / (1 + logit)).reshape(n, 1)
    prob_y = np.concatenate((prob_0, prob_1), axis=1)
    
    # ---- Ground truth mask ---- #
    
    g_truth = np.zeros((n, d))
    if data_type in ["Syn1", "Syn2", "Syn3"]:
        switch_features = {}
        idx = np.zeros(n, dtype=int) # all instances use the single logit mechanism
    elif data_type in ["Syn4", "Syn5", "Syn6"]:
        idx, switch_features = one_switch_x11(X, 2)
    elif data_type == "Syn7":
        idx, switch_features = two_switch_x1_x2(X, 4)
    elif data_type == "Syn8":
        idx, switch_features = quartile_allocation(X, 4)
    else:
        raise ValueError(f"Unknown data_type for ground truth: {data_type}")
    for i, allocated_feats in enumerate(features): # features is a list of lists. For simple syns, where there is no split, i=0 only
        for feat in allocated_feats:
            g_truth[idx == i, feat] = 1
    if len(switch_features) > 0:
        for feat, switch_indices in switch_features.items():
            g_truth[switch_indices, feat] = 1

    # ---- Sampling ---- #
    y = np.zeros((n, 2))
    y[:,0] = np.random.binomial(1, prob_0.flatten())
    y[:,1] = 1 - y[:,0]
    return y, prob_y, g_truth
    
#%% Generate X and Y
'''
n: Number of samples
data_type: Syn1 to Syn6
out: Y or Prob_Y
'''    
    
def generate_data(n=10000, data_type='Syn1', seed = 0, out = 'Y',
                  num_features=11, data_mode_detail='synthetic'):

    # For same seed
    np.random.seed(seed)

    # X generation
    if data_mode_detail == 'synthetic':
        X = generate_X(n, num_features)
    
    elif 'credit_data' in data_mode_detail:
        from tests_credit_default.credit_default_tools import load_data
        X_train, X_val, X_test = load_data()
        if data_mode_detail == 'credit_data_train':
            X = X_train
        elif data_mode_detail == 'credit_data_val': #note, in experiments, I had val and test swapped
            X = X_val
        elif data_mode_detail == 'credit_data_test':
            X = X_test
    
    if hasattr(X, "to_numpy"):
        X = X.to_numpy(dtype=float)

    Y, Prob_Y, g_truth = Label_Generation(X, data_type)
    
    # Output
    if out == 'Prob':
        Y_Out = Prob_Y
    elif out == 'Y':
        Y_Out = Y
    else:
        raise ValueError("param 'out' must be 'Y' or 'Prob'")
        
    return X, Y_Out, g_truth

if __name__ == "__main__":
    # Example usage
    g_truths = []
    Ys = []
    Xs = []
    for data_type in ['Syn7']:#['Syn1', 'Syn2', 'Syn3', 'Syn4', 'Syn5', 'Syn6']:
        print(f"Generating data for {data_type}")
        X, Y, g_truth = generate_data(n=1000, data_type=data_type, seed=42, out='Prob') #using 'Prob' here to keep deterministic for testing
        print(f"X shape: {X.shape}, Y shape: {Y.shape}, Ground truth shape: {g_truth.shape}")
        g_truths.append(g_truth)
        Xs.append(X)
        Ys.append(Y)    
    