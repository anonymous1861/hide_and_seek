import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import os
import fsspec

import re

def order_indices_by_priority(df, priority_order):

    priority_order = [idx for idx in priority_order if idx in df.index]
    remaining = [idx for idx in df.index if idx not in priority_order]
    
    new_index_order = priority_order + remaining
    df = df.reindex(new_index_order)
    return df

def load_pickle(path):  
    # Load the dictionary from the pickle file
    if "/home" not in path:
        path = os.path.expanduser(f'~/{path}')
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def load_results_from_pickle(folder,
                            time_threshold=None,
                            keep_run_type=None,
                            is_output=False,
                            is_synthetic=True): #"%y_%m_%d_%H_%M_%S" "25_07_08_14_59_36"
    
    path2 = f'~/Data/{folder}/*'
    # Create a filesystem instance (local filesystem in this case)
    fs = fsspec.filesystem('file')
    # List all files in a directory (recursive=False by default)
    ALL_FILES2 = fs.glob(path2)
    
    results_files2 = [file for file in ALL_FILES2 if "results" in file]
    
    all_series = []
    for file2 in results_files2:
        temp_dict = load_pickle(file2)
        filtered_dict = {k: v for k, v in temp_dict.items() if k != "Output"}

        if time_threshold is not None:
            threshold = datetime.strptime(time_threshold, "%y_%m_%d_%H_%M_%S")
            if ('time_run' not in filtered_dict.keys()) or (pd.to_datetime(filtered_dict["time_run"], format="%Y-%m-%d_%H-%M-%S", errors="coerce") < threshold):
                continue
        if keep_run_type is not None:
            if filtered_dict['run_type'] != keep_run_type:
                continue
        if is_output:
            output = temp_dict['Output']

            metrics = ['roc_auc_score', 'average_precision_score', 'accuracy_score']
            models = ['val', 'dis']
            stats = ['mean', 'std']
            
            new_dict = {}
            
            for i, metric in enumerate(metrics):
                for j, model in enumerate(models):
                    new_dict[f'{metric}_{model}'] = output[i, j]
                    # new_dict[f'{metric}_{model}_std'] = output[i, j+2]
        
        match = re.search(r"_(Syn\d+)_", file2)
        
        if is_synthetic:
            if match:
                syn = match.group(1)
        
        if is_output:
            series1 = pd.Series(new_dict)
            if is_synthetic:
                series2['syn'] = syn
            series2 = pd.Series(filtered_dict)
            # Concatenate along the rows
            combined_series = pd.concat([series2, series1],axis=0)
            # combined_series.name = series1.name
            all_series.append(combined_series)
        else:
            series2 = pd.Series(filtered_dict)
            # series2.name = syn
            if is_synthetic:
                series2['syn'] = syn
            all_series.append(series2)
        
    # return all_series
    iclr_results = pd.concat(all_series, axis=1)
    iclr_results = iclr_results[sorted(iclr_results.columns)]
    # if 'run_type' in iclr_results.index:
    #     iclr_results = iclr_results.drop('run_type')
    # display(np.round(iclr_results,4))
    priority_order = [
    "syn", "TPR_mean", "FDR_mean", "TPR_std", "FDR_std",
    "roc_auc_score_val", "roc_auc_score_dis",
    "average_precision_score_val", "average_precision_score_dis",
    "accuracy_score_val", "accuracy_score_dis"
    ]
    iclr_results = order_indices_by_priority(iclr_results, priority_order)
    return iclr_results

def performance_metric(score, g_truth): #not used

        n = len(score)
        Temp_TPR = np.zeros([n,])
        Temp_FDR = np.zeros([n,])
        
        for i in range(n):
    
            # TPR    
            TPR_Nom = np.sum(score[i,:] * g_truth[i,:])
            TPR_Den = np.sum(g_truth[i,:])
            Temp_TPR[i] = 100 * float(TPR_Nom)/float(TPR_Den+1e-8)
        
            # FDR
            FDR_Nom = np.sum(score[i,:] * (1-g_truth[i,:]))
            FDR_Den = np.sum(score[i,:])
            Temp_FDR[i] = 100 * float(FDR_Nom)/float(FDR_Den+1e-8)
    
        return np.mean(Temp_TPR), np.mean(Temp_FDR), np.std(Temp_TPR), np.std(Temp_FDR)


def tpr_fdr_analysis(df, data_set): #not used
    data_temp = df.copy()
    Sel_Prob_Test = pd.DataFrame(data_temp.loc['Sel_Prob_Test', data_set])
    score = pd.DataFrame(data_temp.loc['score', data_set])
    g_test = pd.DataFrame(data_temp.loc['g_test', data_set])
    return Sel_Prob_Test, score, g_test


def rank_runs(df): #not used
    """
    ranks runs by combination of TPR_mean and FDR_mean
    groups by time_run. assumes 6 per time_run
    """
    
    cols_to_use = ['time_run', 'TPR_mean', 'FDR_mean', 'hide_hidden_dim', 'seek_hidden_dim', 'hide_num_hidden_layers', 
'seek_num_hidden_layers', 'epochs', 'lamda','batch_size']
    
    df = df.T[cols_to_use]
    df['score'] = (df['TPR_mean']+(100-df['FDR_mean']))/2

    
    # Get the counts of each 'time_run'
    time_run_counts = df.groupby('time_run').size()

    # Find the 'time_run' values with counts not equal to 6
    time_runs_without_6 = time_run_counts[time_run_counts != 6].index
    print('runs without 6 - dropping these!')
    print(time_runs_without_6)
    
    # Filter out the rows where 'time_run' is not in the list of those without 6
    df_filtered = df[~df['time_run'].isin(time_runs_without_6)]
    
    df_filtered = df_filtered.groupby('time_run').mean()
    df_filtered = df_filtered.sort_values(by='score',ascending=False)
    print('\nbest run:')
    print(df_filtered.index[0])
    return df_filtered[['score'] + [col for col in cols_to_use if col != 'time_run']]


def compute_rowwise_metrics(row, just_f1=True):
    g = np.array(row['g_test'])        # shape (n, p)
    pred = np.array(row['binary_mask'])
    
    # True Positives, False Positives, False Negatives per row
    TP = np.sum((pred == 1) & (g == 1), axis=1)
    FP = np.sum((pred == 1) & (g == 0), axis=1)
    FN = np.sum((pred == 0) & (g == 1), axis=1)
    
    # Compute metrics per row
    TPR = TP / (TP + FN + 1e-10)
    FDR = FP / (TP + FP + 1e-10)
    F1  = 2 * TP / (2 * TP + FP + FN + 1e-10)
    
    # Take mean across all rows in this experiment
    if just_f1 == True:
        return pd.Series({
                'F1' : np.mean(F1)
            })
    else:
        return pd.Series({
                'TPR': np.mean(TPR),
                'FDR': np.mean(FDR),
                'F1' : np.mean(F1)
            })