from joblib import Parallel, delayed
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm
import gc
import sys

from tools import run_feature_selection_model, parse_args

if __name__ == '__main__':
    args = parse_args(lmbda_default=0.1, 
                      epochs_default=10_000, 
                      batch_size_default=1_000,
                     seed_default=0, 
                     syn_idx_default=3, 
                     num_syn_features_default=11,
                     train_N_default=10_000,#10_000, 
                     test_N_default=10_000,
                     hide_hidden_dim_default=32, 
                     seek_hidden_dim_default=32,
                     hide_num_hidden_layers_default=2, 
                     seek_num_hidden_layers_default=2,
                     lmbda_exponent_default=2,
                     data_mode_default='synthetic')
                     #'credit_data_val', #'synthetic')
    
    lmbda = args.lmbda
    batch_size = args.batch_size
    epochs = args.epochs

    seed = args.seed
    syn_idx = args.syn_idx
    num_syn_features = args.num_syn_features
    train_N = args.train_N
    test_N = args.test_N

    data_mode = args.data_mode

    hide_hidden_dim = args.hide_hidden_dim
    seek_hidden_dim = args.seek_hidden_dim
    hide_num_hidden_layers = args.hide_num_hidden_layers
    seek_num_hidden_layers = args.seek_num_hidden_layers
    lmbda_exponent = args.lmbda_exponent
    
    batchnorm_hs = args.batchnorm_hs
    
    # args.return_losses_on_val = True #to edit manually

    return_losses_on_val = args.return_losses_on_val
    

    print(f"Running with seed={seed}, lmbda={lmbda}")

    single_data_set=True
    # syn_idx = 3 #use params instead if needing to run one syn at a time e.g. for 1_000_000 training samples

    data_sets = ['Syn1','Syn2','Syn3','Syn4','Syn5','Syn6']
    folder_for_pickle = 'AI_STATS_normalized/1e6'#None#'ICML_val_losses'
    model_type = "realx"
    task = 'classification'
    save_experiment_data = True
    
    num_important_features = 'use_gtruth'
    # num_important_features_s = [3,4]
    
    if model_type == 'invase' and (lmbda != 0.1): raise ValueError("invase usually requires lambda = 0.1")
    if model_type == 'hide_and_seek' and (lmbda != 0.3): raise ValueError("hide_and_seek usually requires lambda = 0.3")
    if model_type == 'lime' and (epochs != 500 or batch_size != None): raise ValueError("lime baseline usually requires uses epochs=500, batch_size=None")
    if model_type == 'realx' and (lmbda != 0.15): raise ValueError("realx usually requires lmbda = 0.15")
    if num_syn_features != 11:
        print("WARNING NOT TESTING ON STANDARD NUMBER OF SYN FEATURES")
    
    if single_data_set == True:
        data_set = data_sets[syn_idx]
        results = run_feature_selection_model(data_type=data_set, 
                            folder_for_pickle=folder_for_pickle,
                            num_important_features=num_important_features,
                            model_type=model_type,
                            batch_size=batch_size,
                            epochs=epochs,
                            lmbda=lmbda,
                            task=task,
                            hide_hidden_dim=hide_hidden_dim, 
                            seek_hidden_dim=seek_hidden_dim, 
                            hide_num_hidden_layers=hide_num_hidden_layers, 
                                seek_num_hidden_layers=seek_num_hidden_layers,
                                train_N = train_N,
                                test_N = test_N,
                                seed=seed,
                                num_syn_features=num_syn_features,
                                return_results=True,
                                batchnorm_hs=batchnorm_hs,
                                save_experiment_data=save_experiment_data,
                                lmbda_exponent=lmbda_exponent,
                                return_losses_on_val=return_losses_on_val,
                                data_mode=data_mode
                                )
        row = {
                "syn": data_set,
                "TPR": round(results["TPR_mean"]),
                "FDR": round(results["FDR_mean"]),
                "F1": round(results["f1"])
            }

        df = pd.DataFrame([row])
        
    else:
        print('running in parallel')
        results_list = Parallel(n_jobs=len(data_sets))(
        delayed(run_feature_selection_model)(data_type=data_set, 
                                folder_for_pickle=folder_for_pickle,
                                num_important_features=num_important_features,
                                model_type=model_type,
                                batch_size=batch_size,
                                epochs=epochs,
                                lmbda=lmbda,
                                task=task,
                                hide_hidden_dim=hide_hidden_dim, 
                                seek_hidden_dim=seek_hidden_dim, 
                                hide_num_hidden_layers=hide_num_hidden_layers, 
                                    seek_num_hidden_layers=seek_num_hidden_layers,
                                    train_N = train_N,
                                test_N = test_N,
                                seed=seed,
                                num_syn_features=num_syn_features,
                                return_results=True,
                                batchnorm_hs=batchnorm_hs,
                                save_experiment_data=save_experiment_data,
                                lmbda_exponent=lmbda_exponent,
                                return_losses_on_val=return_losses_on_val,
                                data_mode=data_mode)
        for data_set in tqdm(data_sets)
        )

        rows = []
        for data_set, results in zip(data_sets[:-1], results_list[:-1]):
            rows.append({
                "syn": data_set,
                "TPR": round(results["TPR_mean"]),
                "FDR": round(results["FDR_mean"]),
                "F1": round(results["f1"])
            })

        df = pd.DataFrame(rows)

        # compute the means
        mean_row = {
            "syn": "mean",
            "TPR": round(df["TPR"].mean()),
            "FDR": round(df["FDR"].mean()),
            "F1": round(df["F1"].mean())
        }

        # append the mean row
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    print("\n===== RUN CONFIG =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("======================\n")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

    gc.collect()
    if model_type != 'l2x' and model_type != 'realx':
        import torch
        torch.cuda.empty_cache()

