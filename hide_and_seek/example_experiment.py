import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hide_and_seek.tools import run_feature_selection_model
from hide_and_seek.loading_runs import load_results_from_pickle

if __name__ == "__main__":

    lmbdas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    epochs = 500
    batch_size = None
    model_type = 'hide_and_seek'
    data_type = 'education_data' #this can be any name, unless experiment is on Syn1-6
    scale_data = True # normalization
    num_important_features = None #not needed for hide_and_seek, invase, realx. For other models either specify or, for synthetic data, use 'use_gtruth'
    calculate_TPR_FDR_metrics = False #set to false when no ground truth is known
    folder_for_pickle = 'testing_testing' # location to save pickled results. Will currently save in "~/Data/{folder_for_pickle}". Edit path in 'save_results_as_pickle' function if needed
    seed = 0 # hide_and_seek does not do sampling, but this is used for NN initialization and column shuffling
    task = 'classification' #note regression setting is still under development
    save_experiment_data = True #set to false if you want to save storage space and you don't need to save masks

    #load your data
    from tests_lsac.lsac_tools import load_lsac_data
    df = load_lsac_data()

    y_col = 'year_12_completion'

    y = df[y_col]
    X = df.drop(columns = y_col)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
                                                X,
                                                y,
                                                test_size=0.2,
                                                random_state=42,
                                                stratify=y,
                                            )
    
    #run model
    full_data_dict = {'x_train':X_train.values, #send as arrays, not df
                     'y_train':y_train.values,
                     'x_test':X_val.values,
                     'y_test':y_val.values,
                     'g_test':None
                     }

    #run for different lmbdas to tune
    for lmbda in lmbdas:
        results = run_feature_selection_model(data_type=data_type, 
                                folder_for_pickle=folder_for_pickle,
                                full_data_dict=full_data_dict,
                                model_type=model_type,
                                batch_size=batch_size,
                                epochs=epochs,
                                lmbda=lmbda,
                                task=task,
                                seed=seed,
                                num_important_features=num_important_features,
                                calculate_TPR_FDR_metrics=calculate_TPR_FDR_metrics,
                                save_experiment_data=save_experiment_data,
                                scale_data=scale_data
                                )
    
    #load results from all experiments into one dataframe
    results = load_results_from_pickle(folder=folder_for_pickle,
                            is_synthetic=False
                                )
    print('results dataframe contains: ', list(results.columns))