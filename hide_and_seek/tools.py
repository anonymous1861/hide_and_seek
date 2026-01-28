import numpy as np
import pandas as pd
import pickle
import argparse

from datetime import datetime

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hide_and_seek.Data_Generation import generate_data

PARENT_DIR_L2X = '/home/tyellins/projects/causal_flow/event_causality/hide_and_seek/l2x_models'

def int_or_none(x):
    return None if x.lower() == "none" else int(x)

def parse_args(lmbda_default=0.3, epochs_default=500, batch_size_default=None,
               seed_default=0, syn_idx_default=3, num_syn_features_default=11,
               train_N_default=10_000, test_N_default=10_000,
               hide_hidden_dim_default=32, seek_hidden_dim_default=32,
               hide_num_hidden_layers_default=2, seek_num_hidden_layers_default=2,
               lmbda_exponent_default=2,
               data_mode_default='synthetic'):
    parser = argparse.ArgumentParser()

    # model hyperparams
    parser.add_argument("--lmbda", type=float, default=lmbda_default)
    parser.add_argument("--epochs", type=int, default=epochs_default) #500 for håide_and_seek, 10_000 for invase
    parser.add_argument("--batch-size", type=int_or_none, default=batch_size_default) #None for hide_and_seek. 1_000 for invase

    # syn params
    parser.add_argument("--seed", type=int, default=seed_default)
    parser.add_argument("--syn-idx", type=int_or_none, default=syn_idx_default)
    parser.add_argument("--num-syn-features", type=int, default=num_syn_features_default) #11 for all syn experiments except 100 for large dataset
    parser.add_argument("--train-N", type=int, default=train_N_default)
    parser.add_argument("--test-N", type=int, default=test_N_default)
    
    parser.add_argument("--data_mode", type=str, default=data_mode_default)

    # model-structure hyperparams
    parser.add_argument("--hide-hidden-dim", type=int, default=hide_hidden_dim_default)
    parser.add_argument("--seek-hidden-dim", type=int, default=seek_hidden_dim_default)
    parser.add_argument("--hide-num-hidden-layers", type=int, default=hide_num_hidden_layers_default)
    parser.add_argument("--seek-num-hidden-layers", type=int, default=seek_num_hidden_layers_default)
    parser.add_argument("--lmbda-exponent", type=float, default=lmbda_exponent_default)
    
    parser.add_argument("--batchnorm-hs", action="store_true", help="Use batchnorm. Default is False")
    parser.add_argument("--return_losses_on_val", action="store_true", help="Default is False")

    args = parser.parse_args() #use parser.parse_args(args=[]) when copying to jupyter
    return args

def create_data(data_type, 
                data_out, 
                train_N, 
                test_N, 
                train_seed, 
                test_seed,
                num_features,
                data_mode):
    """
    Generate training and testing data for INVASE.
    
    Args:
        data_type (str): Type of dataset to generate
        data_out (str): 'Y' for binary output, 'Prob' for probability output
        train_N (int): Number of training samples
        test_N (int): Number of testing samples
        train_seed (int): Random seed for training set
        test_seed (int): Random seed for testing set
        data_mode (str): should be 'synthetic', 'credit_data_val' or 'credit_data_test'
    Returns:
        (x_train, y_train, g_train, x_test, y_test, g_test)
    """

    if data_mode == 'synthetic':
        data_detail_train = data_mode
        data_detail_test = data_mode
    elif 'credit_data' in data_mode:
        data_detail_train = 'credit_data_train'
        if 'val' in data_mode:
            data_detail_test = 'credit_data_val'
        elif 'test' in data_mode:
            data_detail_test = 'credit_data_test'


    x_train, y_train, g_train = generate_data(
        n=train_N, data_type=data_type, seed=train_seed, out=data_out,
        num_features=num_features, data_mode_detail=data_detail_train
    )

    x_test, y_test, g_test = generate_data(
        n=test_N, data_type=data_type, seed=test_seed, out=data_out,
        num_features=num_features, data_mode_detail=data_detail_test
    )

    return x_train, y_train, g_train, x_test, y_test, g_test

def compute_f1(binary_mask, g_truth):
    
    g = g_truth        # shape (n, p)
    pred = binary_mask
    
    # True Positives, False Positives, False Negatives per row
    TP = np.sum((pred == 1) & (g == 1), axis=1)
    FP = np.sum((pred == 1) & (g == 0), axis=1)
    FN = np.sum((pred == 0) & (g == 1), axis=1)
    
    # Compute metrics per row
    # TPR = TP / (TP + FN + 1e-10)
    # FDR = FP / (TP + FP + 1e-10)
    F1  = 2 * TP / (2 * TP + FP + FN + 1e-10)
    
    # return pd.Series({
    #             'TPR': np.mean(TPR),
    #             'FDR': np.mean(FDR),
    #             'F1' : np.mean(F1)
    #         })

    # Take mean across all rows in this experiment
    return np.mean(F1)*100


def prediction_metrics(y_true, y_pred_probs,
                      verbose=False):
    """
    y_true: 1D array of labels (e.g., [0, 1, 2, 0...])
    y_pred_probs: 2D array of probabilities (batch_size, num_classes)
    """
    y_true = y_true.argmax(axis=1) if y_true.ndim == 2 else y_true #if one-hot encoded
    y_pred_labels = y_pred_probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred_labels)

    # Check if we are in a binary or multiclass scenario
    num_classes = y_pred_probs.shape[1]
    if num_classes == 2:
        # For Binary: use the probability of the positive class (column 1)
        roc_auc = roc_auc_score(y_true, y_pred_probs[:, 1])
    else:
        roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr') #alternative: 'ovo'
    
    if verbose == True:
        print("Accuracy:", acc)
        print("ROC-AUC:", roc_auc)

    return acc, roc_auc       

#%% Performance Metrics
def performance_metric(binary_mask, g_truth):

    n = len(binary_mask)
    Temp_TPR = np.zeros([n,])
    Temp_FDR = np.zeros([n,])
    
    for i in range(n):

        # TPR    
        TPR_Nom = np.sum(binary_mask[i,:] * g_truth[i,:])
        TPR_Den = np.sum(g_truth[i,:])
        Temp_TPR[i] = 100 * float(TPR_Nom)/float(TPR_Den+1e-8)
    
        # FDR
        FDR_Nom = np.sum(binary_mask[i,:] * (1-g_truth[i,:]))
        FDR_Den = np.sum(binary_mask[i,:])
        Temp_FDR[i] = 100 * float(FDR_Nom)/float(FDR_Den+1e-8)

    return np.mean(Temp_TPR), np.mean(Temp_FDR), np.std(Temp_TPR), np.std(Temp_FDR)


def shuffle_numpy_cols(X, replace=False, random_state=None):
    """
    Shuffle each column of a NumPy array independently.

    Parameters:
        X (np.ndarray): Input 2D array.
        replace (bool): Whether to sample with replacement. Defaults to False.
        random_state (int or None): Seed for reproducibility.

    Returns:
        np.ndarray: Array with columns independently shuffled.
    """

    if hasattr(np.random, "default_rng"):  # NumPy >= 1.17
        rng = np.random.default_rng(random_state)
    else:  # Older NumPy
        rng = np.random.RandomState(random_state)

    X = np.asarray(X)  # Ensure input is a NumPy array
    X_shuffled = np.empty_like(X)

    for i in range(X.shape[1]):
        X_shuffled[:, i] = rng.choice(X[:, i], size=X.shape[0], replace=replace)

    return X_shuffled

def save_results_as_pickle(results,
                          syn_type,
                           model_type,
                           folder,
                          name_end,
                          timestamp='notimestamp'):
    name = f'results_{timestamp}_{syn_type}_{model_type}_{name_end}'
    save_path = f'{os.path.expanduser("~/Data")}/{folder}/{name}.pkl'
    # save_path = '/Users/25787274/git/hide-and-seek/Data_local'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(save_path)

def find_n_largest_values(arr, n):
    """
    Finds the n largest values in each row of a 2D NumPy array
    and returns a binary array indicating their positions.
    """
    
    arr = np.abs(arr)
    top_idx = np.argsort(arr, axis=1)[:, -n:] #note: no random tiebreaker but shap values are continuous. Perhaps edit in future.
    
    binary_mask = np.zeros_like(arr, dtype=int)
    rows = np.arange(len(binary_mask))[:, None]
    binary_mask[rows, top_idx] = 1
    return binary_mask

def run_feature_selection_models(data_type,
                                folder,
                                run_type,
                                num_important_features=None, #used for all except hide_and_seek and invase
                                full_data_dict=None, #option to provide x_train, y_train, x_test, y_test, g_test
                                model_type='invase',
                                timestamp='notimestamp',
                                batch_size=None,
                                epochs=10000,
                                lmbda=0.3,
                                task='classification',
                                hide_hidden_dim=100, #only used if model_type == 'hide_and_seek'
                                seek_hidden_dim=200, #only used if model_type == 'hide_and_seek',
                                hide_num_hidden_layers=1, #only used if model_type == 'hide_and_seek'
                                seek_num_hidden_layers=1, #only used if model_type == 'hide_and_seek'
                                pickle_results=True,
                                return_results=False,
                                include_model=False,
                                calculate_TPR_FDR_metrics=True,
                                xgb_params = None,
                                use_custom_nn_for_lime=False,
                                train_N = 10_000,
                                test_N = 10_000,
                                seed = 0,
                                include_y_test = True,
                                num_syn_features = 11,
                                batchnorm_hs = False,
                                num_classes = 2,
                                save_experiment_data = False,
                                lmbda_exponent = 2,
                                return_losses_on_val=False,
                                data_mode='synthetic'
                                ):
    """
    'data_mode' can be 'synthetic' or 'credit_data_val' or 'credit_data_test'
    """

    train_seed = seed
    test_seed = seed + 1

    if full_data_dict is not None:
        x_train = full_data_dict['x_train']
        y_train = full_data_dict['y_train']
        x_test = full_data_dict['x_test']
        y_test = full_data_dict['y_test']
        g_test = full_data_dict['g_test']
    else:
        # Data output can be either binary (Y) or Probability (Prob)
        data_out_sets = ['Y','Prob']
        data_out = data_out_sets[0]

        if (model_type == 'l2x') and (train_N != 1_000_000): #needs more data to work
            print('WARNING: SHOULD USE 1_000_000 TRAINING SAMPLES FOR l2x FOR REASONABLE RESULTS')
    
        #%% Data Generation (Train/Test)
        
        x_train, y_train, _, x_test, y_test, g_test = create_data(
                                                data_type=data_type,
                                                data_out=data_out,
                                                train_N=train_N,
                                                test_N=test_N,
                                                train_seed=train_seed,
                                                test_seed=test_seed,
                                                num_features=num_syn_features,
                                                data_mode=data_mode
                                            )
    
    scaled_in_tools = False
    if model_type not in ['lasso']: #have now removed scaler from all models except lasso. scaling happens here.
        print("SCALING IN TOOLS")
        print("SCALING IN TOOLS")
        print("SCALING IN TOOLS")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        scaled_in_tools = True

    baseline_loss = None
    #baseline model - no feature masking
    if ((model_type == 'lime') and (use_custom_nn_for_lime == False)) or (model_type == 'shap'): #no longer using baseline loss for hide_and_seek
        y_train = y_train.argmax(axis=1) if y_train.ndim == 2 else y_train

        from hide_and_seek.model import train_nn
        baseline_model = train_nn(X_train=x_train,
                                y_train= y_train,
                                lmbda=None,
                                n_epochs=2*epochs,
                                seed=train_seed,
                                task=task,
                                hide_hidden_dim=hide_hidden_dim,
                                seek_hidden_dim=seek_hidden_dim,
                                hide_num_hidden_layers=hide_num_hidden_layers,
                                seek_num_hidden_layers=seek_num_hidden_layers,
                                batch_size=None,
                                train_baseline=True,
                                baseline_loss=None,
                                print_description=data_type,
                                batchnorm=batchnorm_hs,
                                num_classes=num_classes,
                                lmbda_exponent=lmbda_exponent
                                )
        baseline_model = baseline_model['model']
        baseline_model = baseline_model.cpu()

    if num_important_features is None:
        #num_important_features is used for all models except hide_and_seek and invase
        num_important_features = np.max(g_test.astype(bool).sum(axis=1)) #max possible number of important features across all instances
        print("num_important_features: ",num_important_features)

    if y_train.ndim == 2:
        num_classes = y_train.shape[1]
    elif y_train.ndim == 1:
        num_classes = int(y_train.max()) + 1
    else:
        raise ValueError("y_train has invalid number of dimensions")

    if model_type == 'invase':
        assert y_train.ndim == 2
        from INVASE_master_ICLR.INVASE import PVS
        import tensorflow as tf
        model = PVS(x_train, data_type,
                    batch_size=batch_size,
                    epochs=epochs,
                    lamda=lmbda)
        
        tf.config.run_functions_eagerly(True) 
        tf.data.experimental.enable_debug_mode() 
        
        # 2. Algorithm training
        model.train(x_train, y_train)

        # 3. Get the selection probability on the testing set
        mask = model.output(x_test)
        binary_mask = 1.*(mask > 0.5)

    elif model_type == 'realx':
        assert y_train.ndim == 2
        from realx_main.realx import REALX #TODO Tal return to original
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.layers import Dense, Input, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras import regularizers
        from tensorflow.keras import backend as K

        realx_lmbda = lmbda
        realx_epochs = epochs
        realx_batch_size = batch_size
        realx_optimizer = Adam(1e-4)
        realx_loss = 'categorical_crossentropy'
        realx_metrics = ['acc', 'AUC']

        input_shape = x_train.shape[1]
        # y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        # y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        
        model_input = Input(shape=(input_shape,), dtype='float32')
        out = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(model_input)
        out = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(out)
        select_prob = Dense(input_shape, kernel_regularizer=regularizers.l2(1e-3))(out)

        
        selector_model = Model(model_input, select_prob)
        model_input = Input(shape=(input_shape,), dtype='float32')
        out= Dense(200, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(model_input)
        out = BatchNormalization()(out)
        out= Dense(200, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(out)
        out = BatchNormalization()(out)
        prob = Dense(num_classes, activation ='softmax', kernel_regularizer=regularizers.l2(1e-3))(out)

        predictor_model = Model(model_input, prob)
        realx = REALX(selector_model, predictor_model, lamda=realx_lmbda)

        realx.predictor.compile(loss=realx_loss,
                        optimizer=realx_optimizer,
                        metrics=realx_metrics)
        realx.predictor.fit(x_train,
                            y_train,
                            epochs=realx_epochs,
                            batch_size=realx_batch_size,
                            verbose=0)
        realx.build_selector()

        # Train
        realx.selector.compile(loss=None,
                            optimizer=realx_optimizer,
                            metrics=realx_metrics)
        realx.selector.fit(x_train,
                        y_train,
                        epochs=realx_epochs,
                        batch_size=realx_batch_size,
                        verbose=0)
        
        #1. Get Selections 
        score = realx.select(x_test, realx_batch_size, True)

        #2. Get Predictions
        y_score = realx.predict(x_test, realx_batch_size)
        # y_pred = y_score.argmax(1)
        # y_test_num = y_test.argmax(1)

        #custom additions
        realx_y_test_pred = y_score
        mask = score.copy()
        binary_mask = score.copy()
        # print(score)

    elif model_type == 'hide_and_seek':
        from hide_and_seek.model import train_nn, pred_nn
        
        y_train = y_train.argmax(axis=1) if y_train.ndim == 2 else y_train

        output = train_nn(X_train=x_train,
                            y_train=y_train,
                            lmbda=lmbda, #note lmbda performs a different role here than in invase 
                            n_epochs=epochs,
                            seed=train_seed,
                            task=task,
                            hide_hidden_dim=hide_hidden_dim,
                            seek_hidden_dim=seek_hidden_dim,
                            hide_num_hidden_layers=hide_num_hidden_layers,
                            seek_num_hidden_layers=seek_num_hidden_layers,
                            batch_size=batch_size,
                            baseline_loss=baseline_loss,
                            print_description=data_type,
                            batchnorm=batchnorm_hs,
                            num_classes=num_classes,
                            lmbda_exponent=lmbda_exponent,
                            return_losses_on_val=return_losses_on_val
                            )
        model = output['model']
        if return_losses_on_val == True:
            losses_on_val = output['losses_on_val']

        hide_and_seek_y_test_pred, mask = pred_nn(model=model,
                                        X_test=x_test,
                                        X_train=x_train,
                                        return_masks=True,
                                        seed=test_seed
                                        )
        binary_mask = 1.*(mask > 0.5)
    
    elif model_type == 'l2x':
        #note - this requires env 'l2x2018'
        from L2X.l2x_for_testing import L2X
        
        data_dict = {'x_train':x_train,
                     'y_train':y_train,
                     'x_val':x_test} #mismatch is ok

        l2x_activation = 'relu' if data_type in ['Syn1','Syn2'] else 'selu'
        print(l2x_activation)
        binary_mask, _, l2x_y_test_pred = L2X(datatype=data_type,
                    num_important_features=num_important_features,
                    train=True,
                    parent_dir=PARENT_DIR_L2X,
                    data_dict=data_dict,
                    activation=l2x_activation, #matches up with INVASE choice
                    return_pred_and_mask=True) 
        mask = binary_mask.copy() # as l2x originally intended. experimentally, this binary option was better for the clustering problem than using the continuous masks
        print(data_type, ': ', num_important_features)

    elif model_type == 'lime':
        from lime import lime_tabular
        import torch
        import pandas as pd
        # from lime code: "As opposed to lime_text.TextExplainer, tabular explainers need a training set. The reason for this is because we compute statistics on each feature (column). If the feature is numerical, we compute the mean and std, and discretize it into quartiles.""
        
        #y_train already made class vector in baseline
        model = lime_tabular.LimeTabularExplainer(x_train, 
                                                   feature_names=None, 
                                                   class_names=None, 
                                                   discretize_continuous=True)
        
        if use_custom_nn_for_lime == True: #True for mnist test - using custom nn classifier as baseline
            from tests_mnist.classifier_for_lime import run_model_lime_nn_classifier
            baseline_model, device = run_model_lime_nn_classifier(x_train, 
                                                          y_train, 
                                                          x_test, 
                                                          y_test,
                                                          epochs=epochs)
        full_data_explanations = {}
        print('starting lime explanations')
        # x_test = x_test[:100,:]
        # for i in [189, 486, 1127, 220, 825, 264]: #mnist images in paper
        for i in range(x_test.shape[0]):
            
            # exp = model.explain_instance(x_test[i], baseline_model.predict_proba, num_features=x_test.shape[1], top_labels=1)
            exp = model.explain_instance(x_test[i], baseline_model.predict_proba, num_features=num_important_features, top_labels=1)
            
            full_data_explanations[i]={int(k):float(v) for k,v in iter(list(exp.as_map().values())[0])}
        
        lime_explanations = pd.DataFrame(full_data_explanations).T
        lime_explanations = lime_explanations.reindex(columns=range(x_test.shape[1]))
        
        binary_mask = lime_explanations.notna().astype(int).values
        mask = binary_mask.copy()

        lime_y_test_pred = baseline_model.predict_proba(x=x_test)

    # elif model_type == 'shap': #shap_xgboost had better results
    #     import shap

    #     background = shap.sample(x_train, 100)

    #     def safe_predict(X):
    #         return baseline_model.predict_proba(X, clip_eps=1e-7)
        
    #     model = shap.KernelExplainer(model=safe_predict, 
    #                                      data=background, 
    #                                      link="logit")

    #     shap_values = model.shap_values(x_test, nsamples=100)[:,:,0] #takes class0 as baseline_model.predict_proba returns probabilities for both classes
        
    #     binary_mask = find_n_largest_values(shap_values, num_important_features)
    #     mask = np.abs(shap_values)

    #     shap_y_test_pred = safe_predict(x_test)
    #     # model.predict(xgb.DMatrix(x_test))
    #     # xgb_y_pred = np.vstack([xgb_y_pred,1-xgb_y_pred]).T

    elif model_type == 'shap_xgboost':
        #note - this requires env 'xgboost'
        #note - not adapted for multiclass
        import xgboost as xgb
        import shap

        # #for syn1-5, 100 trees was better than early stopping. So not doing it. Syn6 difference was negligble 
        # # create a val set for early stopping
        # x_train_new, x_val, y_train_new, y_val = train_test_split(
        #             x_train, y_train[:, 0], test_size=0.1, random_state=train_seed
        #         ) 

        # # Train an XGBoost model
        # dtrain = xgb.DMatrix(x_train_new, label=y_train_new)
        # dval = xgb.DMatrix(x_val, label=y_val)
        # watchlist = [(dtrain, 'train'), (dval, 'val')]
        y_train = y_train.argmax(axis=1) if y_train.ndim == 2 else y_train
        
        dtrain = xgb.DMatrix(x_train, label=y_train)
        watchlist = [(dtrain, 'train')]

        if xgb_params is None:
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 3,
                'eta': 0.1
            }
            num_boost_round = 100
        else:
            num_boost_round = xgb_params['num_boost_round']
            del xgb_params['num_boost_round']
        
        xgb_params['seed']=train_seed
        evals_result = {}
        
        model = xgb.train(
                        xgb_params,
                        dtrain,
                        num_boost_round=num_boost_round,
                        evals=watchlist,
                        evals_result=evals_result,
                        verbose_eval=num_boost_round // 10
                        # early_stopping_rounds=10
                    )

        # Use SHAP to explain the model
        explainer = shap.TreeExplainer(model) #chooses best explainer. for ours, will use TreeExplainer
        shap_values = explainer(x_test).values
        
        # Get the binary_masks based on SHAP values
        binary_mask = find_n_largest_values(shap_values, num_important_features)
        mask = np.abs(shap_values)
        # mask = shap_values # don't use absolute in clustering california experiment

        #predictions for saving
        xgb_y_test_pred = model.predict(xgb.DMatrix(x_test))
        xgb_y_test_pred = np.vstack([xgb_y_test_pred,1-xgb_y_test_pred]).T

    elif model_type == 'lasso':
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty='l1',
                            solver='liblinear',   # supports L1
                            C=1                 # inverse of regularisation strength
                        )
                    )
        y_train = y_train.argmax(axis=1) if y_train.ndim == 2 else y_train

        model.fit(x_train, y_train)
        
        coefs = model.named_steps['logisticregression'].coef_
        
        coefs = np.tile(coefs, (len(x_test),1))
        binary_mask = find_n_largest_values(coefs, num_important_features)
        mask = np.abs(coefs)
        lasso_y_test_pred = model.predict_proba(x_test)

    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(criterion='gini', 
                                    n_estimators=100,
                                    max_depth=5, 
                                    random_state=train_seed)
        model.fit(x_train, y_train)
        
        rf_y_test_pred = model.predict_proba(x_test)[0]
        
        from sklearn.metrics import log_loss
        loss = log_loss(y_test, rf_y_test_pred)
        print(f'loss for {data_type} with random forest: {loss}')

        importance = model.feature_importances_
        
        importance = np.tile(importance, (len(x_test),1))
        binary_mask = find_n_largest_values(importance, num_important_features)
        
        mask = np.abs(importance)

    elif model_type == 'tabnet':
        #tabnet - still experimenting. Not reported in paper.
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch

        model = TabNetClassifier(
                n_d=8, n_a=8,            # sizes of decision/attention steps
                n_steps=3,               # number of sequential steps
                gamma=1.3,               # relaxation parameter
                lambda_sparse=1e-3,      # sparsity regularization
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=1e-2),
                verbose=10
            )
        from sklearn.model_selection import train_test_split
        y_train = np.argmax(y_train, axis=1)
        X_tr, X_vl, y_tr, y_vl = train_test_split(x_train, y_train, test_size=0.1)

        model.fit(X_tr, y_tr,
                    eval_set=[(X_vl, y_vl)],
                    eval_name=['val'],
                    eval_metric=['logloss'],
                    max_epochs=100,
                    patience=10,
                    batch_size=1024,
                    virtual_batch_size=128
                )
        
        tabnet_preds_class = model.predict(x_test)         # predicted classes
        tabnet_preds_proba = model.predict_proba(x_test)   # class probabilities

        mask = model.explain(x_test)[0] #all bigger than zero. not all in [0,1]
        assert (mask >= 0).all()  
        
        binary_mask = 1.*(mask > 0) # might play with this
    

    else:
        raise ValueError("Unsupported model type.")
    
    # 5. Prediction
    if model_type == 'invase':
        y_test_pred, dis_predict = model.get_prediction(x_test, binary_mask)
    elif model_type == 'hide_and_seek':
        y_test_pred = hide_and_seek_y_test_pred.copy()
    elif model_type == 'l2x':
        y_test_pred = l2x_y_test_pred
    elif model_type == 'shap_xgboost':
        y_test_pred = xgb_y_test_pred
    elif model_type == 'shap':
        y_test_pred = shap_y_test_pred
    elif model_type == 'lime':
        y_test_pred = lime_y_test_pred
    elif model_type == 'random_forest':
        y_test_pred = rf_y_test_pred
    elif model_type == 'lasso':
        y_test_pred = lasso_y_test_pred
    elif model_type == 'tabnet':
        y_test_pred = tabnet_preds_proba
    elif model_type == 'realx':
        y_test_pred = realx_y_test_pred
            
    #%% Output
    results = {}
    
    if calculate_TPR_FDR_metrics == True:
        TPR_mean, FDR_mean, TPR_std, FDR_std = performance_metric(binary_mask=binary_mask, 
                                                              g_truth=g_test)
        f1 = compute_f1(binary_mask=binary_mask, g_truth=g_test)

        print(f'{data_type}: ' + 'TPR mean: ' + str(np.round(TPR_mean,1)) + '\%') # + 'TPR std: ' + str(np.round(TPR_std,1)) + '\%, '  
        print(f'{data_type}: ' + 'FDR mean: ' + str(np.round(FDR_mean,1)) + '\%' ) #  + 'FDR std: ' + str(np.round(FDR_std,1)) + '\%, '  
        print(f'{data_type}: ' + 'F1 mean: ' + str(np.round(f1,1)) + '\%')

        results['TPR_mean']=TPR_mean
        results['FDR_mean']=FDR_mean
        results['TPR_std']=TPR_std
        results['FDR_std']=FDR_std
        
        results['f1'] = f1
    
    acc, roc_auc = prediction_metrics(y_true=y_test, 
                                       y_pred_probs=y_test_pred, 
                                      verbose=False)
    
    results['accuracy']=acc
    results['roc_auc']=roc_auc

    results['batch_size'] = batch_size
    results['save_experiment_data'] = save_experiment_data
    
    if save_experiment_data == True:
        results['binary_mask'] = binary_mask
        results['g_test'] = g_test
        results['mask'] = mask
        results['y_test_pred'] = y_test_pred

        if include_y_test == True:
            results['y_test'] = y_test
    else:
        results['binary_mask'] = None
        results['g_test'] = None
        results['mask'] = None
        results['y_test_pred'] = None
        results['y_test'] = None
    

    if include_model == True:
        results['model'] = model

    if model_type == 'invase':
        results['latent_dim1'] = model.latent_dim1
        results['latent_dim2'] = model.latent_dim2    
        results['activation'] = model.activation
        results['input_shape'] = model.input_shape
        results['input_shape0'] = model.input_shape0    

    if model_type == 'hide_and_seek':
        results['hide_hidden_dim'] = model.hide_hidden_dim
        results['seek_hidden_dim'] = model.seek_hidden_dim
        results['hide_num_hidden_layers'] = model.hide_num_hidden_layers
        results['seek_num_hidden_layers'] = model.seek_num_hidden_layers
        if return_losses_on_val == True:
            results['losses_on_val'] = losses_on_val

    results['epochs'] = epochs
    results['lmbda'] = lmbda
    results['seed'] = seed
    results['model_type'] = model_type
    results['num_important_features'] = num_important_features
    results['lmbda_exponent'] = lmbda_exponent
    results['return_losses_on_val'] = return_losses_on_val
    results['batchnorm_hs'] = batchnorm_hs
    results['num_classes'] = num_classes
    results['num_syn_features'] = num_syn_features
    results['train_N'] = train_N
    results['data_mode'] = data_mode
    results['scaled_in_tools'] = scaled_in_tools
    
    results['run_type']=run_type
    results['time_run']=timestamp
    timestamp_end = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results['time_end']=timestamp_end

    if pickle_results == True:
        save_results_as_pickle(results=results,
                            syn_type=data_type,
                            model_type=model_type,
                            folder=folder,
                            name_end=run_type,
                            timestamp=timestamp)
    if return_results == True:
        return results

    # if model_type == 'invase':    
    #     #%% Prediction Results
    #     Predict_Out = np.zeros([20,3,2])    

    #     for i in range(20):
            
    #         # different teat seed
    #         test_seed = i+2
    #         _, _, _, x_test, y_test, _ = create_data(data_type, data_out)  
                    
    #         # 1. Get the selection probability on the testing set
    #         mask = model.output(x_test)
        
    #         # 2. Selected features
    #         binary_mask = 1.*(mask > 0.5)
        
    #         # 3. Prediction
    #         xs = {}
    #         if use_marginal == True:
    #             xs['x_marginals'] = shuffle_numpy_cols(x_test, replace=True, random_state=test_seed)
            
    #         xs['x']= x_test
    #         val_predict, dis_predict = model.get_prediction(xs, binary_mask)
            
    #         # 4. Prediction Results
    #         Predict_Out[i,0,0] = roc_auc_score(y_test[:,1], val_predict[:,1])
    #         Predict_Out[i,1,0] = average_precision_score(y_test[:,1], val_predict[:,1])
    #         Predict_Out[i,2,0] = accuracy_score(y_test[:,1], 1. * (val_predict[:,1]>0.5) )
        
    #         Predict_Out[i,0,1] = roc_auc_score(y_test[:,1], dis_predict[:,1])
    #         Predict_Out[i,1,1] = average_precision_score(y_test[:,1], dis_predict[:,1])
    #         Predict_Out[i,2,1] = accuracy_score(y_test[:,1], 1. * (dis_predict[:,1]>0.5) )
                
    #     # Mean / Var of 20 different testing sets
    #     Output = np.round(np.concatenate((np.mean(Predict_Out,0),np.std(Predict_Out,0)),axis = 1),4) 
    #     results['Output'] = Output
    #     print(Output)
        


    
