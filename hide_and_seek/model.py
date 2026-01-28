import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("regression setting is still under development")

class TemperatureScaledSigmoid(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature  # Lower T → sharper push to 0/1
    
    def forward(self, x):
        return torch.sigmoid(x / self.temperature)

class net_hide(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 num_hidden_layers=1,
                 batchnorm=False):
        super().__init__()

        layers = []

        #fist layer: input -> hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        if batchnorm == True:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        
        #hidden layers: hidden -> hidden
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batchnorm == True:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer: hidden → input
        layers.append(nn.Linear(hidden_dim, input_dim))
        # layers.append(nn.Sigmoid())
        layers.append(TemperatureScaledSigmoid(temperature=1))  # Optional: temperature scaling for sharper outputs
        self.net = nn.Sequential(*layers)
        
    def forward(self, true_x):
        mask = self.net(true_x)
        return mask
    
class net_seek(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim,
                 num_hidden_layers,
                 lmbda,
                 task='regression',
                 batchnorm=False,
                 num_classes=2
                 ):
        super().__init__()
        self.lmbda = lmbda
        self.last_loss = None #used for baseline model
        layers = []

        #fist layer: input -> hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        if batchnorm == True:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        
        #hidden layers: hidden -> hidden
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batchnorm == True:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer: hidden → input
        if task == 'regression':
            layers.append(nn.Linear(hidden_dim, 1))

        elif task == 'classification':
            layers.append(nn.Linear(hidden_dim, num_classes)) #note - loss functions expect logits, not probabilities

        self.net = nn.Sequential(*layers)
        
    def forward(self, true_x, marginals_x,
                mask):
        
        x = mask * (true_x) + (1 - mask) * marginals_x
        
        y_pred = self.net(x)
        return y_pred
    
    def baseline_forward(self, x):
        
        y_pred = self.net(x)
        return y_pred
    
    def predict_proba(self, x, clip_eps=None):
        """
        Only for binary classification
        method used for LIME and kernelSHAP baseline model for tabular data - no masking

        notes: 
        - neural net needs x to be a tensor, LIME needs it to be numpy
        - LIME (and our implementation of SHAP) expects the output to be probabilities, not logits
        """
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            y_pred = self.baseline_forward(x)
            probs = F.softmax(y_pred, dim=1)
            probs = probs.numpy()

        if clip_eps is not None: #for kernelSHAP
            probs = np.clip(probs, clip_eps, 1 - clip_eps)

        return probs

class hide_and_seek(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hide_hidden_dim,
                 seek_hidden_dim,
                 hide_num_hidden_layers,
                 seek_num_hidden_layers,
                 lmbda,
                 task='regression',
                 batchnorm=False,
                 num_classes=2
                 ):
        super().__init__()
        self.task = task
        self.lmbda = lmbda
        self.hide_hidden_dim = hide_hidden_dim
        self.seek_hidden_dim = seek_hidden_dim
        self.hide_num_hidden_layers = hide_num_hidden_layers
        self.seek_num_hidden_layers = seek_num_hidden_layers
        self.batchnorm = batchnorm
        self.num_classes = num_classes

        self.net_hide = net_hide(input_dim=input_dim,
                                 hidden_dim=hide_hidden_dim,
                                 num_hidden_layers=hide_num_hidden_layers,
                                 batchnorm=batchnorm)
        
        self.net_seek = net_seek(input_dim=input_dim,
                                 hidden_dim=seek_hidden_dim,
                                    num_hidden_layers=seek_num_hidden_layers,
                                 lmbda=lmbda,
                                task=task,
                                batchnorm=batchnorm,
                                num_classes=num_classes
                                 )
        
    def forward(self, true_x, marginals_x):
        mask = self.net_hide(true_x)
        y_pred = self.net_seek(true_x, marginals_x, mask)
        return y_pred, mask
    
# def compute_mean_binary_entropy(probs, eps=1e-10,
#                    ):
#     """
#     Compute entropy of a tensor of probabilities.
    
#     Args:
#         probs (torch.Tensor): Tensor of probabilities (values between 0 and 1).
#         eps (float): Small value to avoid log(0).
    
#     Returns:
#         torch.Tensor: Scalar entropy value.
#     """
#     # Clip probabilities to avoid log(0) and ensure numerical stability
#     clipped_probs = torch.clamp(probs, min=eps, max=1.0 - eps)
    
#     # Compute entropy: -sum(p * log(p))
#     entropy = - clipped_probs * torch.log(clipped_probs) - (1 - clipped_probs) * torch.log(1 - clipped_probs)

#     return entropy.mean()
    
def loss_mse(y_pred, y_true):
    """Calculate mean squared error between predictions and true values."""
    return torch.mean((y_pred - y_true) ** 2) #TODO: check if this is correct for single target vs multitarget

def loss_mse_with_mask_penalty(y_pred, 
                               y_true, 
                               mask, 
                               lmbda, 
                               is_tensor=True):

    if is_tensor == True:
        pred_mse_per_row = torch.mean((y_pred - y_true) ** 2, dim=1)
        mask_mean_size_per_row = mask.mean(dim=1)
    else:
        pred_mse_per_row = np.mean((y_pred - y_true) ** 2) #note difference to above - maybe fix in future edit (single target vs multitarget)
        mask_mean_size_per_row = mask.mean(axis=1)
    
    return (pred_mse_per_row + lmbda * mask_mean_size_per_row).mean()

def cross_entropy(y_pred, 
                    y_true, 
                ):
    
    ce_loss = F.cross_entropy(y_pred, y_true)
    return ce_loss

def custom_cross_entropy(y_pred, 
                         y_true, 
                         mask, 
                         lmbda,
                         epoch=None,
                         n_epochs=None,
                         lmbda_exponent=2,
                         return_separate_losses=False):
    """
    Args:
        y_pred (Tensor): Raw logits, shape (batch_size, num_classes)
        y_true (Tensor): one-hot encoded, shape (batch_size, num_classes)
                         
    Returns:
        Tensor: scalar loss
    """
    ce_loss = F.cross_entropy(y_pred, y_true)
    
    mask_mean_size = mask.mean(dim=1).mean()

    if epoch is not None and n_epochs is not None:
        # Adjust lambda dynamically based on epoch
        lmbda = lmbda * (epoch / n_epochs)**(lmbda_exponent)
    
    # if baseline_loss is not None: #no longer using baseline loss
        # return ce_loss/baseline_loss + lmbda * (mask_mean_size)
    if return_separate_losses == False:
        return ce_loss + lmbda * mask_mean_size
    elif return_separate_losses == True:
        #this is used for analysis of the validation dataset
        return ce_loss, mask_mean_size

def train_nn(X_train,
             y_train,
             lmbda, 
             n_epochs, 
             task='regression',
                hide_hidden_dim=100,
                seek_hidden_dim=200,
                hide_num_hidden_layers=1,
                seek_num_hidden_layers=1,
                batch_size=None,
             seed=42,
             train_baseline=False,
             baseline_loss=None,
             print_description='',
             batchnorm=False,
             num_classes=2,
             lmbda_exponent=2,
             return_losses_on_val=False):
    
    if train_baseline == True:
        print('Training baseline model')
    else:
        print('Training model with feature masking')
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    if task == 'regression':
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    elif task == 'classification':
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    else:
        raise ValueError("Unsupported task type. Use 'regression' or 'classification'.")

    torch.manual_seed(seed)

    input_dim = X_train_tensor.shape[1]
    
    if train_baseline == False:
        model = hide_and_seek(input_dim=input_dim,
                            hide_hidden_dim=hide_hidden_dim,
                            seek_hidden_dim=seek_hidden_dim,
                                hide_num_hidden_layers=hide_num_hidden_layers,
                                seek_num_hidden_layers=seek_num_hidden_layers,
                        lmbda=lmbda,
                        task=task,
                        batchnorm=batchnorm,
                        num_classes=num_classes
                        )
    else:
        model = net_seek(input_dim=input_dim,
                         hidden_dim=seek_hidden_dim,
                         num_hidden_layers=seek_num_hidden_layers,
                         lmbda=None,
                         task=task,
                         batchnorm=batchnorm,
                         num_classes=num_classes)
    
    model = model.to(DEVICE)
    
    # ==== 3. Set up loss and optimizer ====
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==== 4. Train the model ====
    
    if batch_size is None:
        if return_losses_on_val == True:
            n_val = int(0.1 * X_train_tensor.size(0))
            X_val_tensor = X_train_tensor[:n_val].clone()
            y_val_tensor = y_train_tensor[:n_val].clone()

            # Use the remaining 90% for actual training
            X_train_tensor_actual = X_train_tensor[n_val:].clone()
            y_train_tensor_actual = y_train_tensor[n_val:].clone()


            X_train_tensor_actual = X_train_tensor_actual.to(DEVICE)
            y_train_tensor_actual = y_train_tensor_actual.to(DEVICE)
            X_val_tensor = X_val_tensor.to(DEVICE)
            y_val_tensor = y_val_tensor.to(DEVICE)

            losses_on_val = {}
        else:
            X_train_tensor_actual = X_train_tensor.clone()
            y_train_tensor_actual = y_train_tensor.clone()
            X_train_tensor_actual = X_train_tensor_actual.to(DEVICE)
            y_train_tensor_actual = y_train_tensor_actual.to(DEVICE)
            
        for epoch in range(n_epochs):
            model.train()

            optimizer.zero_grad()
            if train_baseline == True:              
                
                y_pred_train = model.baseline_forward(x=X_train_tensor_actual)
                
                if task == 'regression':
                    
                        loss = loss_mse(y_pred=y_pred_train, 
                                        y_true=y_train_tensor_actual 
                                        )
                elif task == 'classification':
                    loss = cross_entropy(y_pred=y_pred_train, 
                                        y_true=y_train_tensor_actual, 
                                      )
                else:
                    raise ValueError("Unsupported task type. Use 'regression' or 'classification'.")
            
            else:
                X_train_tensor_shuffled = shuffle_tensor_cols(X_train_tensor, replace=True, random_state=seed+epoch)
                

                if return_losses_on_val == True:
                    X_val_tensor_shuffled = X_train_tensor_shuffled[:n_val].clone()
                    X_val_tensor_shuffled = X_val_tensor_shuffled.to(DEVICE)
                    X_train_tensor_shuffled_actual = X_train_tensor_shuffled[n_val:].clone()
                else:
                    X_train_tensor_shuffled_actual = X_train_tensor_shuffled.clone()
                
                X_train_tensor_shuffled_actual = X_train_tensor_shuffled_actual.to(DEVICE)

                y_pred_train, mask_train = model(true_x=X_train_tensor_actual, 
                                            marginals_x=X_train_tensor_shuffled_actual)
                
                if task == 'regression':
                    
                        loss = loss_mse_with_mask_penalty(y_pred=y_pred_train, 
                                                    y_true=y_train_tensor, 
                                                    mask=mask_train,
                                                    lmbda=lmbda
                                                    ) #TODO: add baseline_loss here
                elif task == 'classification': #this is what was used in the experiments
                    loss = custom_cross_entropy(y_pred=y_pred_train, 
                                                y_true=y_train_tensor_actual, 
                                                mask=mask_train, 
                                                lmbda=lmbda,
                                                epoch=epoch,
                                                n_epochs=n_epochs,
                                                lmbda_exponent=lmbda_exponent
                                                )
                else:
                    raise ValueError("Unsupported task type. Use 'regression' or 'classification'.")
            
            loss.backward()
            optimizer.step()

            model.eval()
            if return_losses_on_val == True:
                with torch.no_grad():
                    y_pred_val, mask_val = model(true_x=X_val_tensor,
                                                marginals_x=X_val_tensor_shuffled)
                    val_ce_loss, val_mask_mean_size = custom_cross_entropy(y_pred=y_pred_val,
                                                    y_true=y_val_tensor,
                                                    mask=mask_val,
                                                    lmbda=lmbda,
                                                    epoch=epoch,
                                                    n_epochs=n_epochs,
                                                    lmbda_exponent=lmbda_exponent,
                                                    return_separate_losses=True
                                                    )
                    epoch_losses_on_val = {}
                    epoch_losses_on_val['val_ce_loss'] = val_ce_loss.item()
                    epoch_losses_on_val['val_mask_mean_size'] = val_mask_mean_size.item()
                    epoch_losses_on_val['lmbda'] = lmbda
                    epoch_losses_on_val['lmbda_exponent'] = lmbda_exponent

                    losses_on_val[epoch] = epoch_losses_on_val
            
            # Step scheduler
            # scheduler.step()

            if (epoch % (n_epochs/5) == 0) or (epoch == n_epochs - 1):
                if return_losses_on_val == True:
                    print(f"""{print_description} 
                        Epoch: {epoch},
                        Loss: {loss.item():.4f}, 
                        val_ce_loss: {val_ce_loss.item():.4f}, 
                        val_mask_mean_size: {val_mask_mean_size.item():.4f}
                        """)
                else:
                    print(f"{print_description} | Epoch: {epoch} | Loss: {loss.item():.4f}")
            
    else: #batching - not used in experiments. Could be improved.

        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            
            if train_baseline == True: #baseline model - not used in experiments
                #might need to update this after having changed batch shuffle set up above and below
                for X_batch, idxs_batch, y_batch in dataloader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)

                    optimizer.zero_grad()

                    y_pred_batch = model.baseline_forward(x=X_batch)

                    if task == 'regression':
                        loss = loss_mse(y_pred=y_pred_batch,
                                        y_true=y_batch
                                        )
                    elif task == 'classification':
                        loss = cross_entropy(y_pred=y_pred_batch,
                                            y_true=y_batch
                                            )
                    else:
                        raise ValueError("Unsupported task type. Use 'regression' or 'classification'.")

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            else: #main model - with feature masking
                X_train_tensor_shuffled = shuffle_tensor_cols(X_train_tensor, replace=True, random_state=seed+epoch)
                dataset = TensorDataset(X_train_tensor, X_train_tensor_shuffled, y_train_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
                for X_batch, X_batch_shuffled, y_batch in dataloader: #batching gave worse results. Perhaps this can be improved.

                    X_batch = X_batch.to(DEVICE)
                    X_batch_shuffled = X_batch_shuffled.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)

                    optimizer.zero_grad()

                    y_pred_batch, mask_batch = model(true_x=X_batch, 
                                                    marginals_x=X_batch_shuffled)

                    if task == 'regression':
                        loss = loss_mse_with_mask_penalty(y_pred=y_pred_batch,
                                                        y_true=y_batch,
                                                        mask=mask_batch,
                                                        lmbda=lmbda)
                    elif task == 'classification':
                        loss = custom_cross_entropy(y_pred=y_pred_batch,
                                                    y_true=y_batch,
                                                    mask=mask_batch,
                                                    lmbda=lmbda,
                                                    epoch=epoch,
                                                    n_epochs=n_epochs,
                                                    lmbda_exponent=lmbda_exponent)
                    else:
                        raise ValueError("Unsupported task type. Use 'regression' or 'classification'.")
                
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            if (epoch % (n_epochs // 5) == 0) or (epoch == n_epochs - 1):
                avg_loss = total_loss / len(dataloader)
                print(f"{print_description} Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
            
    output = {}
    output['model'] = model
    if return_losses_on_val == True:
        output['losses_on_val'] = losses_on_val
    return output

def pred_nn(model,
            X_test,
            X_train,
            return_masks=True,
            seed=42
           ):

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    #marginals come from training data. first break dependencies, then draw len(X_test) samples 
    X_train_tensor_shuffled = shuffle_tensor_cols(X_train_tensor, random_state=seed)
    torch.manual_seed(seed)
    indices = torch.randint(low=0, high=len(X_train_tensor_shuffled), 
                            size=(len(X_test_tensor),), dtype=torch.long) #this happens with replacement
    X_train_tensor_shuffled = X_train_tensor_shuffled[indices] 

    X_train_tensor_shuffled = X_train_tensor_shuffled.to(DEVICE)
    X_test_tensor = X_test_tensor.to(DEVICE)
    
    # ==== 5. Evaluate on test data ====
    model.eval()
    with torch.no_grad():
        logit, mask_test = model(true_x=X_test_tensor, 
                                      marginals_x=X_train_tensor_shuffled)
        logit = logit.cpu()
        
        y_pred_test = torch.softmax(logit, dim=1).numpy() #probabilities

        if return_masks == True:
            return  y_pred_test, mask_test.cpu().detach().numpy()
        else:
            return  y_pred_test            

def shuffle_tensor_cols(X_tensor, replace=False, random_state=None): #I've checked this
    """
    Shuffle each column of a tensor independently.
    
    Args:
        X_tensor (torch.Tensor): Input tensor of shape (n_samples, n_features)
        replace (bool): Whether to sample with replacement
        random_state (int): Seed for reproducibility
        
    Returns:
        torch.Tensor: Tensor with each column shuffled independently
    """

    if random_state is not None:
        torch.manual_seed(random_state)

    # Create a copy to avoid modifying the original tensor
    shuffled = X_tensor.clone()
    
    # Get tensor dimensions
    n_samples, n_features = X_tensor.shape

    # Shuffle each column independently
    for col in range(n_features):
        if replace:
            # Sample with replacement
            indices = torch.randint(0, n_samples, (n_samples,))
        else:
            # Sample without replacement (permutation)
            indices = torch.randperm(n_samples)
        
        shuffled[:, col] = X_tensor[indices, col]
    
    return shuffled

def mse_y(y_pred,
          y_true,
                ):
    """y_true and y_pred are arrays with n x 1 elements
    This finds the mse element-wise across all elements"""

    sq_error = (y_true-y_pred)**2
    return sq_error.mean()
    
# def report_metrics(y_pred, 
#                    y_true,
#                   mask_pred=None,
#                   mask_true=None,
#                   lmbda=None,
#                   is_tensor=False):

#     y_mse = mse_y(y_pred,
#                         y_true
#                        )
    
#     print(f"y mse: {y_mse.item():.4f}")
        
#     if (mask_pred is not None) and (mask_true is not None):
#         mask_mse = mse_mask(mask_pred,
#                               mask_true
#                              )

#         print(f"mask mse: {mask_mse.item():.4f}")
        
#         if  (lmbda is not None):
#             loss = loss_mse_with_mask_penalty(y_pred=y_pred, 
#                                            y_true=y_true, 
#                                            mask=mask_pred,
#                                            lmbda=lmbda,
#                                               is_tensor=is_tensor)
#             print(f"combined loss: {loss.item():.4f}")