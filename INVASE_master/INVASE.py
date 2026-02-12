'''
Personalized Variable Selection Code (PVS)
for ICLR 2019 Conference
'''
#Tal - using this one for experiments
#%% Necessary packages
# 1. Keras
from keras.layers import Input, Dense, Multiply
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

# 2. Others
import tensorflow as tf
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hide_and_seek.tools import run_feature_selection_models

#@TAL additions
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm


#%% Define PVS class
class PVS():
    
    # 1. Initialization
    '''
    x_train: training samples
    data_type: Syn1 to Syn 6
    '''
    def __init__(self, x_train, data_type,
                 batch_size,
                 epochs,
                 lamda):
        self.latent_dim1 = 100      # Dimension of actor (generator) network
        self.latent_dim2 = 200      # Dimension of critic (discriminator) network
        
        self.batch_size = batch_size      # Batch size
        self.epochs = epochs         # Epoch size (large epoch is needed due to the policy gradient framework)
        self.lamda = lamda            # Hyper-parameter for the number of selected features 

        self.input_shape = x_train.shape[1]     # Input dimension
        self.input_shape0 = x_train.shape[0]   # Number of training samples
        
        # Actionvation. (For Syn1 and 2, relu, others, selu)
        self.activation = 'relu' if data_type in ['Syn1','Syn2'] else 'selu'       

        # Use Adam optimizer with learning rate = 0.0001
        optimizer1 = Adam(0.0001) #@TAL updated - 3 optimizers
        optimizer2 = Adam(0.0001) #@TAL updated - 3 optimizers
        optimizer3 = Adam(0.0001) #@TAL updated - 3 optimizers
        
        # Build and compile the discriminator (critic)
        self.discriminator = self.build_discriminator()
        
        # Use categorical cross entropy as the loss
        self.discriminator.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['acc'])
        
        # Build the generator (actor)
        self.generator = self.build_generator()
        # Use custom loss (my loss)
        
        self.generator.compile(loss=self.my_loss, optimizer=optimizer2)
        
        # Build and compile the value function
        self.valfunction = self.build_valfunction()
        
        # Use categorical cross entropy as the loss
        self.valfunction.compile(loss='categorical_crossentropy', optimizer=optimizer3, metrics=['acc'])
        

    #%% Custom loss definition
    def my_loss(self, y_true, y_pred):
        
        # dimension of the features
        d = y_pred.shape[1]        
        
        # Put all three in y_true 
        # 1. selected probability
        sel_prob = y_true[:,:d] #in original code, values are 0 or 1
        # 2. discriminator output
        dis_prob = y_true[:,d:(d+2)]
        # 3. valfunction output
        val_prob = y_true[:,(d+2):(d+4)]
        # 4. ground truth
        y_final = y_true[:,(d+4):]        
        
        # A1. Compute the rewards of the actor network
        Reward1 = tf.reduce_sum(y_final * tf.math.log(dis_prob + 1e-8), axis = 1)  
        
        # A2. Compute the rewards of the actor network
        Reward2 = tf.reduce_sum(y_final * tf.math.log(val_prob + 1e-8), axis = 1)  

        # Difference is the rewards
        Reward = Reward1 - Reward2

        # B. Policy gradient loss computation. 
        loss1 = Reward * tf.reduce_sum( sel_prob * tf.math.log(y_pred + 1e-8) + (1-sel_prob) * tf.math.log(1-y_pred + 1e-8), axis = 1) - self.lamda * tf.reduce_mean(y_pred, axis = 1)
        
        # C. Maximize the loss1
        loss = tf.reduce_mean(-loss1)

        return loss

    #%% Generator (Actor)
    def build_generator(self):

        model = Sequential()
        
        model.add(Dense(self.latent_dim1, activation=self.activation, name = 'sdense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape))
        model.add(Dense(self.latent_dim1, activation=self.activation, name = 'sdense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(Dense(self.input_shape, activation = 'sigmoid', name = 'sdense3', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()

        feature = Input(shape=(self.input_shape,), dtype='float32')
        select_prob = model(feature)

        return Model(feature, select_prob)

    #%% Discriminator (Critic)
    def build_discriminator(self):

        model = Sequential()
        
        model.add(Dense(self.latent_dim2, activation=self.activation, name = 'dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape)) 
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(self.latent_dim2, activation=self.activation, name = 'dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        model.add(Dense(2, activation ='softmax', name = 'dense3', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()
        
        # There are two inputs to be used in the discriminator
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')
        # 2. Selected Features
        sel_prob = Input(shape=(self.input_shape,), dtype='float32')
        
        # Element-wise multiplication
        model_input = Multiply()([feature, sel_prob])
        prob = model(model_input)
        return Model([feature, sel_prob], prob)

    #%% Value Function
    def build_valfunction(self):

        model = Sequential()
                
        model.add(Dense(200, activation=self.activation, name = 'vdense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape)) 
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(200, activation=self.activation, name = 'vdense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        model.add(Dense(2, activation ='softmax', name = 'vdense3', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()
        
        # There are one inputs to be used in the value function
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')       
        
        # Element-wise multiplication
        prob = model(feature)

        return Model(feature, prob)

    #%% Sampling the features based on the output of the generator
    def Sample_M(self, gen_prob):
        
        # Shape of the selection probability
        n = gen_prob.shape[0]
        d = gen_prob.shape[1]
                
        # Sampling
        samples = np.random.binomial(1, gen_prob, (n,d))
        
        return samples

    #%% Training procedure
    def train(self, x_train, y_train):
        # For each epoch (actually iterations)
        for epoch in range(self.epochs):
            #%% Train Discriminator
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size) #note this samples with replacement

            x_batch = x_train[idx,:]
            y_batch = y_train[idx,:]

            # Generate a batch of probabilities of feature selection
            gen_prob = self.generator.predict(x_batch)
            
            # Sampling the features based on the generated probability
            sel_prob = self.Sample_M(gen_prob)
            
            # Compute the prediction of the critic based on the sampled features (used for generator training)
            dis_prob = self.discriminator.predict([x_batch, sel_prob])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch([x_batch, sel_prob], y_batch)
            #%% Train Valud function

            # Compute the prediction of the critic based on the sampled features (used for generator training) #Tal - I don't think this comment is true
            val_prob = self.valfunction.predict(x_batch)

            # Train the discriminator #Tal - I don't think this comment is true
            v_loss = self.valfunction.train_on_batch(x_batch, y_batch)
            
            #%% Train Generator
            # Use three things as the y_true: sel_prob, dis_prob, and ground truth (y_batch)
            y_batch_final = np.concatenate( (sel_prob, np.asarray(dis_prob), np.asarray(val_prob), y_batch), axis = 1 )

            # Train the generator
            g_loss = self.generator.train_on_batch(x_batch, y_batch_final)

            #%% Plot the progress
            dialog = 'Epoch: ' + str(epoch) + ', d_loss (Acc)): ' + str(d_loss[1]) + ', v_loss (Acc): ' + str(v_loss[1]) + ', g_loss: ' + str(np.round(g_loss,4))

            if epoch % 100 == 0:
                print(dialog)
    
    #%% Selected Features        
    def output(self, x_train):
        
        gen_prob = self.generator.predict(x_train)
        
        return np.asarray(gen_prob)
     
    #%% Prediction Results 
    def get_prediction(self, x_train, m_train):

        val_prediction = self.valfunction.predict(x_train)

        dis_prediction = self.discriminator.predict([x_train, m_train])

        return np.asarray(val_prediction), np.asarray(dis_prediction)


#%% Main Function
if __name__ == '__main__':
    
    data_sets = ['Syn1','Syn2','Syn3','Syn4','Syn5','Syn6']
    # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # use_marginal = False
    folder = 'invase_master_iclr_HS_runs/exploring_TPR_FDR'
    model_type = "invase"
    batch_size = 1000 #1000
    # epochs = 10000 #10000
    # lamda = 0.1 #0.1
    # run_type = "testing use_marginal2"

    print("I'm running in ICLR_HS ye")
    # print(run_type)
    
    for use_marginal in [False,True]:
        for lamda in [0.1, 0.01]:
            for epochs in [5000, 10000]:
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                run_type=f'{str(use_marginal)}_{lamda}_{epochs}'
                print(run_type)
                Parallel(n_jobs=6)(
                delayed(run_feature_selection_models)(data_type=data_set, 
                                        folder=folder,
                                        run_type=run_type,
                                        model_type=model_type,
                                        timestamp=timestamp,
                                        use_marginal=use_marginal,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        lamda=lamda)
                for data_set in tqdm(data_sets)
                )