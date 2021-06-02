import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter, defaultdict
from utils.AutoRec import AutoRec
import tensorflow as tf


class DualAutoRec:
    def __init__(self, mode='DUAL'):
        self.mode = mode
        if self.mode not in ['DUAL', 'DUAL-SEP']:
            raise Exception("ValueError: mode must be in one of ['DUAL', 'DUAL-SEP']")
        self.uAuto = AutoRec()
        self.iAuto = AutoRec()
        
    
    def train_user_auto_encoder(self, user_param_ordered_dict):
        print("Training User AutoEncoder...")
        self.uAuto.train(**user_param_ordered_dict)
        print('User AutoEncoder Done...\n\n')
        
    def train_item_auto_encoder(self, item_param_ordered_dict):
        print("Training Item AutoEncoder....")
        self.iAuto.train(**item_param_ordered_dict)
        print('Item AutoEncoder Done...\n\n')
        
    def RMSE(self, user_X, item_X, y, user_y = None, item_y = None):
        user_latent_space = self.uAuto.predict_latent(user_X)
        item_latent_space = self.iAuto.predict_latent(item_X)
        u_MF = tf.matmul(user_latent_space, self.P) + self.bp
        i_MF = tf.matmul(item_latent_space, self.Q) + self.bq
        MF = tf.matmul(u_MF, tf.transpose(i_MF))
        sse_mf = (MF - y)**2
        y_mask = np.where(y==0, 0, 1)
        sse_mf_masked = tf.multiply(sse_mf, y_mask)
        mf_rmse = np.sqrt(np.sum(sse_mf_masked) / np.sum(y_mask))
        user_rmse = self.uAuto.test(user_X, user_y) if user_y is not None else None
        item_rmse = self.iAuto.test(item_X, item_y) if item_y is not None else None
        return mf_rmse, user_rmse, item_rmse
    
    
    def train_dual_sep(self, 
                       train_y,
                       user_param_ordered_dict,
                       item_param_ordered_dict, 
                       val_y = None,
                       hidden_dim=100,
                       p_lambda=30, 
                       learning_rate=0.005, 
                       num_epochs=1500,
                       dropout=0.95,
                       retrain_user = True,
                       retrain_item = True,
                       verbose = True):

        if retrain_user:
            self.train_user_auto_encoder(user_param_ordered_dict)
        if retrain_item:
            self.train_item_auto_encoder(item_param_ordered_dict)
        
        train_user_X, val_user_X = user_param_ordered_dict['train_X'], user_param_ordered_dict['val_X']
        train_user_y, val_user_y =  user_param_ordered_dict['train_y'], user_param_ordered_dict['val_y']
        train_item_X, val_item_X = item_param_ordered_dict['train_X'], item_param_ordered_dict['val_X']
        train_item_y, val_item_y =  item_param_ordered_dict['train_y'], item_param_ordered_dict['val_y']
        
        user_latent_space = self.uAuto.predict_latent(train_user_X)
        item_latent_space = self.iAuto.predict_latent(train_item_X)
        user_latent_dim = user_latent_space.shape[1]
        item_latent_dim = item_latent_space.shape[1]
        
        self.P = tf.Variable(tf.random.normal([user_latent_dim, hidden_dim], stddev=0.01))
        self.bp = tf.Variable(tf.random.normal([hidden_dim], stddev=0.01))
        self.Q = tf.Variable(tf.random.normal([item_latent_dim, hidden_dim], stddev=0.01))
        self.bq = tf.Variable(tf.random.normal([hidden_dim], stddev=0.01))
        
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                u_MF = tf.matmul(user_latent_space, self.P) + self.bp
                i_MF = tf.matmul(item_latent_space, self.Q) + self.bq
                MF = tf.matmul(u_MF, tf.transpose(i_MF))
                penalty_norm = p_lambda * (tf.square(tf.norm(self.Q)) + tf.square(tf.norm(self.P)))
                sse_mf = (MF - train_y)**2
                train_y_mask = np.where(train_y==0, 0, 1)
                sse_mf_masked = tf.multiply(sse_mf, train_y_mask)
                loss = tf.reduce_sum(sse_mf_masked) + penalty_norm
            gradients = tape.gradient(loss, [self.Q, self.P, self.bp, self.bq])
            optimizer.apply_gradients(zip(gradients,  [self.Q, self.P, self.bp, self.bq]))
            
            if verbose and (epoch % (num_epochs // 20) == 0 or epoch < 10):
                train_rmse, _, _ = self.RMSE(train_user_X, train_item_X, train_y)
                msg = "Epoch: {}, loss: {}, train_rmse: {}".format(epoch, float(loss), train_rmse)
                if val_user_X is not None and val_item_X is not None and val_y is not None:
                    val_rmse, _, _= self.RMSE(val_user_X, val_item_X, val_y)
                    msg += ", val_rmse: {}".format(val_rmse)
                print(msg)
                

    def train_dual_one_loss(self, 
                          train_y,
                          user_param_ordered_dict,
                          item_param_ordered_dict, 
                          val_y = None,
                          hidden_dim=100,
                          p_lambda=30, 
                          learning_rate=0.005, 
                          num_epochs=1500,
                          dropout=0.95, 
                          verbose=True):
        train_user_X, val_user_X = user_param_ordered_dict['train_X'], user_param_ordered_dict['val_X']
        train_user_y, val_user_y =  user_param_ordered_dict['train_y'], user_param_ordered_dict['val_y']
        train_item_X, val_item_X = item_param_ordered_dict['train_X'], item_param_ordered_dict['val_X']
        train_item_y, val_item_y =  item_param_ordered_dict['train_y'], item_param_ordered_dict['val_y']
        u_hidden_dim = user_param_ordered_dict['hidden_dim']
        i_hidden_dim = item_param_ordered_dict['hidden_dim']

        ### Weights for User AutoEncoder
        uV = tf.Variable(tf.random.normal([train_user_X.shape[1], u_hidden_dim], stddev=0.01))
        uW = tf.Variable(tf.random.normal([u_hidden_dim, train_user_y.shape[1]], stddev=0.01))
        umu = tf.Variable(tf.random.normal([u_hidden_dim], stddev=0.01))
        ub = tf.Variable(tf.random.normal([train_user_y.shape[1]], stddev=0.01))
        self.uAuto.mode = 'semi' if train_user_X.shape[1] > train_user_y.shape[1] else 'auto'

        ### Weights for Item AutoEncoder
        iV = tf.Variable(tf.random.normal([train_item_X.shape[1], i_hidden_dim], stddev=0.01))
        iW = tf.Variable(tf.random.normal([i_hidden_dim, train_item_y.shape[1]], stddev=0.01))
        imu = tf.Variable(tf.random.normal([i_hidden_dim], stddev=0.01))
        ib = tf.Variable(tf.random.normal([train_item_y.shape[1]], stddev=0.01))
        self.iAuto.mode = 'semi' if train_item_X.shape[1] > train_item_y.shape[1] else 'auto'
        
        ### Weights for joining latent space
        self.P = tf.Variable(tf.random.normal([u_hidden_dim, hidden_dim], stddev=0.01))
        self.bp = tf.Variable(tf.random.normal([hidden_dim], stddev=0.01))
        self.Q = tf.Variable(tf.random.normal([i_hidden_dim, hidden_dim], stddev=0.01))
        self.bq = tf.Variable(tf.random.normal([hidden_dim], stddev=0.01))

        
        train_user_y_mask = np.where(train_user_y==0, 0, 1)
        train_item_y_mask = np.where(train_item_y==0, 0, 1)             
        train_y_mask = np.where(train_y==0, 0, 1)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                u_layer_1 = tf.nn.dropout(tf.sigmoid(tf.matmul(train_user_X, uV) + umu), user_param_ordered_dict['dropout'])
                u_layer_2 = tf.matmul(u_layer_1, uW) + ub
                i_layer_1 = tf.nn.dropout(tf.sigmoid(tf.matmul(train_item_X, iV) + imu), item_param_ordered_dict['dropout'])
                i_layer_2 = tf.matmul(i_layer_1, iW) + ib
                u_MF = tf.matmul(u_layer_1, self.P) + self.bp
                i_MF = tf.matmul(i_layer_1, self.Q) + self.bq
                MF = tf.matmul(u_MF, tf.transpose(i_MF))
                
                sse_u = (u_layer_2 - train_user_y)**2
                sse_i = (i_layer_2 - train_item_y)**2
                sse_mf = (MF - train_y)**2
                
                sse_u_masked = tf.multiply(sse_u, train_user_y_mask)
                sse_i_masked = tf.multiply(sse_i, train_item_y_mask)
                sse_mf_masked = tf.multiply(sse_mf, train_y_mask)
                
                penalty_norm = p_lambda * (tf.square(tf.norm(uW)) + tf.square(tf.norm(uV)) + \
                                           tf.square(tf.norm(iW)) + tf.square(tf.norm(iV)) + \
                                           tf.square(tf.norm(self.Q)) + tf.square(tf.norm(self.P)))

                loss = tf.reduce_sum(sse_i_masked) + tf.reduce_sum(sse_u_masked) + tf.reduce_sum(sse_mf_masked) + penalty_norm
                
            gradients = tape.gradient(loss, [uV, uW, umu, ub, iV, iW, imu, ib, self.Q, self.P, self.bp, self.bq])
            optimizer.apply_gradients(zip(gradients,  [uV, uW, umu, ub, iV, iW, imu, ib, self.Q, self.P, self.bp, self.bq]))
            
            self.uAuto.V = uV
            self.uAuto.W = uW
            self.uAuto.b = ub
            self.uAuto.mu = umu
            self.iAuto.V = iV
            self.iAuto.W = iW
            self.iAuto.b = ib
            self.iAuto.mu = imu
        
            if verbose and (epoch % (num_epochs // 20) == 0 or epoch < 10):
                train_rmse= self.RMSE(train_user_X, train_item_X, train_y, train_user_y, train_item_y)
                msg = "Epoch: {}, loss: {}, train_rmse: {}".format(epoch, float(loss), train_rmse)
                if val_user_X is not None and val_item_X is not None and val_y is not None:
                    val_rmse = self.RMSE(val_user_X, val_item_X, val_y, val_user_y, val_item_y)
                    msg += ", val_rmse: {}".format(val_rmse)
                print(msg)
                
    def train(self, 
               train_y,
               user_param_ordered_dict,
               item_param_ordered_dict, 
               val_y = None,
               hidden_dim=100,
               p_lambda=30, 
               learning_rate=0.005,
               num_epochs=1500,
               dropout=0.95,
               retrain_user = True,
               retrain_item = True,
               verbose = True):
        if self.mode == "DUAL":
            self.train_dual_one_loss(train_y, user_param_ordered_dict, item_param_ordered_dict, val_y, hidden_dim, p_lambda, learning_rate, num_epochs, dropout, verbose)
        if self.mode == 'DUAL-SEP':
            self.train_dual_sep(train_y, user_param_ordered_dict, item_param_ordered_dict, val_y, hidden_dim, p_lambda, learning_rate, num_epochs, dropout, retrain_user, retrain_item, verbose)
            
            
    def test(self, user_X, item_X, y):
        return self.RMSE(user_X, item_X, y)
    
    def predict(self, user_X, item_X):
        user_latent_space = self.uAuto.predict_latent(user_X)
        item_latent_space = self.iAuto.predict_latent(item_X)
        u_MF = tf.matmul(user_latent_space, self.P) + self.bp
        i_MF = tf.matmul(item_latent_space, self.Q) + self.bq
        MF = tf.matmul(u_MF, tf.transpose(i_MF))
        return MF
    
    