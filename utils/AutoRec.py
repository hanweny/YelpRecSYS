import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter, defaultdict

import tensorflow as tf


class AutoRec:
    def __init__(self):
        self.V = None
        self.W = None
        self.mu = None
        self.b = None
        self.mode = 'auto'

    def RMSE(self, X, y, y_mask=None):
        layer_1 = tf.sigmoid(tf.matmul(X, self.V) + self.mu)
        layer_2 =  tf.matmul(layer_1, self.W) + self.b
        if y_mask is None:
            y_mask = np.where(y==0, 0, 1)
        masked_pred = (layer_2 - y)**2 * y_mask
        return np.sqrt(np.sum(masked_pred) / np.sum(y_mask))
    
    def train(self, 
              train_X, 
              train_y, 
              train_y_mask=None,
              val_X=None, 
              val_y=None, 
              val_y_mask=None,
              hidden_dim=100,
              p_lambda=30, 
              learning_rate=0.005, 
              num_epochs=1500,
              dropout=0.95,
              verbose=True):
   
        num_users = train_X.shape[0]
        num_items = train_y.shape[1]
        input_dim = train_X.shape[1]
        num_extra_attr = input_dim - num_items
        if num_extra_attr > 0:
            self.mode = 'semi'
            
        self.V = tf.Variable(tf.random.normal([input_dim, hidden_dim], stddev=0.01))
        self.W = tf.Variable(tf.random.normal([hidden_dim, num_items], stddev=0.01))
        self.mu = tf.Variable(tf.random.normal([hidden_dim], stddev=0.01))
        self.b = tf.Variable(tf.random.normal([num_items], stddev=0.01))
        
        if train_y_mask is None:
            train_y_mask = np.where(train_y==0, 0, 1)
        if val_X is not None:
            if val_y_mask is None:
                val_y_mask = np.where(val_y==0, 0, 1)
                
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                layer_1 = tf.nn.dropout(tf.sigmoid(tf.matmul(train_X, self.V) + self.mu), dropout)
                layer_2 = tf.matmul(layer_1, self.W) + self.b
                penalty_norm = p_lambda * (tf.square(tf.norm(self.W)) + tf.square(tf.norm(self.V)))
                sse = (layer_2 - train_y)**2
                sse_masked = tf.multiply(sse, train_y_mask)
                loss = tf.reduce_sum(sse_masked) + penalty_norm 
            gradients = tape.gradient(loss, [self.W, self.V, self.mu, self.b])
            optimizer.apply_gradients(zip(gradients, [self.W, self.V, self.mu, self.b])) 
            if verbose and (epoch % (num_epochs // 20) == 0 or epoch < 10):
                train_rmse = self.RMSE(train_X, train_y, train_y_mask)
                msg = "Epoch: {}, loss: {}, train_rmse: {}".format(epoch, float(loss), train_rmse)
                if val_X is not None:
                    val_rmse = self.RMSE(val_X, val_y, val_y_mask)
                    msg += ", val_rmse: {}".format(val_rmse)
                print(msg)
                
    def test(self, test_X, test_y, test_y_mask=None):
        return self.RMSE(test_X, test_y, test_y_mask)
    
    def predict(self, X):
        layer_1 = tf.sigmoid(tf.matmul(X, self.V) + self.mu)
        layer_2 = tf.matmul(layer_1, self.W) + self.b
        return layer_2
    
    def predict_latent(self, X):
        layer_1 = tf.sigmoid(tf.matmul(X, self.V) + self.mu)
        return layer_1