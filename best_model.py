
"""
Colby Wise
COMS6998 - Homework1
Matrix Factorization

Factorizers a N x M user:item matrix into
U: N x r and V: r X M matrices where:

- U - User : feature matrix
- V - Movie: feature matrix

Calculates mean squared error (MSE) and
mean reciprocal rank (MMR) on test data
after the training data decomposition
"""


import pandas as pd
import numpy as np
import time
import pickle
import math

from preprocessing import *
from model_evaluation import *



"""
Class MatrixFactorize takes train and test data then
factors the training data in U,V matrix decompositions.
Using U,V it then predicts movie ratings on the test
data returning the MSE and MRR on test data
"""
class MatrixFactorize(object):

    """
    Initialization
    @param:
        train - pandas dataframe of training data
        test - pandas dataframe of test data
        lr  - learning rate
        r - features to learn
        epoch - epochs to run training 
        lambd - regularization rate (lambda)

    """
    def __init__(self, train, test, lr, r, iters, lambd):
        self.lr = lr
        self.features = r
        self.iters = iters
        self.lambd = lambd
        self.train = train
        self.test = test
        
        all_data = pd.concat([train, test])
        self.n_items = len(all_data['movieId'].unique())
        self.n_users = len(all_data['userId'].unique())
        print("Number of users:", self.n_users)
        print("Number of movies:", self.n_items)  
        
        self.U = np.random.randn(self.n_users, self.features)
        self.V = np.random.randn(self.features, self.n_items)
        
        self.loss_record = {"train": [], "test": [], "epoch": [],
                            "r": self.features,"lr" : self.lr}       
    
    """
    Updates loss dictionary that captures training/test
    MSE during training 
    @param:
        test_mse - MSE from test data
        train_mse - MSE from train data
        epoch - current epoch of training
    @Return:
        None
    """
    def record_loss(self, test_mse, train_mse, epoch):
        self.loss_record["train"].append(train_mse)
        self.loss_record["test"].append(test_mse)
        self.loss_record["epoch"].append(epoch)
    
    """
    Predict rating using current U, V matrices
    @param:
        test_sample - random sample from test data
        U - User matrix
        V - Movie matrix
        show - default(False), prints subset of predict values
    @Return:
        mse - mean squared error for test sample
    """   
    def predict(self, test_sample, U, V, show=False):
        preds = []
        loss = 0
        print("Running test validation...")
        cntr = 1
        for row in test_sample.itertuples():
            user, movie, rating = row[2], row[3], row[4]
            pred = np.dot(U[user,:], np.transpose(V[:,movie]) )
            preds.append(pred) 
            
            if not (cntr % 10**5) and show:
                print("u: {} \t m: {} \t r: {} \t r_hat: {}".format(user, movie, rating, pred))
                
            err = rating - pred
            loss += err**2
            cntr += 1
        MSE = (loss/len(preds))
        return MSE
    
    """
    Punny name ... saves pickle objects during training phase
    @param:
        out - data structure (dict, etc) to save
        fname - filename
    @Return:
        None
    """   
    def god_save_the_queen(self, out, fname):
        with open(fname, "wb") as f:
            pickle.dump(out, f)

    
    """
    Uses train and test data to learn U,V matrix decomposition
    """     
    def factorizeMatrix(self):
        print("Factorizing...")
        print("=> learning rate: {}, epochs: {}, r: {}".format(self.lr, self.iters, self.features))
              
        epoch_start = time.time()
        for epoch in range(1,self.iters+1):
            print("\nStarting iteration {}...".format(epoch))
            self.lr = self.lr * .995 # Hacky annealing
            
            best_test_MSE = 50
            cntr = 1
            _s = time.time()
            for row in self.train.itertuples():
                user, movie, rating = row[2], row[3], row[4]
                err = rating - np.dot( self.U[user,:], np.transpose(self.V[:,movie]) )
                train_MSE = err ** 2
                dV = self.lr * (err * 2 * self.U[user,:] - self.lambd * self.V[:,movie])
                dU = self.lr * (err * 2 * self.V[:,movie] - self.lambd * self.U[user,:])
                self.V[:,movie] = self.V[:,movie] + dV
                self.U[user,:] = self.U[user,:] + dU
                cntr += 1
                                
                # Periodically Print Progress 
                if not (cntr % 10**5):
                    _e = time.time()
                    print( "\n {} min runtime to process {:,} rows...\n".format(int((_e-_s)//60), cntr) )
                    test_sample = self.test.sample(frac=0.05)
                    U, V = self.U, self.V
                    test_MSE = calc_MSE(test_sample, U, V, show=False) 
                    self.record_loss(test_MSE, train_MSE, epoch)
                    print( "Train MSE: {0:0.4f}, Test MSE: {1:0.4f}".format(train_MSE, test_MSE))
                    
                    # Periodically Check Test MSE
                    if test_MSE <= best_test_MSE:
                        best_test_MSE = test_MSE
                        u_outfile = "U_mat:_r={}_lambda={}_epoch={}.pkl".format(self.features, self.lambd, epoch)
                        v_outfile = "V_mat:_r={}_lambda={}_epoch={}.pkl".format(self.features, self.lambd, epoch)
                        loss_file = "loss:{:.3f}_r={}_lambda={}_epoch={}.pkl".format(test_MSE, self.features, self.lambd, epoch)
                        self.god_save_the_queen(self.U, u_outfile)
                        self.god_save_the_queen(self.V, v_outfile)
                        self.god_save_the_queen(self.loss_record, loss_file)
            # Track epoch runtime
            epoch_end = time.time()
            print("\n Epoch {} runtime: {} min".format(epoch, int((epoch_end-epoch_start)//60)))

        MRR = calc_MRR(self.test, self.U, self.V)
        with open('MRR.txt', 'w') as f:
            f.write("log:_MRR={:.3f}:_r={}_lambda={}".format(MRR, self.features, self.lambd))

        return self.U, self.V, self.loss_record


if __name__ == "__main__":

    def update_movieId(movie):
        return item_toKey[movie]

    def update_userId(user):
        return user_toKey[user]

    # Helper method to view class properties
    def properties(cls):   
        return [i for i in cls.__dict__.keys() if i[:1] != '_']


    train_file = 'ml-20m/train.csv'
    test_file = 'ml-20m/test.csv'

    train = get_data(train_file)
    test = get_data(test_file)
    all_data = pd.concat([train, test])
    key_toUser, user_toKey = get_user_dicts(all_data)
    key_toItem, item_toKey = get_item_dicts(all_data)

    train['userId'] = train['userId'].apply(update_userId)
    train['movieId'] = train['movieId'].apply(update_movieId)
    test['userId'] = test['userId'].apply(update_userId)
    test['movieId'] = test['movieId'].apply(update_movieId)

    lr = .01
    r = 40
    epochs = 3
    lamda = .02

    MF = MatrixFactorize(train, test, lr, r, epochs, lamda)
    U, V, loss_record = MF.factorizeMatrix()

    #print( properties(MF) )

