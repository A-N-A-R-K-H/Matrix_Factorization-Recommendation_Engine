
"""
Colby Wise
COMS6998 - Homework1
Matrix Factorization

Helper functions to calculate mean squared error (MSE) 
and mean reciprocal rank (MMR) on test data using
the U, V decomposition matrices
"""    

import pandas as pd
import numpy as np
import time
import math
    
"""
Predicts user/movie rating using a user vector and
movie vector: U * V.T
@param:
    u_vec - user vector
    v_vec - movie vector
    rnd - default(False), rounds pred up by 0.5 ie. 2.25 = 2.5
@Return:
    pred - Unrounded (or rounded) predicted rating
"""
def predict_rating(u_vec, v_vec, rnd=False):
    pred = np.dot( u_vec, v_vec )
    return round(pred * 2) / 2 if rnd else pred

"""
Predicts user/movie rating using a user vector and
movie vector: U * V.T
@param:
    test_sample - random sample of test data
    U - factored user matrix
    V - factored movie matrix
    show - default(False), if True prints a few predict values vs true
@Return:
    MSE - mean squared error for test sample
"""
def calc_MSE(test_sample, U, V, show=False):
    preds = []
    loss = 0
    print("Running test validation...")
    cntr = 1
    for row in test_sample.itertuples():
        user, movie, rating = row[2], row[3], row[4]
        pred = predict_rating(U[user,:], V[:,movie], rnd=False)
        preds.append(pred) 
        
        if not (cntr % 10**5) and show:
            print("u: {} \t m: {} \t r: {} \t r_hat: {}".format(user, movie, rating, pred))
            
        err = rating - pred
        loss += err**2
        cntr += 1
    MSE = (loss/len(preds))
    return MSE

"""
Calculates mean reciprocal rank for all predictions greater
than 3.0. First predicts ratings, then finds ratings >= 3.0
and finally ranks predicts per user to calc MRR
@param:
    df - test data
    U - factored user matrix
    V - factored movie matrix
@Return:
    MRR - mean reciprocal rank for all ratings >= 3.0
"""
def calc_MRR(df, U, V):
    start = time.time()
    MRR = []
    preds = []
    for row in df.itertuples():
        user, movie = row[2], row[3]
        u_vec, v_vec = U[user,:], V[:,movie]
        preds.append( predict_rating(u_vec, v_vec, rnd=True) )
        
    df['pred'] = pd.Series(preds)
    _s1 = len(df)
    df = df.loc[ df['pred'] >= 3.0 ]
    _s2 = len(df)
    print("Percent of rows removed given predictions < 3.0: {0:0.2f}%".format((_s1-_s2)//_s1))
    Q_length = len(df)
    df.sort_values(by=['userId','pred'], ascending=False)

    for user in df['userId'].unique():
        user_data = df.loc[ df['userId'] == user ]
        if len(user_data) > 0:
            rankings = user_data.index[ user_data['rating'] == user_data['pred'] ].tolist()
            if rankings:
                rank = rankings[0] + 1 # To account for 0 indexing
                MRR.append( round(1/rank, 3) )  

    MRR = (sum(MRR)/Q_length) - 1 # Remove index adjustment
    print("MRR Calculation Runtime: {} min".format( int((time.time()-start)//60) ))
    print("Model MRR: ", MRR)

    return MRR 

