"""
Colby Wise
COMS6998 - Homework1
Matrix Factorization

This file contains data preprocessing function.
"""

import pandas as pd



"""
Read in CSV data file and remove 'timestamp' column
@param:
    file - file directory
@return:
    df - pandas dataframe
"""
def get_data(file):
    df = pd.read_csv(file)
    df = df.drop('timestamp', axis=1)
    return df

"""
Creates two dictionaries used to retrieve oriinal userIds
from re-indexed Ids
@param:
    data - pandas dataframe
@return:
    key_toUser - dict of new userId as keys, oldIds as values
    user_toKey - dict to get new userId from original Id
"""
def get_user_dicts(data):
    cols = data.columns
    key_toUser = {}
    for idx,user in enumerate(data[cols[1]].unique()):
        key_toUser[idx] = user
    user_toKey = dict((v,k) for k,v in key_toUser.items())
    assert(len(key_toUser) == len(user_toKey))
    print("Number of users: ", len(key_toUser))
    return key_toUser, user_toKey


"""
Creates two dictionaries used to retrieve oriinal movieIds
from re-indexed Ids
@param:
    data - pandas dataframe
@return:
    key_toUser - dict of new movieId as keys, oldIds as values
    user_toKey - dict to get new movieId from original Id
"""
def get_item_dicts(data):
    cols = data.columns
    key_toItem = {}
    for idx,item in enumerate(data[cols[2]].unique()):
        key_toItem[idx] = item
    item_toKey = dict((v,k) for k,v in key_toItem.items())
    assert(len(item_toKey) == len(key_toItem))
    print("Number of items: ", len(key_toItem))
    return key_toItem, item_toKey

"""
Helper function that performs remapping across dataframe
@param:
    user - original userId
@return:
    Returns the new userId
"""
def update_userId(user):
    return user_toKey[user]

"""
Helper function that performs remapping across dataframe
@param:
    movie - original movieId
@return:
    Returns the new movieId
"""
def update_movieId(movie):
    return item_toKey[movie]

