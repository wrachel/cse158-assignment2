#%% 

import json
import gzip
import pandas as pd
import numpy as np

#%%
recipes = pd.read_csv('archive/PP_recipes.csv')
users = pd.read_csv('archive/PP_users.csv')
recipes_RAW = pd.read_csv('archive/RAW_recipes.csv')
interactions_RAW = pd.read_csv('archive/RAW_interactions.csv')
train_interaction = pd.read_csv('archive/interactions_train.csv')
valid_interaction = pd.read_csv('archive/interactions_validation.csv')
test_interaction = pd.read_csv('archive/interactions_test.csv')
#%%

import fastFM as fm

