#%% 

import json
import gzip
import pandas as pd
import numpy as np

#%%
import fastFM as fm

#%%
file_path = ""
recipes_RAW = pd.read_csv(file_path + 'data/RAW_recipes.csv')
interactions_RAW = pd.read_csv(file_path + 'data/RAW_interactions.csv')
train_interaction = pd.read_csv(file_path + 'data/interactions_train.csv')
valid_interaction = pd.read_csv(file_path + 'data/interactions_validation.csv')
test_interaction = pd.read_csv(file_path + 'data/interactions_test.csv')
#%%
# (NOT GOOD) encode the name --> can compare similarities between the names (NLP)
# (GOOD) encode ingredients --> can get similarities between ingredients (jaccard?)
# (GOOD) n_steps --> difficulty level
# (GOOD) maybe contributor id?? so if user likes recipes of one user they will like other recipes from the same user 
# (GOOD) n_ingredients

# Options for the model:
# NLP - on the name, ingredeints 
# Some kind of recommender / similarity model (Latent factor, factorization) to predict

#%%
recipes_RAW = recipes_RAW.rename(columns={'id': 'recipe_id'})
all_interactions = pd.merge(interactions_RAW, recipes_RAW, how='left', on='recipe_id')

#%%
cleaned_interactions = all_interactions.drop(columns=['date', ''])
#%%

PP_users = pd.read_csv(file_path + '/data/PP_users.csv')
PP_recipes = pd.read_csv(file_path + '/data/PP_recipes.csv')


#%%

n_interactions = all_interactions.drop(columns=['date', 'review', 'name', 'submitted', 'tags', 'nutrition', 'steps', 'description', 'ingredients'])

#%%
import matplotlib.pyplot as plt 
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.histplot(all_interactions['rating'], bins=5, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of User Ratings', fontsize=16)
plt.xlabel('Ratings', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#%%
rating_counts = all_interactions['rating'].value_counts().reindex(range(6), fill_value=0)

# Plotting the bar graph of ratings count
plt.figure(figsize=(8, 6))
rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Count of User Ratings', fontsize=16)
plt.xlabel('Ratings', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(range(6), labels=range(6))  # Ensure x-axis is labeled 0 to 5
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%

# plot the time distribution of the recipes 
recipes_RAW['submitted'] = pd.to_datetime(recipes_RAW['submitted'])
frequency_by_year = recipes_RAW.groupby(recipes_RAW['submitted'].dt.year).size()

plt.figure(figsize=(8, 6))
frequency_by_year.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frequency of Recipes by Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Recipes', fontsize=14)
plt.xticks(rotation=30)  # Keep x-axis labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
# plot the time distribution of the reviews 
interactions_RAW['date'] = pd.to_datetime(interactions_RAW['date'])
frequency_by_year = interactions_RAW.groupby(interactions_RAW['date'].dt.year).size()

plt.figure(figsize=(8, 6))
frequency_by_year.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frequency of Reviews by Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)
plt.xticks(rotation=30)  # Keep x-axis labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
reviews_per_user = all_interactions['user_id'].value_counts()

# Define custom bins (larger bins for larger review counts)
bins = [0, 2, 5, 10, 20, 50, 100, 200, 500, 1000, reviews_per_user.max()]
reviews_per_user_binned = pd.cut(reviews_per_user, bins=bins)

# Plot the distribution of how many reviews a user leaves (grouped by larger bins)
plt.figure(figsize=(10, 6))
reviews_per_user_binned.value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Reviews per User', fontsize=16)
plt.xlabel('Number of Reviews', fontsize=14)
plt.ylabel('Number of Users', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#%%
# distribution of minutes
# number of ingredients
df = recipes_RAW
values = recipes_RAW['minutes'].value_counts()
# Define custom bins (larger bins for larger review counts)
custom_bins = [0, 10, 30, 60, 100, 200, 300, 500, values.max()]  # Added an edge for the last bin
bin_labels = ['0-10', '10-30', '30-60', '60-100', '100-200', '200-300', '300-500', '500+']  
df['minutes_bin'] = pd.cut(df['minutes'], bins=custom_bins, labels=bin_labels, right=False)
bin_counts = df['minutes_bin'].value_counts().sort_index()

# Plotting the histogram with custom bins
plt.figure(figsize=(10, 6))
plt.bar(bin_counts.index, bin_counts.values, color='skyblue', edgecolor='black')
plt.title('Distribution of minutes per recipe', fontsize=16)
plt.xlabel('Minutes', fontsize=14)
plt.ylabel('Number of Recipes', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
# number of steps
df = recipes_RAW
values = recipes_RAW['n_steps'].value_counts()
# Define custom bins (larger bins for larger review counts)
custom_bins = [0, 4, 8, 12, 16, 20, 24, 28, 32, 50, values.max()]  # Added an edge for the last bin
bin_labels = ['0-4','4-8', '8-12', '12-16', '16-20', '20-24', '24-28', '28-32', '32-50', '50+']  
df['steps_bin'] = pd.cut(df['n_steps'], bins=custom_bins, labels=bin_labels, right=False)
bin_counts = df['steps_bin'].value_counts().sort_index()

# Plotting the histogram with custom bins
plt.figure(figsize=(10, 6))
plt.bar(bin_counts.index, bin_counts.values, color='skyblue', edgecolor='black')
plt.title('Distribution of steps per recipe', fontsize=16)
plt.xlabel('Number of Steps', fontsize=14)
plt.ylabel('Number of Recipes', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#%%

# number of ingredients
df = recipes_RAW
ingredients = recipes_RAW['n_ingredients'].value_counts()

# Define custom bins (larger bins for larger review counts)
custom_bins = [0, 5, 7, 10, 15, 20, 30, ingredients.max()]  # Added an edge for the last bin
bin_labels = ['0-5', '5-7', '7-10', '10-15', '15-20', '20-30', '30+']  
df['ingredients_bin'] = pd.cut(df['n_ingredients'], bins=custom_bins, labels=bin_labels, right=False)
bin_counts = df['ingredients_bin'].value_counts().sort_index()

# Plotting the histogram with custom bins
plt.figure(figsize=(10, 6))
plt.bar(bin_counts.index, bin_counts.values, color='skyblue', edgecolor='black')
plt.title('Distribution of ingredients per recipe', fontsize=16)
plt.xlabel('Number of Ingredients', fontsize=14)
plt.ylabel('Number of Recipes', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%


#%%
# ______________________________________--
# DNN 
from sklearn.model_selection import train_test_split
train, test = train_test_split(n_interactions, test_size=0.2, random_state=42)
X_train = train.drop(columns=['rating'])
X_test = test.drop(columns=['rating'])
Y_train = train['rating']
Y_test = test['rating']
#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_X = scaler.fit_transform(X_train)
test_X = scaler.transform(X_train)


#%%

import tensorflow as tf
def buildDNN(activation='relu', X, Y):
    input_layer = tf.keras.layers.Input(shape=(X.shape[1],), name="input_layer")
    dense_1 = tf.keras.layers.Dense(128, activation=activation, name="dense_1")(input_layer)
    dense_2 = tf.keras.layers.Dense(64, activation=activation, name="dense_2")(dense_1)
    dense_3 = tf.keras.layers.Dense(32, activation=activation, name="dense_3")(dense_2)
    output_layer = tf.keras.layers.Dense(1, activation="linear", name="output_layer")(dense_3)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    model.summary()
