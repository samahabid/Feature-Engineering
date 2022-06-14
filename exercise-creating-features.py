#!/usr/bin/env python
# coding: utf-8

# **This notebook is an exercise in the [Feature Engineering](https://www.kaggle.com/learn/feature-engineering) course.  You can reference the tutorial at [this link](https://www.kaggle.com/ryanholbrook/creating-features).**
# 
# ---
# 

# # Introduction #
# 
# In this exercise you'll start developing the features you identified in Exercise 2 as having the most potential. As you work through this exercise, you might take a moment to look at the data documentation again and consider whether the features we're creating make sense from a real-world perspective, and whether there are any useful combinations that stand out to you.
# 
# Run this cell to set everything up!

# In[1]:


# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering_new.ex3 import *

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# Prepare data
df = pd.read_csv("../input/fe-course-data/ames.csv")
X = df.copy()
y = X.pop("SalePrice")


# -------------------------------------------------------------------------------
# 
# Let's start with a few mathematical combinations. We'll focus on features describing areas -- having the same units (square-feet) makes it easy to combine them in sensible ways. Since we're using XGBoost (a tree-based model), we'll focus on ratios and sums.
# 
# # 1) Create Mathematical Transforms
# 
# Create the following features:
# 
# - `LivLotRatio`: the ratio of `GrLivArea` to `LotArea`
# - `Spaciousness`: the sum of `FirstFlrSF` and `SecondFlrSF` divided by `TotRmsAbvGrd`
# - `TotalOutsideSF`: the sum of `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `Threeseasonporch`, and `ScreenPorch`

# In[4]:


# YOUR CODE HERE
X_1 = pd.DataFrame()  # dataframe to hold new features

X_1["LivLotRatio"] = df.GrLivArea / df.LotArea
X_1["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
X_1["TotalOutsideSF"] = df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch + df.Threeseasonporch + df.ScreenPorch


# Check your answer
q_1.check()


# In[5]:


# Lines below will give you a hint or solution code
#q_1.hint()
#q_1.solution()


# -------------------------------------------------------------------------------
# 
# If you've discovered an interaction effect between a numeric feature and a categorical feature, you might want to model it explicitly using a one-hot encoding, like so:
# 
# ```
# # One-hot encode Categorical feature, adding a column prefix "Cat"
# X_new = pd.get_dummies(df.Categorical, prefix="Cat")
# 
# # Multiply row-by-row
# X_new = X_new.mul(df.Continuous, axis=0)
# 
# # Join the new features to the feature set
# X = X.join(X_new)
# ```
# 
# # 2) Interaction with a Categorical
# 
# We discovered an interaction between `BldgType` and `GrLivArea` in Exercise 2. Now create their interaction features.

# In[7]:


# YOUR CODE HERE
# One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
X_2 = pd.get_dummies(df.BldgType, prefix="Bldg")

# Multiply
X_2 = X_2.mul(df.GrLivArea, axis=0)


# Check your answer
q_2.check()


# In[8]:


# Lines below will give you a hint or solution code
#q_2.hint()
#q_2.solution()


# # 3) Count Feature
# 
# Let's try creating a feature that describes how many kinds of outdoor areas a dwelling has. Create a feature `PorchTypes` that counts how many of the following are greater than 0.0:
# 
# ```
# WoodDeckSF
# OpenPorchSF
# EnclosedPorch
# Threeseasonporch
# ScreenPorch
# ```

# In[10]:


X_3 = pd.DataFrame()

# YOUR CODE HERE
X_3["PorchTypes"] =  df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)


# Check your answer
q_3.check()


# In[11]:


# Lines below will give you a hint or solution code
#q_3.hint()
#q_3.solution()


# # 4) Break Down a Categorical Feature
# 
# `MSSubClass` describes the type of a dwelling:

# In[ ]:


df.MSSubClass.unique()


# You can see that there is a more general categorization described (roughly) by the first word of each category. Create a feature containing only these first words by splitting `MSSubClass` at the first underscore `_`. (Hint: In the `split` method use an argument `n=1`.)

# In[13]:


X_4 = pd.DataFrame()

X_4["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]

# Check your answer
q_4.check()


# In[12]:


# Lines below will give you a hint or solution code
#q_4.hint()
#q_4.solution()


# # 5) Use a Grouped Transform
# 
# The value of a home often depends on how it compares to typical homes in its neighborhood. Create a feature `MedNhbdArea` that describes the *median* of `GrLivArea` grouped on `Neighborhood`.

# In[15]:


X_5 = pd.DataFrame()

# YOUR CODE HERE
X_5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")

# Check your answer
q_5.check()


# In[16]:


# Lines below will give you a hint or solution code
#q_5.hint()
#q_5.solution()


# Now you've made your first new feature set! If you like, you can run the cell below to score the model with all of your new features added:

# In[ ]:


X_new = X.join([X_1, X_2, X_3, X_4, X_5])
score_dataset(X_new, y)


# # Keep Going #
# 
# [**Untangle spatial relationships**](https://www.kaggle.com/ryanholbrook/clustering-with-k-means) by adding cluster labels to your dataset.

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/feature-engineering/discussion) to chat with other learners.*
