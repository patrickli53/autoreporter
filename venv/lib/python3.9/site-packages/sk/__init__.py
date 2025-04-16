import numpy as np
import pandas as pd

from sklearn import datasets

from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier,  RandomForestRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    SGDClassifier, SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    mean_absolute_error, mean_squared_error, explained_variance_score,
)

from sklearn.model_selection import train_test_split
