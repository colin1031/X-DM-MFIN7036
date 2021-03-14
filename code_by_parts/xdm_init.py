#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 22:17:41 2021

@author: u3583042
"""


import os
import datetime
from datetime import datetime as dt
import statsmodels.formula.api as smf
import time
import twint
import pandas as pd
import asyncio
import nest_asyncio
import glob
import nltk
import textstat
import openpyxl
import statsmodels.formula.api as smf
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from numpy import nan
from wordcloud import WordCloud
from datetime import datetime as dt
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.model_selection import learning_curve
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def set_dic(path):
    os.chdir(path)
    print('working directory set to:' + str(os.getcwd()))