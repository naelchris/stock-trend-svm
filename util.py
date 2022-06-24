from email.policy import default
import numpy as np
import pandas as pd
import math
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from datetime import datetime

import yfinance as yf


import indicator

import pickle as pk

def read_all_stock_files(folder_path):
    allFiles = []
    for (_, _, files) in os.walk(folder_path):
        allFiles.extend(files)
        break

    dataframe_dict = {}
    for stock_file in allFiles:
        df = pd.read_csv(folder_path +  "/" + stock_file)
        dataframe_dict[(stock_file.split('.'))[0]] = df
    
    return dataframe_dict

#TODO : function to importing model
def import_all_model(folder_path):
    modelTicker_dict = {}
    allFiles = []

    for (_, _, files) in os.walk(folder_path):
        allFiles.extend(files)
    
    for stock_model in allFiles:
        model = joblib.load(folder_path+ '/'+ stock_model)
        modelTicker_dict[stock_model] = model

    return modelTicker_dict

def preprocess_data(dataset):
    feature_name = ['Close','Volume', 'EMA_40', 'EMA_100', 'ADX', 'RSI', 'OBV', 'ATR',
       'NATR', 'TRANGE', 'macd_line', 'macd_9_day', 'macd_diff', 'ma_50_day',
       'ma_200_day', 'ma_50_200', 'BBANDS_U', 'BBANDS_M', 'BBANDS_L']

    components = 8

    X = dataset[feature_name]

    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)

    pca = PCA(components)
    pca.fit(X)
    X = pca.transform(X)
    
    return X


def pca_preprocessing(dataset, signal_period) :

  X_train, X_test, y_train, y_test, dataset_train, dataset_test, dataset = split_dataset(dataset, signal_period)
  components = 8

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  pca = PCA(n_components=components)
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)

  print("shape of the scaled and 'PCA'ed features train: ", np.shape(X_train))
  print("shape of the scaled and 'PCA'ed features test: ", np.shape(X_test))


  print (f"Variance Ratio of the {components} Principal Components Ananlysis: ", np.sum(np.var(X_train, axis=0)/(np.sum(np.var(X_train, axis=0)))))
  print (f"Variance Ratio of the {components} Principal Components Ananlysis: ",np.sum(np.var(X_test, axis=0)/(np.sum(np.var(X_test, axis=0)))))

  return X_train, X_test, y_train, y_test, dataset_train, dataset_test, dataset



def split_dataset(dataset, signal_period):
  feature_name = ['Close','Volume', 'EMA_40', 'EMA_100', 'ADX', 'RSI', 'OBV', 'ATR',
       'NATR', 'TRANGE', 'macd_line', 'macd_9_day', 'macd_diff', 'ma_50_day',
       'ma_200_day', 'ma_50_200', 'BBANDS_U', 'BBANDS_M', 'BBANDS_L']

  signals = ['signal_3', 'signal_7', 'signal_10']
       
  #   'returns','returns_2',  'rtn_lag1', 
  #  'rtn_lag2', 'rtn_lag3', 'rtn_lag4', 'rtn_lag5', 'rtn_lag1_bin',
  #  'rtn_lag2_bin', 'rtn_lag3_bin', 'rtn_lag4_bin', 'rtn_lag5_bin']
       

  #feature_name = ['Close', 'Adj Close', 'Volume', 'EMA_40', 'EMA_100', 'OBV', 'ATR', 'TRANGE', 'ma_50_day', 'ma_50_200']
  # feature_name = ['Close','Volume', 'EMA_40', 'EMA_100', 'ADX', 'RSI', 'OBV', 'ATR',
  #      'NATR', 'TRANGE', 'macd_line', 'macd_9_day', 'macd_diff', 'ma_50_day',
  #      'ma_200_day', 'ma_50_200']

  X = dataset[feature_name]
  y = dataset[signal_period]
  split = int(len(dataset)*0.8)

  X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

  dataset_train, dataset_test = dataset[:split], dataset[split:]

  # fix imbalance between uptrend and downtrend
  oversample = SMOTE()

  X_train, y_train = oversample.fit_resample(X_train, y_train)

  
  return X_train, X_test, y_train, y_test, dataset_train, dataset_test, dataset



def GetStockPriceData(stock_df):
    df = stock_df[['Date','Open','High','Low','Close']]
    df["Date"] = pd.to_datetime(df["Date"])

    return df

def GetEvaluationMatrix(y_test, prediction):
    cm = confusion_matrix(y_test, prediction)

    tp = cm[0,0]
    tn = cm[1,1]
    fp = cm[0,1]
    fn = cm[1,0]

    report = classification_report(y_test, prediction, output_dict=True)
    report['true_positive'] = int(tp)
    report['true_negative'] = int(tn)
    report['false_positive'] = int(fp)
    report['false_negative'] = int(fn)

    return report

def PreparePredictionData(ticker, period):

    #period dict
    signal = {3:"signal_3", 7:"signal_7", 10:"signal_10", default:"signal_3"}

    #dataset = stock_df[ticker.upper()]
    dataset = GetDatasetFromYahoo(ticker)

    #create indicator from prices
    dataset = indicator.feature_indicator(dataset)

    # prepare data train signal 3 period
    _, X_test, _, y_test, _, dataset_test, dataset = pca_preprocessing(dataset, signal[period])

    return X_test, y_test, dataset_test

def ModelPrediction(model_list,ticker, period, dataset):

    stock_name = ticker.upper()
    model_name = f'{stock_name}.JK_{period}.pkl'
    
    model = model_list[model_name]

    prediction = model.predict(dataset)

    return prediction

def GetDatasetFromYahoo(ticker:str):
    now = datetime.now() # current date and time
    start_time = '2010-12-01'
    end_time = '2022-05-29'

    tick = f'{ticker.upper()}.JK'

    stock_df = yf.download(tick, start=start_time, end=end_time, progress=False)

    return stock_df


def RunPrediction(files_model, ticker, period):
    X_test, y_test, dataset_test = PreparePredictionData(ticker, period)

    prediction = ModelPrediction(files_model, ticker, period, X_test)

    dataset_test['strategy_svm'] = prediction

    stock_df = dataset_test.reset_index()
    dataset_test = stock_df[['Date','Open','High','Low','Close','Volume']]
    dataset_test['strategy_svm'] = prediction



    return prediction, y_test, dataset_test


def GetDataClosePrice(ticker:str, period:int, files_model):
    prediction, y_test, dataset_test = RunPrediction(files_model, ticker, period)

    signal = dataset_test[['strategy_svm']].iloc[-1]

    return dataset_test, signal





#TODO : function to predict the signal from start to end

#TODO : function to backtest the rule

#TODO : function to get csv prediction

#TODO : logic for reading from csv or load model

# TODO : function preprocess the new data