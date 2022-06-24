from turtle import back
from flask import Flask, request, jsonify, render_template
from matplotlib import ticker
import yfinance as yf
import pandas as pd
import util
from fastapi import FastAPI
import backtest
from fastapi.middleware.cors import CORSMiddleware
import joblib



# #initiate flask app
# app = create_app()

# prediction csv
all_prediction_files = util.read_all_stock_files('stock_prediction')

# model per stock
all_model_files = util.import_all_model('stock_model_new')


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "hello world"}

#get todays open close volume
@app.get("/data/{ticker_name}/{period}")
async def get_ticker(ticker_name: str, period:int):
    tick = ticker_name.upper()
    result, signal = util.GetDataClosePrice(tick, period, all_model_files)

    return {'data': result.to_dict(orient="records"), 'signal_today': signal}

#back testing
@app.get("/backtesting/{ticker_name}/{period}")
async def get_backtesting(ticker_name: str, period:int):
    stock_name = ticker_name.upper()

    prediction, y_test, dataset_test = util.RunPrediction(all_model_files,stock_name, period)


    #backtesting with stoploss
    trade_df_stoploss, returns_stoploss = backtest.RunBacktesting(dataset_test, stock_name, True)
    stats_stoploss = backtest.BacktestingStatistics(returns_stoploss)

    #backtesting without stoploss

    trade_df_no_stoploss, returns_no_stoploss = backtest.RunBacktesting(dataset_test, stock_name, False)
    stats_no_stoploss = backtest.BacktestingStatistics(returns_no_stoploss)

    print(stats_no_stoploss)

    ## TODO : fix indexing for return dict
    response = {'backtest_stoploss_trade': trade_df_stoploss.to_dict('records') , 
                'backtest_stoploss_stats': stats_stoploss, 
                'backtest_no_stoploss_trade': trade_df_no_stoploss.to_dict('records'), 
                'backtest_no_stoploss_stats': stats_no_stoploss}

    return response

#backtesting with no money management


# get model confusion matrix

# get model evaluation

@app.get("/evaluation/{ticker_name}/{period}")
async def get_evaluation(ticker_name: str, period:int):
    prediction, y_test, dataset_test = util.RunPrediction(all_model_files,ticker_name, period)
    print(dataset_test)
    report = util.GetEvaluationMatrix(y_test, prediction)

    #model parameter
    model_name = f'{ticker_name.upper()}.JK_{period}.pkl'
    model_params = str(all_model_files[model_name].best_params_)

    report['model_params'] = model_params

    return {'report': report}


#TODO: delete when it's done, code for prediction
# dataset_test, label_3 = util.PreparePredictionData(all_prediction_files, "BBCA", 7)

# prediction = util.ModelPrediction(all_model_files, "BBCA", 7, dataset_test)

# print(len(prediction))
# report = util.GetEvaluationMatrix(label_3.to_numpy(), prediction)
# print(report)


# prediction, y_test, dataset_test = util.RunPrediction(all_model_files,"BBCA", 3)
# print(dataset_test)

# trade_df, returns = backtest.RunBacktesting(dataset_test, "BBCA")
# print(returns)

 
# report = util.GetEvaluationMatrix(y_test, prediction)
# print(report)
# print("------")


print(all_model_files['BBCA.JK_3.pkl'].best_params_)