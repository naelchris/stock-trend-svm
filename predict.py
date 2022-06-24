import util

def predict_stock_svm(model, data_test):
    prediction = model.prediction(data_test)
    return prediction
