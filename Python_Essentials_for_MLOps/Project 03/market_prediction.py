"""All imports"""
import os
import logging
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


logging.basicConfig(filename='errors.log', level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def log_errors(func):
    """
    Decorator para capturar erros e escrever no log
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as val_err:
            logging.error('Erro na função %s : %s', {func.__name__},  {val_err}, exc_info=True)
        except NameError as name_err:
            logging.error('Erro na função %s : %s', {func.__name__},  {name_err}, exc_info=True)
        except FileNotFoundError as file_err:
            logging.error('Erro ao tentar buscar o arquivo: %s', {file_err}, exc_info=True)
    return wrapper


# Loading S&P 500 data from a CSV file or fetching from Yahoo Finance and saving to CSV
if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")
sp500.index = pd.to_datetime(sp500.index)

# Plotting S&P 500 data
sp500.plot.line(y="Close", use_index=True)  # Plot to visualize the index

# Removing unnecessary columns
del sp500["Dividends"]
del sp500["Stock Splits"]

# Creating target variable for classification (1 if Close price increases tomorrow, else 0)
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

# Creating a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# Splitting data into training and testing sets
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Features used for prediction
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Training the model
model.fit(train[predictors], train["Target"])

# Making predictions using the trained model
pred = model.predict(test[predictors])
pred = pd.Series(pred, index=test.index)

# Calculating precision score for the predictions
precision_score(test["Target"], pred)

@log_errors
def predict(_train, _test, _predictors, _model):
    """
    Predicts the target variable for the test data using the trained model.
    
    Parameters:
        train (DataFrame): Training data.
        test (DataFrame): Test data.
        predictors (list): List of predictor columns.
        model (object): Trained machine learning model.
        
    Returns:
        DataFrame: Predicted values and actual target values.
    """
    _model.fit(_train[_predictors], train["Target"])
    preds = model.predict(_test[_predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([_test["Target"], preds], axis=1)
    return combined

@log_errors
def backtest(_data, _model, _predictors, _train, _test, predict_function, start=2500, step=250):
    """
    Backtests the machine learning model using the specified data and parameters.
    
    Parameters:
        data (DataFrame): Data for backtesting.
        model (object): Trained machine learning model.
        predictors (list): List of predictor columns.
        start (int): Starting index for backtesting.
        step (int): Step size for backtesting.
        
    Returns:
        DataFrame: Combined predicted values and actual target values from backtesting.
    """
    all_predictions = []
    for i in range(start, _data.shape[0], step):
        _train = _data.iloc[0:i].copy()
        _test = _data.iloc[i:(i+step)].copy()
        prediction = predict_function(_train, _test, _predictors, _model)
        all_predictions.append(prediction)
    return pd.concat(all_predictions)

# Performing backtesting and evaluating the model
predictions = backtest(sp500, model, predictors, predict, train, test)
predictions["Predictions"].value_counts()
precision_score(predictions["Target"], predictions["Predictions"])
#predictions["Target"].value_counts() / predictions.shape[0] #Just to see the result

# Creating additional features for prediction based on different time horizons
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

# Dropping rows with missing values in columns other than "Tomorrow"
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])

@log_errors
def predict2(_train, _test, _predictors, _model):
    """
    Predicts the target variable for the test data using the trained model 
    and thresholds the probabilities at 0.6.
    
    Parameters:
        train (DataFrame): Training data.
        test (DataFrame): Test data.
        predictors (list): List of predictor columns.
        model (object): Trained machine learning model.
        
    Returns:
        DataFrame: Predicted values and actual target values.
    """
    _model.fit(_train[predictors], _train["Target"])
    preds = _model.predict_proba(_test[_predictors])[:, 1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([_test["Target"], preds], axis=1)
    return combined

# Performing backtesting with new predictors and evaluating the model
predictions = backtest(sp500, model, new_predictors, train, test, predict2)
predictions["Predictions"].value_counts()
precision_score(predictions["Target"], predictions["Predictions"])
#predictions["Target"].value_counts() / predictions.shape[0] #Just to see the result

# Displaying the final predictions
#predictions
