from sklearn.metrics import root_mean_squared_error


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, predictions)

    return rmse
