from data_loader import load_data
from preprocessing import handle_missing_values
from dataset import split_dataset

from train import train_model
from evaluate import evaluate_model

df = load_data("data/train.csv")
df = handle_missing_values(df)

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = split_dataset(X, y)

model = train_model(X_train, y_train)

rmse = evaluate_model(model, X_test, y_test)

print(rmse)
