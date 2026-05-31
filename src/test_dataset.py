from data_loader import load_data
from preprocessing import handle_missing_values
from dataset import split_dataset

df = load_data("data/train.csv")
df = handle_missing_values(df)

X = df.drop(columns=["SalePrice"])

y = df["SalePrice"]
X_train, X_test, y_train, y_test = split_dataset(X, y)

print(X_train.shape)
print(X_test.shape)
