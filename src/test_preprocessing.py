from data_loader import load_data
from preprocessing import handle_missing_values

df = load_data("data/train.csv")

print(df.isnull().sum().sum())

df_clean = handle_missing_values(df)

print(df_clean.isnull().sum().sum())
