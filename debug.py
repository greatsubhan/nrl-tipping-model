import pandas as pd

print("=== fixtures_round_2.csv ===")
fixtures = pd.read_csv("fixtures_round_2.csv")
print("Columns:", list(fixtures.columns))
print("First row:")
print(fixtures.head(1))

print("\n=== model_round_2.csv ===")
model = pd.read_csv("model_round_2.csv")
print("Columns:", list(model.columns))
print("First row:")
print(model.head(1))
