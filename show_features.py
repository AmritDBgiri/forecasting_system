import pandas as pd
from data_processor import load_and_clean, add_features

# Load and engineer
df_clean = load_and_clean("data/sales_data.csv")
df_feat  = add_features(df_clean)

# Show one state as example
state = "California"
sample = df_feat[df_feat["State"] == state].dropna().tail(5)

print("=" * 65)
print(f"  FEATURE ENGINEERING OUTPUT — {state}")
print("=" * 65)

print("\n LAG FEATURES (past sales values):")
print(sample[["Date", "Sales", "lag_1", "lag_2", "lag_4", "lag_8"]].to_string(index=False))

print("\n ROLLING STATISTICS (trend smoothing):")
print(sample[["Date", "roll_mean_4", "roll_std_4", "roll_mean_8", "roll_std_8"]].to_string(index=False))

print("\n CALENDAR FEATURES:")
print(sample[["Date", "week_of_year", "month", "quarter", "year"]].to_string(index=False))

print("\n HOLIDAY FLAG:")
print(sample[["Date", "holiday_week"]].to_string(index=False))

print("\n FULL FEATURE SUMMARY:")
print(f"  Total rows (all states) : {len(df_feat)}")
print(f"  Total states            : {df_feat['State'].nunique()}")
print(f"  Features created        : {len(df_feat.columns) - 2}")  # minus State, Date
print(f"  Columns: {list(df_feat.columns)}")

print("\n MISSING VALUE CHECK (after gap-fill):")
print(df_clean[["Sales"]].isnull().sum().rename("Nulls after cleaning"))