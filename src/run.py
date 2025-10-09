# run.py
import yaml
from data import load_data
from preprocess import preprocess_data
import train
import evaluate_models

# --- Load config file ---
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

raw_dir = cfg["paths"]["raw"]
processed_dir = cfg["paths"]["processed"]

train_raw = f"{raw_dir}/census_income_learn.csv"
test_raw  = f"{raw_dir}/census_income_test.csv"

# Target column name from config
target = cfg["target"]

def main():
    # 1. Load raw data
    train_df, test_df = load_data(train_raw, test_raw)
    print("Loaded raw:", train_df.shape, test_df.shape)

    # 2. Preprocess data
    train_clean = preprocess_data(train_df, target_col=target)
    test_clean  = preprocess_data(test_df, target_col=target)
    print("Cleaned:", train_clean.shape, test_clean.shape)

    # 3. Save processed files
    train_out = f"{processed_dir}/train_cleaned.csv"
    test_out  = f"{processed_dir}/test_cleaned.csv"
    train_clean.to_csv(train_out, index=False)
    test_clean.to_csv(test_out, index=False)
    print(f"Saved cleaned files to:\n  {train_out}\n  {test_out}")

    # 4. Train models
    print("\n === Starting Training ===")
    train.main()

    # 5. Evaluate models
    print("\n === Starting Evaluation ===")
    evaluate_models.main()

    print("\n All done!")

if __name__ == "__main__":
    main()
