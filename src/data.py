# data.py
import pandas as pd

# direct list of column names (from metadata)
# column names taken directly from metadata (everything before ":")
columns = [
    "age",
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
    "wage_per_hour",
    "enroll_in_edu_inst_last_wk",
    "marital_stat",
    "major_industry_code",
    "major_occupation_code",
    "race",
    "hispanic_origin",
    "sex",
    "member_of_a_labor_union",
    "reason_for_unemployment",
    "full_or_part_time_employment_stat",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "tax_filer_stat",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_stat",
    "detailed_household_summary_in_household",
    "instance_weight",
    "migration_code_change_in_msa",
    "migration_code_change_in_reg",
    "migration_code_move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "num_persons_worked_for_employer",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits",
    "weeks_worked_in_year",
    "year",
    "income"  # target column
]

def load_data(train_path, test_path):
    """Load the train and test census files with predefined header."""
    df_train = pd.read_csv(train_path, header=None, names=columns, sep=", ", na_values="?", engine="python")
    df_test  = pd.read_csv(test_path,  header=None, names=columns, sep=", ", na_values="?", engine="python")

    # Drop the instance_weight column as per metadata to ignore it
    df_train = df_train.drop(columns=["instance_weight"])
    df_test = df_test.drop(columns=["instance_weight"])
    return df_train, df_test

if __name__ == "__main__":
    print("data loaded successfully")

