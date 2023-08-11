
import pandas as pd
import numpy as np

def check_submission(submission_file, fund_pool_file):
    # Load submission and fund pool
    df_submission = pd.read_csv(submission_file)
    df_fund_pool = pd.read_csv(fund_pool_file)

    # Check if all symbols in the submission are in the fund pool
    if not df_submission['symbol'].isin(df_fund_pool['symbol']).all():
        return "Error: There are symbols in the submission that are not in the fund pool."

    # Check if any symbol in the fund pool is missing in the submission
    if not df_fund_pool['symbol'].isin(df_submission['symbol']).all():
        return "Error: There are symbols in the fund pool that are missing in the submission."

    # Check if the sum of the portfolio weights is 1
    if not np.isclose(df_submission['portfolio'].sum(), 1):
        return "Error: The sum of the portfolio weights is not 1."

    # Check if the rank_id is from 1 to n (assuming the rank_id is sorted)
    if not pd.Series(range(1, len(df_submission) + 1)).equals(df_submission['rank_id'].sort_values().reset_index(drop=True)):
        return "The 'rank_id' is not a constant ascending sequence starting from 1 after sorting."

    return "The submission file is valid."

# Test the function
# print(check_submission("predict_table.csv", "fund_pool.csv"))