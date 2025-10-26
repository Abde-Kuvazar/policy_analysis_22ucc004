# src/prepare_rl_dataset.py
import pandas as pd
from pathlib import Path
import numpy as np

# ----------------------
# Paths
# ----------------------
accepted_path = Path('data/processed/sample_processed_for_modeling.csv')
rejected_path = Path('data/raw/rejected_2007_to_2018q4.csv')
output_path = Path('data/processed/rl_dataset.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)

# ----------------------
# Load accepted loans
# ----------------------
accepted = pd.read_csv(accepted_path, parse_dates=['issue_d_parsed'], low_memory=False)

# Columns for RL dataset
essential_columns = [
    'loan_amnt','int_rate','term','grade','emp_length','home_ownership',
    'annual_inc','verification_status','purpose','addr_state','dti',
    'open_acc','total_acc','revol_bal','revol_util','application_type','target'
]

existing_cols = [c for c in essential_columns if c in accepted.columns]

# Prepare accepted loans
accepted_common = accepted[existing_cols].copy()
accepted_common['action'] = 1
accepted_common['reward'] = accepted_common.apply(
    lambda row: float(row['loan_amnt'] * (row['int_rate']/100)) if int(row['target'])==0 else -float(row['loan_amnt']), axis=1
)

# ----------------------
# Write accepted loans first
# ----------------------
accepted_common.to_csv(output_path, index=False)

# ----------------------
# Process rejected loans in chunks (RAM safe)
# ----------------------
chunksize = 50000
cols_to_use = ['Amount Requested', 'Application Date', 'Risk_Score', 
               'Debt-To-Income Ratio', 'State', 'Employment Length']

for chunk in pd.read_csv(rejected_path, usecols=cols_to_use, parse_dates=['Application Date'],
                         chunksize=chunksize, low_memory=True):

    # Rename columns
    chunk.rename(columns={
        'Amount Requested': 'loan_amnt',
        'Application Date': 'issue_d_parsed',
        'Risk_Score': 'fico_range_low',  # placeholder
        'Debt-To-Income Ratio': 'dti',
        'State': 'addr_state',
        'Employment Length': 'emp_length'
    }, inplace=True)

    # Fill missing columns
    for col in existing_cols:
        if col not in chunk.columns:
            if col in ['int_rate','annual_inc','open_acc','total_acc','revol_bal','revol_util']:
                chunk[col] = 0.0
            elif col == 'target':
                chunk[col] = 1  # treat rejected as defaulted
            else:
                chunk[col] = 'Unknown'

    # Add action and reward
    chunk['action'] = 0
    chunk['reward'] = 0.0

    # Keep correct column order
    chunk = chunk[existing_cols + ['action','reward']]

    # Append to CSV without loading full DataFrame
    chunk.to_csv(output_path, mode='a', index=False, header=False)

print("RL dataset saved at:", output_path)
