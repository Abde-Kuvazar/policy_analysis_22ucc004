# PREPROCESSING PIPELINE
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = Path.cwd() / 'data' / 'processed' / 'sample_processed_for_modeling.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['issue_d_parsed'], low_memory=False)
print("loaded", df.shape)

# ------------- Feature selection (based on EDA) ----------------
NUMERIC = [c for c in ['loan_amnt','int_rate','annual_inc','dti','fico_avg','revol_bal','revol_util','open_acc','total_acc'] if c in df.columns]
CAT = [c for c in ['term','grade','sub_grade','emp_length','home_ownership','verification_status','purpose','addr_state','application_type'] if c in df.columns]

# create or ensure numeric helpers exist
if 'annual_inc' in df.columns and 'annual_inc_log' not in df.columns:
    df['annual_inc_log'] = np.log1p(df['annual_inc'].clip(lower=0))

if 'fico_avg' not in df.columns and 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
    df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2

# Map emp_length to numeric if present
if 'emp_length' in df.columns:
    def emp_map(x):
        if pd.isnull(x): return np.nan
        if '10+' in str(x): return 10.0
        if '<' in str(x): return 0.0
        try:
            return float(str(x).split()[0])
        except:
            return np.nan
    df['emp_length_num'] = df['emp_length'].map(emp_map)
    # replace 'emp_length' in CAT with 'emp_length_num' numeric
    if 'emp_length' in CAT:
        CAT.remove('emp_length')
        NUMERIC.append('emp_length_num')

# update lists
if 'annual_inc_log' in df.columns and 'annual_inc' in NUMERIC:
    # use log income instead of raw
    NUMERIC = [c for c in NUMERIC if c != 'annual_inc']
    NUMERIC.append('annual_inc_log')

print("NUMERIC:", NUMERIC)
print("CAT:", CAT)

# ------------- Time-based split (train <= 2015, val=2016, test >= 2017) ----------------
df = df.dropna(subset=['target'])  # ensure target present
df['issue_d_parsed'] = pd.to_datetime(df['issue_d_parsed'], errors='coerce')
train_df = df[df['issue_d_parsed'] <= '2015-12-31'].copy()
val_df = df[(df['issue_d_parsed'] > '2015-12-31') & (df['issue_d_parsed'] <= '2016-12-31')].copy()
test_df = df[df['issue_d_parsed'] > '2016-12-31'].copy()

# fallback if split produces empty sets (rare in sampling) -> random time-insensitive split
if len(train_df) < 1000 or len(test_df) < 1000:
    print("Time split too small, doing stratified random split instead.")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)
    val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['target'], random_state=42)[1]

print("train/val/test sizes:", len(train_df), len(val_df), len(test_df))

# ------------- Pipeline ----------------
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For categorical, use OneHot for low-cardinality, else frequency+ohe. We will one-hot but limit high-cardinality via top-k
# we'll top-K the addr_state to top 30 and mark rest as 'other'
def topk_cat(series, k=30):
    top = series.value_counts().index[:k]
    return series.where(series.isin(top), other='__other__')

# Apply topk to heavy cardinality cats
if 'addr_state' in CAT:
    train_df['addr_state'] = topk_cat(train_df['addr_state'], k=30)
    val_df['addr_state'] = topk_cat(val_df['addr_state'], k=30)
    test_df['addr_state'] = topk_cat(test_df['addr_state'], k=30)

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, NUMERIC),
    ('cat', categorical_transformer, CAT)
], sparse_threshold=0)

# Fit preprocessor on training data only
X_train = preprocessor.fit_transform(train_df[NUMERIC + CAT])
X_val = preprocessor.transform(val_df[NUMERIC + CAT])
X_test = preprocessor.transform(test_df[NUMERIC + CAT])

y_train = train_df['target'].astype(int).values
y_val = val_df['target'].astype(int).values
y_test = test_df['target'].astype(int).values



# Save numpy arrays for training DL
np.save('models/X_train.npy', X_train)
np.save('models/y_train.npy', y_train)
np.save('models/X_val.npy', X_val)
np.save('models/y_val.npy', y_val)
np.save('models/X_test.npy', X_test)
np.save('models/y_test.npy', y_test)

print("X shapes:", X_train.shape, X_val.shape, X_test.shape)
joblib.dump(preprocessor, Path.cwd() / 'models' / 'preprocessor.joblib')
