"""
prepare_rl_dataset.py
Prepare the LendingClub dataset for offline RL training.
This extends your existing data_prep.py work to create RL-specific data structures.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys

class RLDatasetBuilder:
    """
    Convert preprocessed loan data into offline RL format.
    Creates (state, action, reward, next_state, done) tuples.
    """
    
    def __init__(self, preprocessed_data_path=None, df=None, scaler=None):
        """
        Initialize with either preprocessed data path or DataFrame.
        
        Args:
            preprocessed_data_path: Path to your preprocessed CSV/pickle
            df: Already preprocessed DataFrame
            scaler: Pre-fitted StandardScaler (will create new if None)
        """
        if df is not None:
            self.df = df.copy()
        elif preprocessed_data_path:
            print(f"Loading preprocessed data from {preprocessed_data_path}...")
            if preprocessed_data_path.endswith('.pkl'):
                self.df = pd.read_pickle(preprocessed_data_path)
            else:
                self.df = pd.read_csv(preprocessed_data_path)
        else:
            raise ValueError("Must provide either preprocessed_data_path or df")
        
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.scaler_fitted = scaler is not None
        
    def build_rl_dataset(self, feature_cols, loan_amnt_col='loan_amnt', 
                         int_rate_col='int_rate', loan_status_col='loan_status',
                         test_size=0.2, random_state=42):
        """
        Build the offline RL dataset.
        
        Args:
            feature_cols: List of column names to use as state features
            loan_amnt_col: Column name for loan amount
            int_rate_col: Column name for interest rate
            loan_status_col: Column name for loan status
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            rl_dataset: Dictionary with train/test splits and metadata
        """
        print("\n=== Building RL Dataset ===")
        
        # 1. Filter to resolved loans only
        resolved_statuses = ['Fully Paid', 'Charged Off', 'Default']
        df = self.df[self.df[loan_status_col].isin(resolved_statuses)].copy()
        print(f"Filtered to {len(df):,} resolved loans")
        
        # 2. Verify required columns exist
        required_cols = [loan_amnt_col, int_rate_col, loan_status_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        # 3. Extract features (states)
        X = df[feature_cols].values
        
        # 4. Scale features if not already fitted
        if not self.scaler_fitted:
            print("Fitting scaler on features...")
            X = self.scaler.fit_transform(X)
            self.scaler_fitted = True
        else:
            X = self.scaler.transform(X)
        
        # 5. Create actions (all loans in dataset were approved)
        actions = np.ones(len(df), dtype=np.int32)  # 1 = Approve
        
        # 6. Calculate rewards
        print("Calculating rewards...")
        rewards = self._calculate_rewards(
            df[loan_status_col].values,
            df[loan_amnt_col].values,
            df[int_rate_col].values
        )
        
        print(f"  Mean reward: ${rewards.mean():,.2f}")
        print(f"  Std reward: ${rewards.std():,.2f}")
        print(f"  Min reward: ${rewards.min():,.2f}")
        print(f"  Max reward: ${rewards.max():,.2f}")
        
        # 7. Create next_states (terminal states for single-step episodes)
        next_states = X.copy()
        dones = np.ones(len(df), dtype=np.float32)
        
        # 8. Train/test split
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        # 9. Package into RL dataset format
        rl_dataset = {
            'train': {
                'states': X[train_idx],
                'actions': actions[train_idx],
                'rewards': rewards[train_idx],
                'next_states': next_states[train_idx],
                'dones': dones[train_idx]
            },
            'test': {
                'states': X[test_idx],
                'actions': actions[test_idx],
                'rewards': rewards[test_idx],
                'next_states': next_states[test_idx],
                'dones': dones[test_idx]
            },
            'metadata': {
                'feature_cols': feature_cols,
                'state_dim': X.shape[1],
                'action_dim': 2,  # Binary: Deny=0, Approve=1
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'reward_stats': {
                    'mean': float(rewards.mean()),
                    'std': float(rewards.std()),
                    'min': float(rewards.min()),
                    'max': float(rewards.max())
                }
            }
        }
        
        print(f"\n✓ RL Dataset built successfully:")
        print(f"  State dimension: {rl_dataset['metadata']['state_dim']}")
        print(f"  Train samples: {rl_dataset['metadata']['n_train']:,}")
        print(f"  Test samples: {rl_dataset['metadata']['n_test']:,}")
        
        return rl_dataset
    
    def _calculate_rewards(self, loan_statuses, loan_amounts, interest_rates):
        """
        Calculate rewards based on loan outcomes.
        
        Reward structure:
        - Fully Paid: +loan_amount * (interest_rate / 100)
        - Charged Off/Default: -loan_amount
        
        Args:
            loan_statuses: Array of loan status strings
            loan_amounts: Array of loan amounts
            interest_rates: Array of interest rates
            
        Returns:
            rewards: Array of reward values
        """
        rewards = np.zeros(len(loan_statuses))
        
        # Fully Paid = positive reward (interest earned)
        fully_paid_mask = loan_statuses == 'Fully Paid'
        rewards[fully_paid_mask] = (
            loan_amounts[fully_paid_mask] * 
            interest_rates[fully_paid_mask] / 100
        )
        
        # Charged Off/Default = negative reward (lose principal)
        default_mask = np.isin(loan_statuses, ['Charged Off', 'Default'])
        rewards[default_mask] = -loan_amounts[default_mask]
        
        return rewards
    
    def save_dataset(self, rl_dataset, output_dir='../data'):
        """
        Save RL dataset and scaler to disk.
        
        Args:
            rl_dataset: The RL dataset dictionary
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save RL dataset
        dataset_path = os.path.join(output_dir, 'rl_loan_dataset.pkl')
        with open(dataset_path, 'wb') as f:
            pickle.dump(rl_dataset, f)
        print(f"✓ Saved RL dataset to: {dataset_path}")
        
        # Save scaler separately
        scaler_path = os.path.join(output_dir, 'rl_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved scaler to: {scaler_path}")
        
        return dataset_path, scaler_path
    
    @staticmethod
    def load_dataset(dataset_path='../data/rl_loan_dataset.pkl', 
                     scaler_path='../data/rl_scaler.pkl'):
        """
        Load saved RL dataset and scaler.
        
        Returns:
            rl_dataset, scaler
        """
        with open(dataset_path, 'rb') as f:
            rl_dataset = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"✓ Loaded RL dataset from: {dataset_path}")
        print(f"  Train samples: {rl_dataset['metadata']['n_train']:,}")
        print(f"  Test samples: {rl_dataset['metadata']['n_test']:,}")
        
        return rl_dataset, scaler


if __name__ == "__main__":
    """
    Example usage - run this after your DL preprocessing is done.
    """
    
    # Load your preprocessed data
    # Option 1: If you saved preprocessed data
    # builder = RLDatasetBuilder(preprocessed_data_path='../data/preprocessed_loans.csv')
    
    # Option 2: If you want to load from your data_prep module
    from data_prep import load_and_preprocess_data
    
    print("Loading and preprocessing data...")
    df, feature_cols, target = load_and_preprocess_data(
        data_path='../data/accepted_2007_to_2018Q4.csv.gz',
        sample_size=100000  # Adjust as needed
    )
    
    builder = RLDatasetBuilder(df=df)
    
    # Build RL dataset
    rl_dataset = builder.build_rl_dataset(
        feature_cols=feature_cols,
        test_size=0.2
    )
    
    # Save for training
    builder.save_dataset(rl_dataset)
    
    print("\n✓ RL dataset preparation complete!")
    print("  Next step: Run train_rl.py to train the agent")