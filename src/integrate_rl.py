"""
integrate_rl.py
Quick start script to integrate RL code with your existing project.
Run this to check if everything is set up correctly.
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if all required packages are installed."""
    print("\n" + "="*70)
    print("ENVIRONMENT CHECK")
    print("="*70)
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check PyTorch CUDA
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except:
        pass
    
    return True


def check_file_structure():
    """Check if required files and directories exist."""
    print("\n" + "="*70)
    print("FILE STRUCTURE CHECK")
    print("="*70)
    
    # Get project root (parent of src)
    project_root = Path(__file__).parent.parent
    
    required_dirs = ['data', 'models', 'src', 'notebooks']
    required_files = [
        'src/data_prep.py',
        'src/prepare_rl_dataset.py',
        'src/train_rl.py',
        'src/compare_models.py'
    ]
    
    print(f"\nProject root: {project_root}\n")
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory NOT found")
            os.makedirs(dir_path, exist_ok=True)
            print(f"  → Created {dir_name}/ directory")
    
    # Check files
    print()
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} NOT found")
    
    return True


def check_data():
    """Check if data files exist."""
    print("\n" + "="*70)
    print("DATA CHECK")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    # Check for raw data
    raw_data_file = data_dir / 'accepted_2007_to_2018Q4.csv.gz'
    if raw_data_file.exists():
        size_mb = raw_data_file.stat().st_size / (1024**2)
        print(f"✓ Raw data file found ({size_mb:.1f} MB)")
    else:
        print(f"✗ Raw data file NOT found at: {raw_data_file}")
        print("  → Download from Kaggle: LendingClub Loan Data")
    
    # Check for RL dataset
    rl_dataset_file = data_dir / 'rl_loan_dataset.pkl'
    rl_scaler_file = data_dir / 'rl_scaler.pkl'
    
    if rl_dataset_file.exists():
        size_mb = rl_dataset_file.stat().st_size / (1024**2)
        print(f"✓ RL dataset prepared ({size_mb:.1f} MB)")
    else:
        print(f"✗ RL dataset NOT prepared")
        print("  → Run: python src/prepare_rl_dataset.py")
    
    if rl_scaler_file.exists():
        print(f"✓ RL scaler saved")
    else:
        print(f"✗ RL scaler NOT found")


def check_models():
    """Check if models are trained."""
    print("\n" + "="*70)
    print("MODEL CHECK")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    
    # Check for DL model
    dl_model_files = list(models_dir.glob('dl_model*'))
    if dl_model_files:
        print(f"✓ DL model found: {dl_model_files[0].name}")
    else:
        print(f"✗ DL model NOT found")
        print("  → You should have already trained this")
    
    # Check for RL model
    rl_model_file = models_dir / 'cql_loan_agent.pth'
    if rl_model_file.exists():
        size_kb = rl_model_file.stat().st_size / 1024
        print(f"✓ RL agent trained ({size_kb:.1f} KB)")
    else:
        print(f"✗ RL agent NOT trained")
        print("  → Run: python src/train_rl.py")


def generate_example_config():
    """Generate example configuration for your features."""
    print("\n" + "="*70)
    print("CONFIGURATION EXAMPLE")
    print("="*70)
    
    example_features = """
# Example feature list (adjust based on your DL model):

feature_cols = [
    # Loan characteristics
    'loan_amnt',        # Loan amount
    'term',             # Loan term (36 or 60 months)
    'int_rate',         # Interest rate
    'installment',      # Monthly payment
    
    # Borrower financials
    'annual_inc',       # Annual income
    'dti',              # Debt-to-income ratio
    'open_acc',         # Number of open accounts
    'total_acc',        # Total credit accounts
    'revol_bal',        # Revolving balance
    'revol_util',       # Revolving line utilization
    
    # Credit history
    'delinq_2yrs',      # Delinquencies in past 2 years
    'inq_last_6mths',   # Inquiries in last 6 months
    'pub_rec',          # Public records
    
    # Categorical (need encoding)
    'grade',            # Loan grade (A-G)
    'sub_grade',        # Sub grade (A1-G5)
    'emp_length',       # Employment length
    'home_ownership',   # Home ownership status
    'purpose',          # Loan purpose
    'addr_state'        # State
]

# Make sure these match EXACTLY with your DL model features!
"""
    
    print(example_features)
    print("\n💡 Tip: Use the SAME features you used for your DL model!")


def show_next_steps():
    """Show recommended next steps."""
    print("\n" + "="*70)
    print("RECOMMENDED NEXT STEPS")
    print("="*70)
    
    steps = """
1. ✓ Environment setup complete (if all checks passed)

2. 📝 Update prepare_rl_dataset.py:
   - Set your feature list (same as DL model)
   - Point to your preprocessed data
   - Run: python src/prepare_rl_dataset.py

3. 🤖 Train RL agent:
   - Run: python src/train_rl.py
   - This will take 5-20 minutes depending on data size
   - Model will be saved to models/cql_loan_agent.pth

4. 📊 Compare models:
   - Update compare_models.py to load your DL model
   - Run: python src/compare_models.py
   - This generates comparison plots and insights

5. 📄 Write report:
   - Use outputs from comparison script
   - Include training curves
   - Discuss differences in metrics
   - Explain interesting cases
   - Propose future improvements

6. 🚀 Submit:
   - Push everything to GitHub
   - Include clear README with instructions
   - Submit PDF report
"""
    
    print(steps)


def create_quick_test():
    """Create a minimal test to verify RL code works."""
    print("\n" + "="*70)
    print("QUICK FUNCTIONALITY TEST")
    print("="*70)
    
    try:
        import torch
        import numpy as np
        
        # Test basic RL components
        print("\nTesting RL components...")
        
        # Import our classes
        sys.path.append(str(Path(__file__).parent))
        from train_rl import QNetwork, CQLAgent
        
        # Create dummy agent
        state_dim = 10
        agent = CQLAgent(state_dim=state_dim, lr=1e-3)
        
        # Test forward pass
        dummy_state = np.random.randn(state_dim)
        action = agent.select_action(dummy_state)
        q_values = agent.get_q_values(dummy_state)
        
        print(f"✓ Agent initialized successfully")
        print(f"  State dimension: {state_dim}")
        print(f"  Selected action: {action}")
        print(f"  Q-values: {q_values}")
        
        # Test training step
        dummy_batch = {
            'states': np.random.randn(32, state_dim),
            'actions': np.random.randint(0, 2, 32),
            'rewards': np.random.randn(32) * 100,
            'next_states': np.random.randn(32, state_dim),
            'dones': np.ones(32)
        }
        
        loss = agent.train_step(dummy_batch)
        print(f"✓ Training step successful")
        print(f"  Loss: {loss['total_loss']:.4f}")
        
        print("\n✓ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Functionality test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("="*70)
    print("RL INTEGRATION SETUP CHECKER")
    print("="*70)
    print("\nThis script checks if your environment is ready for RL training.")
    
    # Run checks
    env_ok = check_environment()
    check_file_structure()
    check_data()
    check_models()
    
    if env_ok:
        test_ok = create_quick_test()
    else:
        print("\n⚠ Fix environment issues before proceeding")
        test_ok = False
    
    # Show configuration example
    generate_example_config()
    
    # Show next steps
    show_next_steps()
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if env_ok and test_ok:
        print("✓ Your environment is ready for RL training!")
        print("✓ Follow the next steps above to proceed.")
    else:
        print("⚠ Please fix the issues mentioned above.")
        print("⚠ Re-run this script after fixing.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()