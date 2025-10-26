"""
compare_models.py
Compare Deep Learning classifier with RL policy agent.
Generate insights for the final report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


class ModelComparator:
    """Compare DL and RL models for loan approval."""
    
    def __init__(self, dl_model, rl_agent, rl_dataset, scaler, feature_names):
        """
        Args:
            dl_model: Trained DL model (PyTorch or sklearn with predict_proba)
            rl_agent: Trained CQL agent
            rl_dataset: Test data dictionary
            scaler: Fitted scaler
            feature_names: List of feature names
        """
        self.dl_model = dl_model
        self.rl_agent = rl_agent
        self.test_data = rl_dataset['test']
        self.scaler = scaler
        self.feature_names = feature_names
        
    def get_dl_predictions(self, threshold=0.5):
        """
        Get DL model predictions.
        
        Args:
            threshold: Probability threshold for classification
            
        Returns:
            actions, probabilities
        """
        states = self.test_data['states']
        
        # Get predictions
        if hasattr(self.dl_model, 'predict_proba'):
            # Sklearn-style model
            probs = self.dl_model.predict_proba(states)[:, 1]
        else:
            # PyTorch model
            self.dl_model.eval()
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states)
                if torch.cuda.is_available():
                    states_tensor = states_tensor.cuda()
                logits = self.dl_model(states_tensor)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
        # Convert to approval decisions
        # If P(default) < threshold, approve (action=1)
        actions = (probs < threshold).astype(int)
        
        return actions, probs
    
    def get_rl_predictions(self):
        """
        Get RL agent predictions.
        
        Returns:
            actions, q_values
        """
        states = self.test_data['states']
        
        actions = []
        q_values_list = []
        
        for state in states:
            action = self.rl_agent.select_action(state, epsilon=0.0)
            q_vals = self.rl_agent.get_q_values(state).flatten()
            actions.append(action)
            q_values_list.append(q_vals)
        
        return np.array(actions), np.array(q_values_list)
    
    def calculate_returns(self, actions):
        """Calculate financial returns for given actions."""
        rewards = self.test_data['rewards']
        # Action 0 (Deny) = 0 return, Action 1 (Approve) = actual reward
        returns = np.where(actions == 1, rewards, 0)
        return returns
    
    def compare_policies(self, dl_threshold=0.5):
        """
        Comprehensive comparison of both policies.
        
        Returns:
            comparison_df, decision_df
        """
        print("\n" + "="*70)
        print("POLICY COMPARISON: DL CLASSIFIER vs RL AGENT")
        print("="*70)
        
        # Get predictions
        dl_actions, dl_probs = self.get_dl_predictions(dl_threshold)
        rl_actions, rl_q_values = self.get_rl_predictions()
        
        # Calculate returns
        dl_returns = self.calculate_returns(dl_actions)
        rl_returns = self.calculate_returns(rl_actions)
        actual_rewards = self.test_data['rewards']
        
        # Summary metrics
        comparison_data = {
            'Metric': [
                'Approval Rate (%)',
                'Avg Return per Loan ($)',
                'Total Return ($)',
                'Loans Approved',
                'Loans Denied'
            ],
            'DL Classifier': [
                dl_actions.mean() * 100,
                dl_returns.mean(),
                dl_returns.sum(),
                int(dl_actions.sum()),
                int((1 - dl_actions).sum())
            ],
            'RL Agent': [
                rl_actions.mean() * 100,
                rl_returns.mean(),
                rl_returns.sum(),
                int(rl_actions.sum()),
                int((1 - rl_actions).sum())
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate improvement
        dl_total = dl_returns.sum()
        rl_total = rl_returns.sum()
        improvement_pct = ((rl_total - dl_total) / abs(dl_total)) * 100 if dl_total != 0 else 0
        
        print(f"\n📊 Summary Statistics:")
        print(comparison_df.to_string(index=False))
        print(f"\n💰 RL Agent Improvement: {improvement_pct:+.2f}% vs DL Classifier")
        
        # Decision-level dataframe
        decision_df = pd.DataFrame({
            'dl_action': dl_actions,
            'rl_action': rl_actions,
            'dl_default_prob': dl_probs,
            'rl_q_deny': rl_q_values[:, 0],
            'rl_q_approve': rl_q_values[:, 1],
            'actual_reward': actual_rewards,
            'dl_return': dl_returns,
            'rl_return': rl_returns,
            'agreement': (dl_actions == rl_actions).astype(int),
            'return_diff': rl_returns - dl_returns
        })
        
        return comparison_df, decision_df
    
    def analyze_disagreements(self, decision_df):
        """Analyze cases where models disagree."""
        
        print("\n" + "="*70)
        print("DISAGREEMENT ANALYSIS")
        print("="*70)
        
        disagreements = decision_df[decision_df['agreement'] == 0]
        agreement_rate = (1 - len(disagreements) / len(decision_df)) * 100
        
        print(f"\n📈 Agreement Rate: {agreement_rate:.1f}%")
        print(f"   Disagreements: {len(disagreements):,} / {len(decision_df):,}")
        
        # RL more aggressive (approves when DL denies)
        rl_aggressive = disagreements[
            (disagreements['dl_action'] == 0) & 
            (disagreements['rl_action'] == 1)
        ]
        
        # RL more conservative (denies when DL approves)
        rl_conservative = disagreements[
            (disagreements['dl_action'] == 1) & 
            (disagreements['rl_action'] == 0)
        ]
        
        print(f"\n🎯 RL More Aggressive: {len(rl_aggressive):,} cases")
        if len(rl_aggressive) > 0:
            print(f"   Avg Default Prob: {rl_aggressive['dl_default_prob'].mean():.3f}")
            print(f"   Avg Actual Reward: ${rl_aggressive['actual_reward'].mean():,.2f}")
            print(f"   Total RL Return: ${rl_aggressive['rl_return'].sum():,.2f}")
            successful = (rl_aggressive['actual_reward'] > 0).sum()
            print(f"   Success Rate: {successful / len(rl_aggressive) * 100:.1f}%")
        
        print(f"\n🛡️  RL More Conservative: {len(rl_conservative):,} cases")
        if len(rl_conservative) > 0:
            print(f"   Avg Default Prob: {rl_conservative['dl_default_prob'].mean():.3f}")
            print(f"   Avg Actual Reward: ${rl_conservative['actual_reward'].mean():,.2f}")
            print(f"   Opportunity Cost: ${rl_conservative['actual_reward'].sum():,.2f}")
            would_succeed = (rl_conservative['actual_reward'] > 0).sum()
            print(f"   Would Have Succeeded: {would_succeed / len(rl_conservative) * 100:.1f}%")
        
        return {
            'rl_aggressive': rl_aggressive,
            'rl_conservative': rl_conservative,
            'agreement_rate': agreement_rate
        }
    
    def find_interesting_cases(self, decision_df, n=5):
        """Find interesting edge cases."""
        
        print("\n" + "="*70)
        print("INTERESTING CASE STUDIES")
        print("="*70)
        
        cases = {}
        
        # Case 1: High-risk but RL approves (and succeeds)
        high_risk_approved = decision_df[
            (decision_df['dl_action'] == 0) &
            (decision_df['rl_action'] == 1) &
            (decision_df['dl_default_prob'] > 0.6) &
            (decision_df['actual_reward'] > 0)
        ].nlargest(n, 'actual_reward')
        
        print(f"\n✅ High-Risk Loans RL Correctly Approved (Top {n}):")
        print(f"   (DL said NO, RL said YES, and they PAID)")
        if len(high_risk_approved) > 0:
            for idx, row in high_risk_approved.iterrows():
                print(f"   • Default Prob: {row['dl_default_prob']:.1%}, "
                      f"Reward: ${row['actual_reward']:,.2f}, "
                      f"Q(Approve): {row['rl_q_approve']:.2f}")
        else:
            print("   None found")
        cases['high_risk_success'] = high_risk_approved
        
        # Case 2: Low-risk but RL denies (and would have succeeded)
        low_risk_denied = decision_df[
            (decision_df['dl_action'] == 1) &
            (decision_df['rl_action'] == 0) &
            (decision_df['dl_default_prob'] < 0.3) &
            (decision_df['actual_reward'] > 0)
        ].nlargest(n, 'actual_reward')
        
        print(f"\n❌ Low-Risk Loans RL Incorrectly Denied (Top {n}):")
        print(f"   (DL said YES, RL said NO, but would have PAID)")
        if len(low_risk_denied) > 0:
            for idx, row in low_risk_denied.iterrows():
                print(f"   • Default Prob: {row['dl_default_prob']:.1%}, "
                      f"Missed Reward: ${row['actual_reward']:,.2f}, "
                      f"Q(Deny): {row['rl_q_deny']:.2f}")
        else:
            print("   None found")
        cases['low_risk_mistake'] = low_risk_denied
        
        # Case 3: Biggest return improvements
        biggest_improvements = decision_df.nlargest(n, 'return_diff')
        print(f"\n💰 Biggest Return Improvements by RL (Top {n}):")
        for idx, row in biggest_improvements.iterrows():
            print(f"   • Improvement: ${row['return_diff']:,.2f}, "
                  f"DL: {row['dl_action']}, RL: {row['rl_action']}, "
                  f"Default Prob: {row['dl_default_prob']:.1%}")
        cases['biggest_improvements'] = biggest_improvements
        
        return cases
    
    def plot_comparison(self, decision_df, save_dir='../models'):
        """Create comprehensive comparison visualizations."""
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Approval rates
        ax1 = fig.add_subplot(gs[0, 0])
        approval_data = pd.DataFrame({
            'Model': ['DL', 'RL'],
            'Approval Rate': [
                decision_df['dl_action'].mean(),
                decision_df['rl_action'].mean()
            ]
        })
        bars = ax1.bar(approval_data['Model'], approval_data['Approval Rate'], 
                       color=['#3498db', '#e74c3c'], alpha=0.7)
        ax1.set_ylabel('Approval Rate', fontsize=11)
        ax1.set_title('Approval Rate Comparison', fontweight='bold', fontsize=12)
        ax1.set_ylim([0, 1])
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom')
        
        # 2. Return distributions
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(decision_df['dl_return'], bins=50, alpha=0.5, 
                label='DL', color='#3498db', density=True)
        ax2.hist(decision_df['rl_return'], bins=50, alpha=0.5, 
                label='RL', color='#e74c3c', density=True)
        ax2.set_xlabel('Return ($)', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Return Distribution', fontweight='bold', fontsize=12)
        ax2.legend()
        
        # 3. Cumulative returns
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(np.cumsum(decision_df['dl_return']), 
                label='DL', linewidth=2, color='#3498db')
        ax3.plot(np.cumsum(decision_df['rl_return']), 
                label='RL', linewidth=2, color='#e74c3c')
        ax3.set_xlabel('Loan Number', fontsize=11)
        ax3.set_ylabel('Cumulative Return ($)', fontsize=11)
        ax3.set_title('Cumulative Returns', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.ticklabel_format(style='plain', axis='y')
        
        # 4. Agreement matrix
        ax4 = fig.add_subplot(gs[1, 0])
        agreement_matrix = pd.crosstab(
            decision_df['dl_action'].map({0: 'DL Deny', 1: 'DL Approve'}),
            decision_df['rl_action'].map({0: 'RL Deny', 1: 'RL Approve'})
        )
        sns.heatmap(agreement_matrix, annot=True, fmt='d', cmap='Blues', 
                   ax=ax4, cbar_kws={'label': 'Count'})
        ax4.set_title('Decision Agreement Matrix', fontweight='bold', fontsize=12)
        
        # 5. Default prob vs Q-values
        ax5 = fig.add_subplot(gs[1, 1])
        scatter = ax5.scatter(decision_df['dl_default_prob'], 
                            decision_df['rl_q_approve'] - decision_df['rl_q_deny'],
                            c=decision_df['rl_action'], cmap='RdYlGn', 
                            alpha=0.5, s=10)
        ax5.set_xlabel('DL Default Probability', fontsize=11)
        ax5.set_ylabel('RL Q(Approve) - Q(Deny)', fontsize=11)
        ax5.set_title('Risk vs RL Preference', fontweight='bold', fontsize=12)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='RL Action')
        ax5.grid(alpha=0.3)
        
        # 6. Return difference by default prob
        ax6 = fig.add_subplot(gs[1, 2])
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        binned_diff = []
        for i in range(len(bins)-1):
            mask = (decision_df['dl_default_prob'] >= bins[i]) & \
                   (decision_df['dl_default_prob'] < bins[i+1])
            binned_diff.append(decision_df[mask]['return_diff'].mean())
        ax6.bar(bin_centers, binned_diff, width=0.08, 
               color=['green' if x > 0 else 'red' for x in binned_diff], alpha=0.7)
        ax6.set_xlabel('Default Probability Bin', fontsize=11)
        ax6.set_ylabel('Avg Return Difference ($)', fontsize=11)
        ax6.set_title('RL Improvement by Risk Level', fontweight='bold', fontsize=12)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.grid(alpha=0.3)
        
        # 7. Q-value distributions
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.hist(decision_df['rl_q_deny'], bins=50, alpha=0.5, 
                label='Q(Deny)', color='red')
        ax7.hist(decision_df['rl_q_approve'], bins=50, alpha=0.5, 
                label='Q(Approve)', color='green')
        ax7.set_xlabel('Q-Value', fontsize=11)
        ax7.set_ylabel('Frequency', fontsize=11)
        ax7.set_title('RL Q-Value Distribution', fontweight='bold', fontsize=12)
        ax7.legend()
        
        # 8. Performance by agreement/disagreement
        ax8 = fig.add_subplot(gs[2, 1])
        agree_returns = decision_df[decision_df['agreement'] == 1]['rl_return'].mean()
        disagree_returns = decision_df[decision_df['agreement'] == 0]['rl_return'].mean()
        bars = ax8.bar(['Models Agree', 'Models Disagree'], 
                      [agree_returns, disagree_returns],
                      color=['#2ecc71', '#f39c12'], alpha=0.7)
        ax8.set_ylabel('Avg Return ($)', fontsize=11)
        ax8.set_title('Returns: Agreement vs Disagreement', fontweight='bold', fontsize=12)
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}', ha='center', va='bottom')
        
        # 9. Total returns comparison
        ax9 = fig.add_subplot(gs[2, 2])
        total_data = pd.DataFrame({
            'Model': ['DL', 'RL'],
            'Total Return': [
                decision_df['dl_return'].sum(),
                decision_df['rl_return'].sum()
            ]
        })
        bars = ax9.bar(total_data['Model'], total_data['Total Return'], 
                      color=['#3498db', '#e74c3c'], alpha=0.7)
        ax9.set_ylabel('Total Return ($)', fontsize=11)
        ax9.set_title('Total Returns Comparison', fontweight='bold', fontsize=12)
        ax9.ticklabel_format(style='plain', axis='y')
        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('DL Classifier vs RL Agent: Comprehensive Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plots saved to: {save_path}")
        
        plt.show()
        
        return fig


def generate_report_insights(comparator, decision_df, disagreement_analysis, interesting_cases):
    """Generate key insights for the final report."""
    
    print("\n" + "="*70)
    print("KEY INSIGHTS FOR REPORT")
    print("="*70)
    
    print("\n📝 1. METRIC DIFFERENCES:")
    print("   • DL Model uses AUC/F1: Measures classification accuracy")
    print("     - Treats all misclassifications equally")
    print("     - Optimizes for predicting default vs non-default")
    print("   • RL Agent uses Policy Value: Measures expected financial return")
    print("     - Considers actual rewards (profit/loss)")
    print("     - Optimizes for maximizing total return")
    
    print("\n📝 2. WHY RL MIGHT APPROVE HIGH-RISK LOANS:")
    high_risk_cases = interesting_cases.get('high_risk_success', pd.DataFrame())
    if len(high_risk_cases) > 0:
        avg_reward = high_risk_cases['actual_reward'].mean()
        print(f"   • Found {len(high_risk_cases)} high-risk loans RL correctly approved")
        print(f"   • Average reward: ${avg_reward:,.2f}")
        print("   • Possible reasons:")
        print("     - High interest rates compensate for risk")
        print("     - RL learned risk-reward tradeoffs, not just risk minimization")
        print("     - Some high-risk features may not indicate default")
    
    print("\n📝 3. POLICY DIFFERENCES:")
    dl_approval = decision_df['dl_action'].mean()
    rl_approval = decision_df['rl_action'].mean()
    print(f"   • DL approval rate: {dl_approval:.1%}")
    print(f"   • RL approval rate: {rl_approval:.1%}")
    if rl_approval > dl_approval:
        print("   • RL is more aggressive (approves more loans)")
        print("   • This suggests RL found profitable opportunities DL missed")
    else:
        print("   • RL is more conservative (approves fewer loans)")
        print("   • This suggests RL avoids some loans DL would approve")
    
    print("\n📝 4. FINANCIAL IMPACT:")
    dl_total = decision_df['dl_return'].sum()
    rl_total = decision_df['rl_return'].sum()
    improvement = rl_total - dl_total
    improvement_pct = (improvement / abs(dl_total)) * 100 if dl_total != 0 else 0
    print(f"   • DL total return: ${dl_total:,.2f}")
    print(f"   • RL total return: ${rl_total:,.2f}")
    print(f"   • Improvement: ${improvement:,.2f} ({improvement_pct:+.1f}%)")
    
    return {
        'dl_approval_rate': dl_approval,
        'rl_approval_rate': rl_approval,
        'dl_total_return': dl_total,
        'rl_total_return': rl_total,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }


if __name__ == "__main__":
    print("="*70)
    print("MODEL COMPARISON SCRIPT")
    print("="*70)
    print("\nThis script compares your trained DL model with the RL agent.")
    print("Make sure you have:")
    print("  1. Trained DL model saved")
    print("  2. Trained RL agent saved")
    print("  3. RL dataset prepared")
    print("\nLoading models...")
    
    # TODO: Load your DL model
    # Example: dl_model = torch.load('../models/dl_model.pth')
    
    # Load RL agent
    from train_rl import CQLAgent
    from prepare_rl_dataset import RLDatasetBuilder
    
    rl_dataset, scaler = RLDatasetBuilder.load_dataset()
    state_dim = rl_dataset['metadata']['state_dim']
    
    rl_agent = CQLAgent(state_dim=state_dim)
    rl_agent.load('../models/cql_loan_agent.pth')
    
    # TODO: Load your DL model here
    # dl_model = ...
    
    # Create comparator
    # feature_names = rl_dataset['metadata']['feature_cols']
    # comparator = ModelComparator(dl_model, rl_agent, rl_dataset, scaler, feature_names)
    
    # Run comparison
    # comparison_df, decision_df = comparator.compare_policies(dl_threshold=0.5)
    # disagreement_analysis = comparator.analyze_disagreements(decision_df)
    # interesting_cases = comparator.find_interesting_cases(decision_df, n=5)
    # comparator.plot_comparison(decision_df)
    # insights = generate_report_insights(comparator, decision_df, disagreement_analysis, interesting_cases)
    
    print("\n✓ Comparison complete!")