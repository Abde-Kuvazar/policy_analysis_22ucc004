"""
train_rl.py
Train an offline RL agent using Conservative Q-Learning (CQL) for loan approval.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from datetime import datetime
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class QNetwork(nn.Module):
    """Q-Network for discrete action space (Deny=0, Approve=1)."""
    
    def __init__(self, state_dim, hidden_dims=[256, 128, 64]):
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output: Q-values for 2 actions [Deny, Approve]
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        return self.network(state)


class CQLAgent:
    """
    Conservative Q-Learning Agent for Offline RL.
    
    CQL prevents overestimation of out-of-distribution actions by
    adding a conservative penalty to the Q-learning objective.
    
    Reference: Kumar et al., "Conservative Q-Learning for Offline RL" (NeurIPS 2020)
    """
    
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, tau=0.005, 
                 alpha=1.0, device=None):
        """
        Args:
            state_dim: Dimension of state space
            lr: Learning rate
            gamma: Discount factor (set to 0.99 even though single-step)
            tau: Soft update parameter for target network
            alpha: CQL penalty coefficient (higher = more conservative)
            device: 'cuda' or 'cpu'
        """
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Q-networks
        self.q_network = QNetwork(state_dim).to(self.device)
        self.q_target = QNetwork(state_dim).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Metrics tracking
        self.training_history = {
            'td_loss': [],
            'cql_loss': [],
            'total_loss': [],
            'q_values': [],
            'q_std': []
        }
        
    def select_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (numpy array)
            epsilon: Exploration rate (0 for pure greedy)
        
        Returns:
            action: 0 (Deny) or 1 (Approve)
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, 2)
        
        with torch.no_grad():
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def get_q_values(self, state):
        """Get Q-values for both actions."""
        with torch.no_grad():
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            state_tensor = torch.FloatTensor(state).to(self.device)
            return self.q_network(state_tensor).cpu().numpy()
    
    def train_step(self, batch):
        """
        Perform one CQL training step.
        
        Args:
            batch: Dictionary with states, actions, rewards, next_states, dones
        
        Returns:
            Dictionary with loss components
        """
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.q_target(next_states)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Temporal Difference loss
        td_loss = nn.MSELoss()(current_q, target_q)
        
        # CQL penalty: encourage lower Q-values for all actions
        # then push up Q-values for actions actually taken in dataset
        logsumexp = torch.logsumexp(current_q_values, dim=1)
        cql_loss = (logsumexp - current_q).mean()
        
        # Total loss
        total_loss = td_loss + self.alpha * cql_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update_target()
        
        # Track metrics
        self.training_history['td_loss'].append(td_loss.item())
        self.training_history['cql_loss'].append(cql_loss.item())
        self.training_history['total_loss'].append(total_loss.item())
        self.training_history['q_values'].append(current_q.mean().item())
        self.training_history['q_std'].append(current_q.std().item())
        
        return {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'total_loss': total_loss.item(),
            'avg_q': current_q.mean().item()
        }
    
    def _soft_update_target(self):
        """Soft update of target network parameters."""
        for target_param, param in zip(self.q_target.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'hyperparameters': {
                'gamma': self.gamma,
                'tau': self.tau,
                'alpha': self.alpha
            }
        }, filepath)
        print(f"✓ Model saved to: {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_history = checkpoint['training_history']
        print(f"✓ Model loaded from: {filepath}")


class OfflineRLTrainer:
    """Trainer for offline RL agent."""
    
    def __init__(self, agent, rl_dataset, batch_size=256):
        self.agent = agent
        self.rl_dataset = rl_dataset
        self.batch_size = batch_size
        
    def train(self, n_epochs=100, eval_frequency=10, verbose=True):
        """
        Train the agent on offline data.
        
        Args:
            n_epochs: Number of epochs
            eval_frequency: Evaluate every N epochs
            verbose: Print progress
        """
        train_data = self.rl_dataset['train']
        n_samples = len(train_data['states'])
        n_batches = n_samples // self.batch_size
        
        print("\n" + "="*60)
        print("TRAINING OFFLINE RL AGENT")
        print("="*60)
        print(f"Epochs: {n_epochs}")
        print(f"Batch size: {self_batch_size}")
        print(f"Batches per epoch: {n_batches}")
        print(f"Total samples: {n_samples:,}")
        print("="*60 + "\n")
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_losses = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                batch = {
                    'states': train_data['states'][batch_indices],
                    'actions': train_data['actions'][batch_indices],
                    'rewards': train_data['rewards'][batch_indices],
                    'next_states': train_data['next_states'][batch_indices],
                    'dones': train_data['dones'][batch_indices]
                }
                
                loss_dict = self.agent.train_step(batch)
                epoch_losses.append(loss_dict)
            
            # Print progress
            if verbose and (epoch + 1) % eval_frequency == 0:
                avg_losses = {
                    k: np.mean([d[k] for d in epoch_losses])
                    for k in epoch_losses[0].keys()
                }
                
                print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Loss: {avg_losses['total_loss']:.4f} "
                      f"(TD: {avg_losses['td_loss']:.4f}, "
                      f"CQL: {avg_losses['cql_loss']:.4f}) | "
                      f"Avg Q: {avg_losses['avg_q']:.2f}")
                
                # Evaluate on test set
                if (epoch + 1) % (eval_frequency * 2) == 0:
                    metrics = self.evaluate()
                    print(f"  → Test Approval Rate: {metrics['approval_rate']:.2%}, "
                          f"Avg Return: ${metrics['avg_return']:.2f}")
        
        print("\n✓ Training complete!\n")
    
    def evaluate(self):
        """Evaluate policy on test set."""
        test_data = self.rl_dataset['test']
        states = test_data['states']
        rewards = test_data['rewards']
        
        # Get policy actions
        actions = np.array([self.agent.select_action(s) for s in states])
        
        # Calculate returns (0 if denied, actual reward if approved)
        policy_returns = np.where(actions == 1, rewards, 0)
        
        metrics = {
            'approval_rate': actions.mean(),
            'avg_return': policy_returns.mean(),
            'total_return': policy_returns.sum(),
            'n_approved': actions.sum(),
            'n_denied': (actions == 0).sum()
        }
        
        return metrics


def plot_training_curves(agent, save_path=None):
    """Plot training metrics."""
    history = agent.training_history
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'], alpha=0.6)
    axes[0, 0].plot(pd.Series(history['total_loss']).rolling(100).mean(), 
                     linewidth=2, label='Moving Avg')
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # TD vs CQL loss
    axes[0, 1].plot(history['td_loss'], alpha=0.5, label='TD Loss')
    axes[0, 1].plot(history['cql_loss'], alpha=0.5, label='CQL Penalty')
    axes[0, 1].set_title('Loss Components', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Q-values
    axes[1, 0].plot(history['q_values'])
    axes[1, 0].set_title('Average Q-Values', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Q-Value')
    axes[1, 0].grid(alpha=0.3)
    
    # Q-value std
    axes[1, 1].plot(history['q_std'])
    axes[1, 1].set_title('Q-Value Standard Deviation', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Std Dev')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    from prepare_rl_dataset import RLDatasetBuilder
    
    # Load RL dataset
    print("Loading RL dataset...")
    rl_dataset, scaler = RLDatasetBuilder.load_dataset(
        dataset_path='../data/rl_loan_dataset.pkl',
        scaler_path='../data/rl_scaler.pkl'
    )
    
    # Initialize agent
    state_dim = rl_dataset['metadata']['state_dim']
    agent = CQLAgent(
        state_dim=state_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=1.0  # CQL penalty coefficient
    )
    
    # Train
    trainer = OfflineRLTrainer(agent, rl_dataset, batch_size=256)
    trainer.train(n_epochs=50, eval_frequency=5, verbose=True)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        if 'rate' in key:
            print(f"{key}: {value:.2%}")
        elif 'return' in key or 'Return' in key:
            print(f"{key}: ${value:,.2f}")
        else:
            print(f"{key}: {value:,}")
    
    # Save model
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/cql_loan_agent.pth'
    agent.save(model_path)
    
    # Plot training curves
    plot_path = '../models/training_curves.png'
    plot_training_curves(agent, save_path=plot_path)
    
    print("\n✓ Training complete! Model and plots saved.")