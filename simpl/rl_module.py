from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.utils import gpu, to_long


class ValueHead(nn.Module):
    """
    Value Head for PPO to estimate the expected return (value) of a prediction
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super(ValueHead, self).__init__()
        # For processing trajectory features
        self.traj_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # For processing prediction features (cls scores and trajectory)
        # Assuming we'll use the best mode's trajectory and its probability
        self.pred_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 1 for cls score + 2 for final position (x,y)
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Combine features and make final prediction
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, actors, pred_out):
        """
        Compute value function for prediction outputs

        Args:
            actors: Actor features from fusion_net [N_actor, d_embed]
            pred_out: Output from pred_net (res_cls, res_reg, res_aux)

        Returns:
            values: Value predictions [batch_size, 1]
        """
        # Extract trajectory features for target vehicles only
        res_cls, res_reg, _ = pred_out

        # Get batch size
        batch_size = len(res_cls)

        # Create value predictions for each batch item (target vehicle)
        values = []
        actor_idx = 0

        for i in range(batch_size):
            # Get actor features for the target vehicle (first actor in each batch)
            actor_feat = self.traj_encoder(actors[actor_idx].unsqueeze(0))

            # Get prediction features for the target vehicle
            cls_probs = res_cls[i][0]  # [num_modes]
            best_mode = torch.argmax(cls_probs)
            best_prob = cls_probs[best_mode]

            # Get the final position of the best trajectory
            traj = res_reg[i][0, best_mode]  # [pred_len, 2]
            final_pos = traj[-1]  # [2]

            # Combine cls probability and final position
            pred_feat = torch.cat([best_prob.unsqueeze(0), final_pos])  # [3]
            pred_feat = self.pred_encoder(pred_feat.unsqueeze(0))  # [1, hidden_dim]

            # Combine actor and prediction features
            combined_feat = torch.cat([actor_feat, pred_feat], dim=1)  # [1, hidden_dim*2]

            # Compute value
            value = self.value_net(combined_feat)
            values.append(value)

            # Update actor index for the next batch item
            actor_idx += len(res_cls[i])

        # Stack values into a tensor
        values = torch.cat(values, dim=0)

        return values


class PPOLoss(nn.Module):
    """
    PPO Loss function for reinforcement learning training
    Treats entire trajectories as single RL steps (bandit-like)
    """
    def __init__(self, config, device):
        super(PPOLoss, self).__init__()
        self.config = config
        self.device = device

        # PPO hyperparameters
        self.clip_param = config.get('ppo_clip_param', 0.2)
        self.value_loss_coef = config.get('ppo_value_loss_coef', 0.5)
        self.entropy_coef = config.get('ppo_entropy_coef', 0.01)

    def forward(self, action_log_probs, values, returns, advantages, old_action_log_probs):
        """
        Calculate PPO loss for entire trajectories as single RL steps (bandit-like)

        Args:
            action_log_probs: Log probabilities of the actions
            values: Value function predictions
            returns: Target values for value function
            advantages: Advantage estimates (in our case, negative of the original losses)
            old_action_log_probs: Log probabilities of the actions from the old policy

        Returns:
            Total loss and a dictionary with component losses
        """
        # Value loss - predict the reward (negative of original losses)
        # Both values and returns should have shape [batch_size]
        value_loss = F.mse_loss(values, returns)

        # Policy loss - simplified for bandit-like setting
        # In a bandit setting, we're not tracking full trajectories, just optimizing
        # the policy to maximize the immediate reward

        # Calculate importance sampling ratio
        ratio = torch.exp(action_log_probs - old_action_log_probs)

        # Calculate surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

        # Take the minimum of the two surrogate objectives
        # This is the pessimistic estimate (PPO clip objective)
        policy_loss = -torch.min(surr1, surr2).mean()

        # Total loss - completely replaces the original supervised loss
        loss = policy_loss + self.value_loss_coef * value_loss

        return loss, {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item()
        }


class RLLossWrapper(nn.Module):
    """
    Wrapper around the original loss function to compute rewards for RL
    """
    def __init__(self, config, device):
        super(RLLossWrapper, self).__init__()
        self.config = config
        self.device = device

        # Import the original loss function for computing rewards
        loss_module_path = config.get('original_loss_fn', 'simpl.av1_loss_fn:LossFunc')
        module_path, class_name = loss_module_path.split(':')
        module = __import__(module_path, fromlist=[class_name])
        loss_class = getattr(module, class_name)
        self.original_loss_fn = loss_class(config, device)

        # Create PPO loss for RL training
        self.ppo_loss = PPOLoss(config, device)

    def forward(self, out, data):
        # Handle RL mode output
        if self.config.get('use_rl', False):
            traj_out, values = out

            # Compute original loss with trajectory output (only for reward calculation)
            orig_loss_out = self.original_loss_fn(traj_out, data)

            # Convert loss to reward for RL (negative loss)
            # Use the total loss directly as a scalar reward
            # Keep as tensor for detach() but use a scalar value
            reward = -orig_loss_out["loss"]  # Negative because we want to maximize reward

            # Create tensors for PPO loss calculation
            # In a bandit-like setting, we use the reward directly as the advantage
            batch_size = values.shape[0]

            # Extract classification probabilities for target vehicles
            res_cls = traj_out[0]  # Classification probabilities [batch_size, num_modes]
            # Note: We could also use trajectory predictions (traj_out[1]) for more sophisticated
            # action probability calculations, but for now we just use the classification probabilities

            # Calculate log probabilities for all modes
            action_log_probs = []
            for i in range(batch_size):
                # Get classification probabilities for the target vehicle
                cls_probs = res_cls[i][0]  # [num_modes]

                # In a multi-modal prediction model, cls_probs represents the probability distribution
                # over different trajectory modes. This is exactly what we need for action probabilities.

                # Find the best mode (highest probability)
                best_mode = torch.argmax(cls_probs)

                # Calculate log probability of the selected mode
                # This is the log probability of the "action" (selecting this trajectory mode)
                log_prob = torch.log(cls_probs[best_mode] + 1e-10)
                action_log_probs.append(log_prob)

            # Stack into tensor
            action_log_probs = torch.stack(action_log_probs)

            # For the first iteration, old_action_log_probs is the same as action_log_probs
            # In a real implementation, we would store these and use them in the next iteration
            old_action_log_probs = action_log_probs.detach()

            # Use the reward as the target for the value function
            # This encourages the value function to predict the expected reward
            # Ensure returns has shape [batch_size] to match values.squeeze() in PPOLoss
            # Use a scalar reward value for all batch items
            reward_scalar = reward.detach().item()
            returns = torch.ones(batch_size, device=self.device) * reward_scalar

            # Use the reward as the advantage
            # This encourages the policy to maximize the reward
            advantages = torch.ones(batch_size, device=self.device) * reward_scalar

            # Compute PPO loss (completely replacing the original loss)
            # Ensure values has shape [batch_size, 1] and returns has shape [batch_size]
            # Squeeze values to match returns shape if needed
            values_flat = values.squeeze(-1) if values.dim() > 1 else values
            ppo_loss, loss_info = self.ppo_loss(
                action_log_probs,
                values_flat,
                returns,
                advantages,
                old_action_log_probs
            )

            # Create loss output dictionary
            loss_out = {
                'loss': ppo_loss,
                'reward': reward_scalar,  # Use scalar value for logging
                'value_loss': loss_info['value_loss'],
                'policy_loss': loss_info['policy_loss']
            }

            return loss_out
        else:
            # Standard supervised learning mode
            return self.original_loss_fn(out, data)
