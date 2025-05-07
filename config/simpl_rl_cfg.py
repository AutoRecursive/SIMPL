import os
import sys
from config.simpl_av2_cfg import AdvCfg as BaseAdvCfg


class AdvCfg(BaseAdvCfg):
    def __init__(self):
        super(AdvCfg, self).__init__()

        # Enable RL
        self.g_cfg['use_rl'] = True

        # PPO hyperparameters
        self.g_cfg['ppo_clip_param'] = 0.2
        self.g_cfg['ppo_epochs'] = 10
        self.g_cfg['ppo_value_loss_coef'] = 0.5
        self.g_cfg['ppo_entropy_coef'] = 0.01
        self.g_cfg['ppo_gamma'] = 0.99
        self.g_cfg['ppo_gae_lambda'] = 0.95
        self.g_cfg['ppo_value_hidden_dim'] = 128
        self.g_cfg['ppo_dropout'] = 0.1
        self.g_cfg['ppo_value_lr'] = 3e-4

    def get_loss_cfg(self):
        loss_cfg = super().get_loss_cfg()
        loss_cfg["loss_fn"] = "simpl.rl_module:RLLossWrapper"
        loss_cfg["original_loss_fn"] = "simpl.av2_loss_fn:LossFunc"
        return loss_cfg
