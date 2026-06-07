# -*- coding: utf-8 -*-
"""
@File    : MOS_SAC.py
@Project : Risk-Aware eVTOL Path Planning
@Author  : DavidLin
@Date    : 2026/06/07
@Contact : davidlin659562@gmail.com
@License : MIT License
@Description:
This module implements the MOSSAC agent for risk-aware eVTOL path planning.
It defines the replay buffer, shared feature encoder, stochastic actor,
twin critic networks, entropy-temperature tuning, policy updates, and
checkpoint save/load utilities.

Main components:
1. ReplayBuffer for storing and sampling reinforcement-learning transitions.
2. Feature_Encoder for fusing radar sequences, vehicle ego-state, casualty-risk
   maps, and meteorological-risk maps into compact state features.
3. Actor network for sampling bounded continuous control actions.
4. Twin Critic networks for estimating soft Q-values.
5. SAC/MOSSAC training utilities, including target-network updates,
   automatic entropy tuning, and model checkpoint management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math


class ReplayBuffer:
    """replay buffer"""
    
    def __init__(self, capacity=100000, device='cuda'):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, obs, action, reward, next_obs, done):
        """add one transition"""
        self.buffer.append((obs, action, reward, next_obs, done))

    @staticmethod
    def _stack_to_tensor(items, device):
        if torch.is_tensor(items[0]):
            return torch.stack(items, dim=0).to(device=device, non_blocking=True).float()
        else:
            return torch.as_tensor(np.stack(items, axis=0), dtype=torch.float32, device=device)
    
    def recent_indices(self, n):
        """return the indices of the most recent n transitions"""
        buffer_len = len(self.buffer)
        n = min(n, buffer_len)
        return list(range(buffer_len - n, buffer_len))

    def sample(self, batch_size, recent_ratio=0.2):
        """sample a batch; recent_ratio controls the proportion of recent transitions sampled preferentially"""
        # buffer_len = len(self.buffer)
        # recent_n = int(batch_size * recent_ratio)
        # recent_n = min(recent_n, buffer_len)
        # random_n = batch_size - recent_n
        
        # indices = set()
        # # 
        # if recent_n > 0:
        #     indices.update(self.recent_indices(recent_n))
        # #  (none)
        # if random_n > 0:
        #     pool = list(set(range(buffer_len)) - indices)
        #     if pool:
        #         indices.update(random.sample(pool, min(random_n, len(pool))))
        # indices = list(indices)
        # # ifbatch_size, 
        # while len(indices) < batch_size:
        #     idx = random.randint(0, buffer_len - 1)
        #     if idx not in indices:
        #         indices.append(idx)
        # # sequence, clockwisesequence
        # indices = sorted(indices)
        # batch = [self.buffer[i] for i in indices]
        # obss, actions, rewards, next_obss, dones = zip(*batch)
        """sample a batch uniformly at random"""
        buffer_len = len(self.buffer)
        indices = random.sample(range(buffer_len), batch_size)
        batch = [self.buffer[i] for i in indices]
        obss, actions, rewards, next_obss, dones = zip(*batch)

        # istoGPU
        obs_batch      = {k: self._stack_to_tensor([o[k] for o in obss],      self.device) for k in obss[0].keys()}
        next_obs_batch = {k: self._stack_to_tensor([o[k] for o in next_obss], self.device) for k in next_obss[0].keys()}
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).unsqueeze(1).to(self.device)

        return obs_batch, actions, rewards, next_obs_batch, dones

    def __len__(self):
        return len(self.buffer)


# ========================================== 1) Shared encoder====================================
class Feature_Encoder(nn.Module):
    """
    Input:
      obs['radar_seq']: [B, 8, 3, 64]     # 8-frame history, 3 channels (distance + mask + relative velocity)
      obs['state_seq']: [B, state_dim]    # 6: [x, y, v, heading, gx, gy]
      obs['casualty_map']: [B, 101, 101]  # casualty risk map (normalized to [0, 1])
      obs['meteor_map']: [B, 101, 101]    # meteorological risk map (normalized to [0, 1])
    Output:
        feature: [B, 256]   # input to the downstream actor and critic networks
    """
    def __init__(self, state_dim=6, radar_in_ch=3, radar_len=64, state_feature_dim=256, hidden_dim=256, map_size=101):
        super().__init__()
        # Extract per-frame radar features with a 1-D CNN.
        self.cnn_out_dim = 128
        self.radar_cnn = nn.Sequential(
            nn.Conv1d(radar_in_ch, 32, kernel_size=5, padding=2, padding_mode='circular'), # [B, 3, 64] -> [B, 32, 64]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, padding_mode='circular'), # [B, 32, 64] -> [B, 64, 64]
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),  # [B, 64, 64] -> [B, 64, 32]
            nn.Flatten(),
            nn.Linear(64 * 32, self.cnn_out_dim), # [B, 64*32] -> [B, cnn_out_dim]  
            nn.ReLU()
        )
        # Process the radar feature sequence with a GRU.
        self.radar_gru = nn.GRU(input_size=self.cnn_out_dim, hidden_size=128, batch_first=True) # Input[B,8,cnn_out_dim], Output[B,8,128]
        self.attn_fc = nn.Linear(128, 1)  # attention-score generation layer

        # ----- Casualty-risk and meteorological-risk map CNN encoder -----
        # Concatenate the two 101x101 risk maps and encode them into a 128-D feature.
        self.casualty_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # [B,2,101,101] -> [B,16,51,51]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B,16,51,51] -> [B,32,26,26]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B,32,26,26] -> [B,64,13,13]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # [B,64,13,13] -> [B,128,7,7]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B,128,7,7] -> [B,128,1,1]
            nn.Flatten(),  # [B,128,1,1] -> [B,128]
        )

        # ----- ego-state feature encoding -----
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), # [B,6] -> [B, hidden_dim]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # [B, hidden_dim] -> [B, hidden_dim]
            nn.ReLU(),
            nn.Linear(hidden_dim, 128), # [B, hidden_dim] -> [B, 128]
            nn.ReLU(),
        )

        # (after fusion) Output action-distribution parameters: Radar(128) + State(128) + Casualty(128) = 384
        self.fc = nn.Sequential(
            nn.Linear(128 + 128 + 128, hidden_dim), # [B, 384] -> [B, hidden_dim]
            nn.ReLU(),
            nn.Linear(hidden_dim, state_feature_dim), # [B, hidden_dim] -> [B, 256]
            nn.ReLU()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        if isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, radar_seq, state, casualty_map, meteor_map):
        # radar_seq: [B, 8, 3, 64]
        # state: [B, state_dim]
        # casualty_map: [B, 101, 101]
        if radar_seq.dim() == 3:  # [8,3,64] -> [1,8,3,64]
            radar_seq = radar_seq.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if casualty_map.dim() == 2:  # [101,101] -> [1,101,101]
            casualty_map = casualty_map.unsqueeze(0)
        if meteor_map.dim() == 2:  # [101,101] -> [1,101,101]
            meteor_map = meteor_map.unsqueeze(0)

        # 1. Radar feature extraction
        B, T, C, L = radar_seq.shape  # B=batch, T=8, C=3, L=64
        radar_seq = radar_seq.reshape(B * T, C, L)  # [B*T,3,64]
        cnn_feat = self.radar_cnn(radar_seq)        # [B*T, cnn_out_dim]
        cnn_feat = cnn_feat.view(B, T, -1)          # [B,8,cnn_out_dim] CNN
        gru_out, _ = self.radar_gru(cnn_feat)       # [B,8,128] GRU
        attn_scores = self.attn_fc(gru_out)         # [B,8,1] attention scores for each time step
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B,8,1] attention weights,normalization
        radar_feat = torch.sum(attn_weights * gru_out, dim=1)  # [B,128] weighted sum
        
        # 2. State feature extraction
        sf = self.state_mlp(state)                  # [B,128]

        # 3. Casualty Map + Meteor Map feature extraction (compute only once)
        casualty_input = casualty_map[0].unsqueeze(0).unsqueeze(0)  # [101,101] -> [1,1,101,101]
        meteor_input = meteor_map[0].unsqueeze(0).unsqueeze(0)      # [101,101] -> [1,1,101,101]
        risk_input = torch.cat([casualty_input, meteor_input], dim=1)  # [1,2,101,101]
        cf_single = self.casualty_cnn(risk_input)  # [1,128]
        cf = cf_single.expand(B, -1)  # [1,128] -> [B,128]

        # 4. feature fusion: Radar(128) + State(128) + Casualty(128) = 384
        state_feature = self.fc(torch.cat([radar_feat, sf, cf], dim=1))  # [B, feature_dim=256]
        return state_feature


# ========= 2) Actor Head: takes only Feature_Encoder features, Output Gaussian distribution policy =========
class Actor(nn.Module):
    """actor policy network (Actor) OutputA(state_feature)

    Input:
        Feature_EncoderOutput state_feature: [B, feature_dim=256]

    Output:
      mean, log_std used forGaussian distribution;  mean [B, action_dim],  log value of the variance [B, action_dim]
      sample() performs tanh-squash with log_prob correction
      
    """

    def __init__(self, state_feature_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),  # [B,256] -> [B, hidden_dim]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # [B, hidden_dim] -> [B, hidden_dim]
            nn.ReLU()
        )

        # Output
        self.mean_head = nn.Linear(hidden_dim, action_dim)  
        self.log_std_head = nn.Linear(hidden_dim, action_dim) 
        
        # action range, map to the desired range
        self.register_buffer('action_scale', torch.FloatTensor([ 0.16, 0.14]))  # scale factor
        self.register_buffer('action_bias', torch.FloatTensor([0.0, 0.0])) 
        
        # numerical-stability parameter
        self.log_std_min = -20  
        self.log_std_max = 2 
        self.epsilon = 1e-6
        
        # weight initialization
        self.apply(self._init_weights) 
    
    def _init_weights(self, m):
        """weight initialization"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state_feature):
        """forward pass"""
        # state_feature: [B, feature_dim=256]
        x = self.backbone(state_feature)  # [B, feature_dim=256] -> [B, hidden_dim]
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max) 
        return mean, log_std

    def sample(self, state_feature):
        """sample an action"""
        mean, log_std = self.forward(state_feature)
        std = log_std.exp()
        
        # reparameterization trick
        normal = torch.randn_like(mean) 
        x_t = mean + std * normal # sample an action
        
        # Tanh squashing
        action = torch.tanh(x_t) # temporarily adjust to [-1, 1] range

        # compute log probability
        log_prob = torch.distributions.Normal(mean, std).log_prob(x_t) -\
                   torch.log(1 - action.pow(2) + self.epsilon) 
        log_prob = log_prob.sum(1, keepdim=True) #[batch_size, action_dim] -> [batch_size, 1]
        
        # scale to the actual action range
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob, torch.tanh(mean) * self.action_scale + self.action_bias  
        


# ========= 3) Critic: takes Feature_Encoder features with a, Output twin Q =========
class Critic(nn.Module):
    """Qnetwork (Critic), Q (state_feature,a)"""

    def __init__(self, state_feature_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1network
        self.q1 = nn.Sequential(
            nn.Linear(state_feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2network
        self.q2 = nn.Sequential(
            nn.Linear(state_feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m): 
        """weight initialization"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state_feature, action):
        """forward pass"""
        sa = torch.cat([state_feature, action], 1)  

        return self.q1(sa), self.q2(sa)

    def Q1(self, state_feature, action):
        """return only the Q1 value"""
        sa = torch.cat([state_feature, action], 1)

        return self.q1(sa)



class SAC:
    """Soft Actor-Criticalgorithm"""
    
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 radar_in_ch=3,
                 radar_len=64,
                 state_feature_dim=256,
                 map_size=101,  # map size
                 lr=4e-4, # actor learning rate 
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2, # entropy coefficient, control the exploration level
                 auto_entropy_tuning=True, # whether to tune automatically entropy coefficient
                 hidden_dim=256,
                 device='cuda'): 
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning

        # network initialize

        # ---- shared encoder & its target copy ----
        self.encoder        = Feature_Encoder(state_dim, radar_in_ch, radar_len, state_feature_dim, hidden_dim, map_size).to(device)
        self.encoder_target = Feature_Encoder(state_dim, radar_in_ch, radar_len, state_feature_dim, hidden_dim, map_size).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # network1: Actor policy network
        self.actor = Actor(state_feature_dim, action_dim, hidden_dim).to(device)

        # network2: Critic contains Q1 and Q2 network
        self.critic = Critic(state_feature_dim, action_dim, hidden_dim).to(device)  # Q1 and Q2

        # network3: Critic_target contains Q1_target and Q2_target network
        self.critic_target = Critic(state_feature_dim, action_dim, hidden_dim).to(device)  # Q1_target and Q2_target
        self.critic_target.load_state_dict(self.critic.state_dict())

        
        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 4e-5},# previouslyis1e-4
            {'params': self.critic.parameters(),  'lr': 4e-4},# previouslyis3e-4
        ])


        # automatic entropy tuning
        if auto_entropy_tuning:
            self.target_entropy = -action_dim  
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)  
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr) 
            self.alpha = self.log_alpha.exp()  

        print(f"SA initialized, device: {device}")

        print(f"Actor parameter count: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"Critic parameter count: {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"Feature Encoder parameter count: {sum(p.numel() for p in self.encoder.parameters()):,}")

    
    def state_feature_from_obs(self, obs, use_target_encoder=False):
        """extract state_feature from obs, radar and mask are temporal inputs, stateuse only the current frame, casualty_mapas a global map"""
        # obs: dict with 'radar_seq' [B,8,3,64], 'mask_seq' [B,8,3,64], 'state_seq' [B,6], 'casualty_map' [B,101,101]
        radar = torch.as_tensor(obs['radar_seq'], dtype=torch.float32, device=self.device)  # [B,8,64]
        mask  = torch.as_tensor(obs['mask_seq'], dtype=torch.float32, device=self.device)  # [B,8,64]
        rel_speeds = torch.as_tensor(obs['rel_speeds_seq'], dtype=torch.float32, device=self.device)  # [B,8,64]
        state = torch.as_tensor(obs['state_seq'], dtype=torch.float32, device=self.device)  # [B,6]
        casualty_map = torch.as_tensor(obs['casualty_map'], dtype=torch.float32, device=self.device)  # [B,101,101]
        meteor_map = torch.as_tensor(obs['meteor_map'], dtype=torch.float32, device=self.device)  # [B,101,101]

        # compatible with a single sampleInput
        if radar.dim() == 2:  # [8, 64] -> [1,8,64]
            radar = radar.unsqueeze(0)
            mask = mask.unsqueeze(0)
            rel_speeds = rel_speeds.unsqueeze(0)
        if state.dim() == 1:  # [6] -> [1,6]
            state = state.unsqueeze(0)
        if casualty_map.dim() == 2:  # [101,101] -> [1,101,101]
            casualty_map = casualty_map.unsqueeze(0)
        if meteor_map.dim() == 2:  # [101,101] -> [1,101,101]
            meteor_map = meteor_map.unsqueeze(0)

        # process only the current framestate: [B,6] convert to relative relationships
        x = state[..., 0]
        y = state[..., 1]
        v = state[..., 2]
        psi = state[..., 3] * (2 * math.pi) # left-closed and right-open interval[0,2pi)
        gx = state[..., 4]
        gy = state[..., 5]

        dx = gx - x
        dy = gy - y
        r = torch.sqrt(dx ** 2 + dy ** 2)
        theta_g = torch.atan2(dy, dx) # goal angle in the global coordinate frame, range[-pi,pi)
        # 2. heading angle psi map to [-π, π) range
        psi = torch.atan2(torch.sin(psi), torch.cos(psi))  # transform to[-pi,pi)range
        beta = torch.atan2(torch.sin(theta_g - psi), torch.cos(theta_g - psi)) # goal angle in the robot coordinate frame, range[-pi,pi)
        # if beta > 0 the goal is on the left, beta < 0 the goal is on the right  
        sin_beta = torch.sin(beta)
        cos_beta = torch.cos(beta)
        sin_psi = torch.sin(psi)
        cos_psi = torch.cos(psi)

        # new state: [B,6] [r, sin_beta, cos_beta, v, sin_psi, cos_psi]
        state_new = torch.stack([r, sin_beta, cos_beta, v, sin_psi, cos_psi], dim=-1)

        # mergeradarandmask: [B,8,3,64]
        radar_3ch = torch.stack([radar, mask, rel_speeds], dim=2)  # [B,8,3,64]

        enc = self.encoder_target if use_target_encoder else self.encoder
        state_feature = enc(radar_3ch, state_new, casualty_map, meteor_map)  # Return[B, z_dim(256)]
        return state_feature

    def select_action(self, obs, deterministic=False):
        """ actornetwork select action"""
        with torch.no_grad():
            state_feature = self.state_feature_from_obs(obs, use_target_encoder=False)  
            # state features, uses encoder network, encoder goal network only in critic evaluation use

            if deterministic:  # evaluation use mean (without noise)
                _, _, action = self.actor.sample(state_feature)
            else:  # training use sample an action (with noise)
                action, _, _ = self.actor.sample(state_feature)
            
            return action.cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size=256):
        """update networksparameters"""

        return self.update_simplified(replay_buffer, batch_size)

    def update_simplified(self, replay_buffer, batch_size):
        """MOSSAC update (without using a V network)"""
        # sample a batch
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)
        
        # compute targetQvalue
        with torch.no_grad():
            state_feature_next = self.state_feature_from_obs(next_obs, use_target_encoder=True)  # s' state features, uses encoder_target
            next_action, next_log_prob, _ = self.actor.sample(state_feature_next) # s' next action under the state a'
            next_q1, next_q2 = self.critic_target(state_feature_next, next_action) # compute Q1_target(s', a') and Q2_target(s', a')
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob # take the smaller Q(s', a') + corresponding information entropy, replace previously V(s')
            target_q = reward + (1 - done.float()) * self.gamma * next_q # r + γ * (Q(s',a') - α*log π(a'|s')) as Q actual value (target value)

        # current Q value
        state_feature = self.state_feature_from_obs(obs, use_target_encoder=False)  # s state features, uses encoder network
        current_q1, current_q2 = self.critic(state_feature, action) # Q(s, a) computed value (current value)

        # Criticloss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        # critic_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)
        
        # update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss (use detach avoid backpropagating gradients Critic)
        with torch.no_grad():
            state_feature_detach = state_feature.detach()
        new_action, log_prob, _ = self.actor.sample(state_feature_detach)
        q1_new, q2_new = self.critic(state_feature_detach, new_action)
        q_new = torch.min(q1_new, q2_new)  # recommended to use two Q network minimum value
        actor_loss = (self.alpha * log_prob - q_new).mean()   # loss = entropy + Q

        # update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update the temperature parameter
        alpha_loss = torch.tensor(0.0)  # default value, avoid undefined values
        if self.auto_entropy_tuning:

            alpha_loss = (self.log_alpha.exp() * (-log_prob - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # updategoalnetwork
        self._soft_update(self.critic_target, self.critic, self.tau)
        self._soft_update(self.encoder_target, self.encoder, self.tau)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            'alpha': self.alpha.item() if torch.is_tensor(self.alpha) else self.alpha
        }
    
    def _soft_update(self, target_net, source_net, tau):
        """updategoalnetwork"""
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def save(self, filepath):
        """save model"""
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "encoder_target": self.encoder_target.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
        }
        
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to: {filepath}")
    
    def load(self, filepath):
        """loadmodel"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.encoder_target.load_state_dict(checkpoint["encoder_target"])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        
        if 'alpha' in checkpoint:
            self.alpha = checkpoint['alpha']
        if 'log_alpha' in checkpoint and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
        
        print(f"Model loaded from {filepath} loaded")


def test_sac():
    """Testing the SAC algorithmMOSSACand the original implementation"""
    print("Testing the SAC algorithm...")

    # devicecheck
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create the SAC instance
    radar_num_points = 64
    state_dim = 6
    action_dim = 2

    print("\n=== testMOSSAC ===")
    sac_simplified = SAC(state_dim=state_dim,
                         action_dim=action_dim,
                         radar_in_ch=3,
                         radar_len=radar_num_points,
                         device=device)

    # createreplay buffer
    replay_buffer = ReplayBuffer(capacity=10000, device=device)

    # generate fixedrisk mapandmeteorological risk map (simulated prior map)
    fixed_casualty_map = np.random.rand(101, 101).astype(np.float32)
    fixed_meteor_map = np.random.rand(101, 101).astype(np.float32)

    # generate dummy data (dictionary format)
    for _ in range(1000):
        obs = {
            'radar_seq': np.random.rand(8, radar_num_points).astype(np.float32),
            'mask_seq': np.random.randint(0, 2, size=(8, radar_num_points)).astype(np.float32),
            'rel_speeds_seq': np.random.randn(8, radar_num_points).astype(np.float32),
            'state_seq': np.random.randn(state_dim).astype(np.float32),
            'casualty_map': fixed_casualty_map,  # use the samerisk map
            'meteor_map': fixed_meteor_map  # use the samemeteorological risk map
        }
        action = np.random.randn(action_dim).astype(np.float32)
        reward = np.random.randn()
        next_obs = {
            'radar_seq': np.random.rand(8, radar_num_points).astype(np.float32),
            'mask_seq': np.random.randint(0, 2, size=(8, radar_num_points)).astype(np.float32),
            'rel_speeds_seq': np.random.randn(8, radar_num_points).astype(np.float32),
            'state_seq': np.random.randn(state_dim).astype(np.float32),
            'casualty_map': fixed_casualty_map,  # use the samerisk map
            'meteor_map': fixed_meteor_map  # use the samemeteorological risk map
        }
        done = np.random.choice([True, False])
        replay_buffer.push(obs, action, reward, next_obs, done)

    # test training
    print("\n=== trainingMOSSAC ===")
    for i in range(5):
        loss_info = sac_simplified.update(replay_buffer, batch_size=64)
        print(f"step {i+1}: {loss_info}")

    # test action selection
    print("\n=== Action-selection test ===")
    test_obs = {
        'radar_seq': np.random.rand(8, radar_num_points).astype(np.float32),
        'mask_seq': np.random.randint(0, 2, size=(8, radar_num_points)).astype(np.float32),
        'rel_speeds_seq': np.random.randn(8, radar_num_points).astype(np.float32),
        'state_seq': np.random.randn(state_dim).astype(np.float32),
        'casualty_map': fixed_casualty_map,  # use the samerisk map
        'meteor_map': fixed_meteor_map  # use the samemeteorological risk map
    }
    action_simplified = sac_simplified.select_action(test_obs)

    print(f"input state shapes: {[v.shape for v in test_obs.values()]}")
    print(f"MOSSAC action: {action_simplified}")

    print("\nSAC algorithm test completed!")


if __name__ == "__main__":
    test_sac()
