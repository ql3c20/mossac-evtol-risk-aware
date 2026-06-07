# -*- coding: utf-8 -*-
"""
@File    : train_MOS_SAC_tensorboard.py
@Project : Risk-Aware eVTOL Path Planning
@Author  : DavidLin
@Date    : 2026/06/07
@Contact : davidlin659562@gmail.com
@License : MIT License
@Description:
This module provides the training, evaluation, visualization, and logging
entry points for the MOSSAC risk-aware eVTOL path-planning framework.
It loads the default obstacle, casualty-risk, and meteorological-risk maps,
constructs the Gym-style environment, initializes the MOSSAC agent, and
records training metrics through TensorBoard and CSV files.

Main components:
1. Trainer for running the MOSSAC training loop and checkpointing.
2. Training-curve plotting and history export utilities.
3. Model evaluation functions for deterministic policy testing.
4. Statistical testing utilities for success, collision, and failure analysis.
5. Path-logging mode for recording trajectories and reward components.

Scenario selection：
# "static": static obstacles only, no dynamic obstacles
# "cross": static obstacles + cross-route dynamic obstacles
# "parallel": static obstacles + parallel-route dynamic obstacles
# "both": static obstacles + cross-route and parallel-route dynamic obstacles
# "random": randomly choose "cross", "parallel", or "both" at each reset (default training mode)
# "random_test": random extra static/dynamic obstacles + randomly choose "cross", "parallel", or "both"
# "random_test_both": random extra static/dynamic obstacles + "both"
# "fixed_test_both": fixed extra static obstacles + random extra dynamic obstacles + "both"
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time
import os
import psutil
import gc
import datetime
from torch.utils.tensorboard import SummaryWriter  

from env_MOSSAC import SimplePathPlanningEnv
from MOS_SAC import SAC, ReplayBuffer
from generate_grid_map import generate_grid_map
from env_MOSSAC import calculate_distance, calculate_reward_components


class SimpleTrainer:
    """MOSSACtrainer"""
    
    def __init__(self, 
                 env,  
                 agent, 
                 replay_buffer,  
                 max_episodes=1000,  
                 max_steps=500,  # maximum steps per episode
                 batch_size=256,  # batch size for each update
                 update_after=1000, 
                 update_every=50,  # network update interval 
                 save_every=100,  
                 render=False,  
                 render_freq=10): 
        
        self.env = env  # environment object
        self.agent = agent  # SAC agent
        self.replay_buffer = replay_buffer  # replay buffer
        
        self.max_episodes = max_episodes  
        self.max_steps = max_steps  
        self.batch_size = batch_size  
        self.update_after = update_after  
        self.update_every = update_every  
        self.save_every = save_every  
        self.render = render  
        self.render_freq = render_freq 
        
        # training statistics
        self.episode_rewards = []  
        self.episode_lengths = []  
        self.success_rate = []  
        self.losses = {'critic': [], 'actor': [], 'alpha': []}  # loss records (Q, policy, entropy)
        
        # learning-rate decay settings
        self.lr_decay_patience = 50  
        self.lr_decay_threshold = 0.01  # success-rate improvement threshold
        self.last_decay_episode = 0  
        self.best_recent_success_rate = 0.0  
        
        # create output directories
        self.run_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_dir = f'models/{self.run_datetime}'
        self.fig_dir = f'figures/{self.run_datetime}' 
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=f'runs/{self.run_datetime}')

        print(f"Trainer initialized")
        print(f"maximum episodes: {max_episodes}")
        print(f"batch size: {batch_size}")
        print(f"update frequency: every {update_every}step")
        if render:
            print(f"real-time visualization: enabled (every {render_freq} episode render)")
        else:
            print(f"real-time visualization: disabled")
    
    def train(self):
        """main training loop"""
        print("\n" + "="*60)
        print("start MOSSA training")
        print("="*60)
        
        total_steps = 0
        start_time = time.time()
        best_success_rate = 0.0
        best_recent_avg_reward = -float('inf')
        
        for episode in range(self.max_episodes):
            episode_start_time = time.time()
            episode_reward = 0
            episode_length = 0
            
            # determine whether this episode should be rendered
            should_render = (self.render and 
                           (episode % self.render_freq == 0 or 
                            episode < 5 or  
                            episode >= self.max_episodes - 5)) 
            
            # reset the environment
            obs = self.env.reset()
            
            if should_render:
                print(f"\n=== rendered episode {episode} ===")
            
            for step in range(self.max_steps):
                # select action
                if total_steps < self.update_after:
                    # random exploration phase
                    action = self.env.action_space.sample() 
                else:
                    # select action with the policy
                    action = self.agent.select_action(obs)  

                # execute action
                next_obs, reward, done, info = self.env.step(action)

                # real-time visualization
                if should_render:
                    self.env.render()
                    if step < 50 or step % 10 == 0:  
                        time.sleep(0.05)  
                
                # store transition
                self.replay_buffer.push(obs, action, reward, next_obs, done)
                
                # update statistics
                episode_reward += reward
                episode_length += 1  
                total_steps += 1  
                
                # update networks
                if (total_steps >= self.update_after and 
                    total_steps % self.update_every == 0 and 
                    len(self.replay_buffer) >= self.batch_size):
                    
                    # perform multiple updates
                    for _ in range(self.update_every):
                        loss_info = self.agent.update(self.replay_buffer, self.batch_size)
                        
                        # record losses
                        self.losses['critic'].append(loss_info['critic_loss'])
                        self.losses['actor'].append(loss_info['actor_loss'])
                        self.losses['alpha'].append(loss_info['alpha_loss'])

                # episode end
                if done:
                    # record success rate
                    success = info.get('success', False)
                    self.success_rate.append(1 if success else 0)
                    
                    if should_render:
                        reason = info.get('reason', 'unknown')
                        print(f"episode end: {reason}, reward: {episode_reward:.2f}, steps: {episode_length}")
                        if episode < 5:  
                            input("press Enter to continue...")
                    break
                
                obs = next_obs
            
            # record episode
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            episode_time = time.time() - episode_start_time

            # TensorBoard record
            # episode TensorBoard record
            self.writer.add_scalar('Reward/episode', episode_reward, episode)
            self.writer.add_scalar('Success/episode', self.success_rate[-1], episode)
            self.writer.add_scalar('Length/episode', episode_length, episode)
            # step TensorBoard record
            if self.losses['critic']:
                self.writer.add_scalar('Loss/critic_step', self.losses['critic'][-1], total_steps)
            if self.losses['actor']:
                self.writer.add_scalar('Loss/actor_step', self.losses['actor'][-1], total_steps)
            if self.losses['alpha']:
                self.writer.add_scalar('Loss/alpha_step', self.losses['alpha'][-1], total_steps)


            # adaptive learning-rate decay policy - based on performance improvement
            if episode > 200:  # train at least 200 episodes before considering decay
                # compute current success rate
                if len(self.success_rate) <= 100:
                    current_success_rate = np.mean(self.success_rate) if self.success_rate else 0
                else:
                    current_success_rate = np.mean(self.success_rate[-100:])
                
                # check whether learning-rate decay is needed
                episodes_since_decay = episode - self.last_decay_episode
                success_rate_improvement = current_success_rate - self.best_recent_success_rate # success rate degrees

                # condition1: distance last decay was more than patience episode
                # condition2: success rate improvement is below the threshold (performance plateau)
                # condition3: current success rate has reached a certain level (avoid premature decay)
                if (episodes_since_decay >= self.lr_decay_patience and
                    success_rate_improvement < self.lr_decay_threshold and 
                    current_success_rate > 0.99):
                    
                    decay_factor = 0.9  # decay factor, more aggressive than a fixed schedule
                    # actor learning rate
                    for param_group in self.agent.actor_optimizer.param_groups:
                        param_group['lr'] *= decay_factor

                    # critic learning rate
                    for param_group in self.agent.critic_optimizer.param_groups:
                        param_group['lr'] *= decay_factor
                    

                    self.last_decay_episode = episode
                    alpha_lr = self.agent.alpha_optimizer.param_groups[0]['lr'] if hasattr(self.agent, 'alpha_optimizer') else 0
                    print(f"performance-plateau detection, learning rate decayed! Actor LR: {self.agent.actor_optimizer.param_groups[0]['lr']:.6f}, Alpha LR: {alpha_lr:.6f}")

                # update the best success rate record
                if current_success_rate > self.best_recent_success_rate:
                    self.best_recent_success_rate = current_success_rate

            # continuous 50 freeze after all episodes succeed encoder parameters
            if len(self.success_rate) >= 50:
                recent_success_rate = np.mean(self.success_rate[-50:])
                if recent_success_rate == 1.0 and (not hasattr(self, 'encoder_frozen') or not self.encoder_frozen):
                    for param in self.agent.encoder.parameters():
                        param.requires_grad = False
                    self.encoder_frozen = True
                    print("Encoder has learned sufficient representations, frozen (continuous 50 episodes succeeded)!")
            # continuous 20 success rate 0.5 then unfreeze encoder parameters
            if hasattr(self, 'encoder_frozen') and self.encoder_frozen and len(self.success_rate) >= 20:
                recent_success_rate_20 = np.mean(self.success_rate[-20:])
                if recent_success_rate_20 < 0.5:
                    for param in self.agent.encoder.parameters():
                        param.requires_grad = True
                    self.encoder_frozen = False
                    print("Critical learning failure, Encoder unfrozen (continuous 20 success rate lower than 50%)!")

            
            # print progress
            if episode % 10 == 0 or episode < 10:
                if len(self.success_rate) <= 100:
                    current_success_rate = np.mean(self.success_rate) if self.success_rate else 0
                else:
                    current_success_rate = np.mean(self.success_rate[-100:])
                recent_avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards) 
                print(f"episode {episode:4d} | "
                      f"reward: {episode_reward:8.2f} | "
                      f"average reward: {recent_avg_reward:8.2f} | "
                      f"average success rate: {current_success_rate*100:.1f}% | "
                      f"steps: {episode_length:3d} | "
                      f"current episode: {episode_time:.2f}s | ")

            # periodic save
            if episode % self.save_every == 0 and episode > 0:
                self.save_checkpoint(episode)
                # clean up GPU

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        print(f"\ntraining completed!")
        self.save_checkpoint('final')
        self.save_history_csvs() 
        self.plot_training_curves()  
        self.writer.close()  

        return best_success_rate

    def plot_training_curves(self):
        """plot training curves"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # episodereward
            if self.episode_rewards:
                axes[0, 0].plot(self.episode_rewards, alpha=0.6, color='blue', linewidth=1)
                if len(self.episode_rewards) > 20:
                    # average
                    window = min(10, len(self.episode_rewards) // 4)
                    moving_avg = []
                    for i in range(len(self.episode_rewards)):
                        start = max(0, i - window + 1)
                        moving_avg.append(np.mean(self.episode_rewards[start:i+1]))
                    axes[0, 0].plot(moving_avg, color='red', linewidth=2, label=f'{window}-Episode MA')
                    axes[0, 0].legend()
            
            axes[0, 0].set_title('Episode Reward')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # success rate
            if len(self.success_rate) > 0:
                success_data = list(self.success_rate)
                episodes = list(range(len(self.episode_rewards) - len(success_data), len(self.episode_rewards)))
                # success rate convert to percentage
                success_percent = [v * 100 for v in success_data]
                axes[0, 1].plot(episodes, success_percent, alpha=0.4, color='green')

                # success rate average 
                window = 100
                moving_avg = []
                avg_episodes = []
                for i in range(len(success_percent)):
                    if i < window:
                        avg = np.mean(success_percent[:i+1])
                    else:
                        avg = np.mean(success_percent[i-window+1:i+1])
                    moving_avg.append(avg)
                    avg_episodes.append(episodes[i])
                axes[0, 1].plot(avg_episodes, moving_avg, color='darkgreen', linewidth=2, label=f'{window}-Episode MA')
                axes[0, 1].legend()
            
            axes[0, 1].set_title('Success Rate (%)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # episode length
            if self.episode_lengths:
                axes[1, 0].plot(self.episode_lengths, alpha=0.6, color='orange')
                if len(self.episode_lengths) > 20:
                    window = min(10, len(self.episode_lengths) // 4)
                    moving_avg = []
                    for i in range(len(self.episode_lengths)):
                        start = max(0, i - window + 1)
                        moving_avg.append(np.mean(self.episode_lengths[start:i+1]))
                    axes[1, 0].plot(moving_avg, color='darkorange', linewidth=2, label=f'{window}-Episode MA')
                    axes[1, 0].legend()
            
            axes[1, 0].set_title('Episode Length')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].grid(True, alpha=0.3)

            # loss function - use twin axis
            if any(self.losses.values()):
                # left y axis: critic and actor loss
                ax1 = axes[1, 1]
                for loss_name in ['critic', 'actor']:
                    loss_values = self.losses[loss_name]
                    if loss_values and len(loss_values) > 0:
                        # smooth the loss
                        if len(loss_values) > 100:
                            window = max(10, len(loss_values) // 50)
                            smoothed = []
                            for i in range(0, len(loss_values), window):
                                end_idx = min(i + window, len(loss_values))
                                smoothed.append(np.mean(loss_values[i:end_idx]))
                            x_coords = np.linspace(0, len(loss_values), len(smoothed))
                            ax1.plot(x_coords, smoothed, label=loss_name, alpha=0.8)
                        else:
                            ax1.plot(loss_values, label=loss_name, alpha=0.8)
                
                ax1.set_xlabel('Update Steps')
                ax1.set_ylabel('Critic & Actor Loss', color='black')
                ax1.tick_params(axis='y', labelcolor='black')
                ax1.legend(loc='upper left')
                ax1.grid(True, alpha=0.3)

                # right y axis: alpha loss
                ax2 = ax1.twinx()
                alpha_values = self.losses['alpha']
                if alpha_values and len(alpha_values) > 0:
                    if len(alpha_values) > 100:
                        window = max(10, len(alpha_values) // 50)
                        smoothed = []
                        for i in range(0, len(alpha_values), window):
                            end_idx = min(i + window, len(alpha_values))
                            smoothed.append(np.mean(alpha_values[i:end_idx]))
                        x_coords = np.linspace(0, len(alpha_values), len(smoothed))
                        ax2.plot(x_coords, smoothed, label='alpha', color='red', alpha=0.8)
                    else:
                        ax2.plot(alpha_values, label='alpha', color='red', alpha=0.8)
                
                ax2.set_ylabel('Alpha Loss', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.legend(loc='upper right')
                
                axes[1, 1].set_title('Training Losses (Dual Y-axis)')
            
            plt.tight_layout()
            fig_path = f'{self.fig_dir}/training_curves.png'  
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()  
        except Exception as e:
            print(f"plot training curves error: {e}")

    def save_checkpoint(self, episode):
        """save checkpoint"""
        checkpoint_path = f'{self.model_dir}/mossac_{episode}.pth'  
        self.agent.save(checkpoint_path)
        if episode in ['best', 'final']:
            print(f"alreadysave model: {checkpoint_path}")

    def save_history_csvs(self):
        """save all history arrays as CSV files once at the end of training"""
        try:
            np.savetxt(f'{self.fig_dir}/episode_rewards.csv', np.array(self.episode_rewards), delimiter=',')
            np.savetxt(f'{self.fig_dir}/episode_lengths.csv', np.array(self.episode_lengths), delimiter=',')
            np.savetxt(f'{self.fig_dir}/success.csv', np.array(self.success_rate, dtype=int), fmt='%d', delimiter=',')
            np.savetxt(f'{self.fig_dir}/critic_loss.csv', np.array(self.losses['critic']), delimiter=',')
            np.savetxt(f'{self.fig_dir}/actor_loss.csv', np.array(self.losses['actor']), delimiter=',')
            np.savetxt(f'{self.fig_dir}/alpha_loss.csv', np.array(self.losses['alpha']), delimiter=',')
        except Exception as e:
            print(f"error while saving history CSV files: {e}")

def main(enable_render=False, render_freq=10):
    """main function"""
    # set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # devicecheck and optimize
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPUdevice: {torch.cuda.get_device_name(0)}")
        print(f"GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # GPUperformance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("enabled CUDNN")
    else:
        print("warning: not detected CUDA GPU, training velocity may be slower")
    
    def load_obstacle_matrix(file_path):
        """Load obstacle_grid_101x101.txt and rotate the grid map 90 degrees clockwise"""
        grid_map = np.loadtxt(file_path, dtype=np.float32)
        grid_map = np.rot90(grid_map, -1)  
        return grid_map

    def load_casualty_map(file_path):
        """Load the risk matrix and rotate it 90 degrees clockwise"""
        casualty_matrix = np.loadtxt(file_path, dtype=np.float32)
        casualty_matrix = np.rot90(casualty_matrix, -1)
        return casualty_matrix

    def load_meteor_map(file_path):
        """Load the meteorological-risk matrix and rotate it 90 degrees clockwise"""
        meteor_matrix = np.loadtxt(file_path, dtype=np.float32)
        meteor_matrix = np.rot90(meteor_matrix, -1)
        return meteor_matrix

    grid_map = load_obstacle_matrix("obstacle_grid_101x101.txt")
    casualty_map = load_casualty_map("casualty_matrix_101x101.txt")
    meteor_map = load_meteor_map("meteor_risk_101x101.txt")
    print("\nCreating environment...")
    env = SimplePathPlanningEnv(
        grid_map=grid_map,
        casualty_map=casualty_map,
        meteor_map=meteor_map,
        start=[10.0, 10.0],
        end=[57.0, 97.0],
        map_size=101,
        max_steps=500,
        dt=0.5,
        num_obstacles=8,
        dynamic_type="random",  
    )


    
    print(f"environment information:")
    print(f"observation space: {env.observation_space.shape}") 
    print(f"action space: {env.action_space.shape}")  
    print(f"action range: {env.action_space.low} to {env.action_space.high}") 
    radar_num_points = env.radar_num_points if hasattr(env, 'radar_num_points') else 64
    state_dim = 6
    action_dim = env.action_space.shape[0]

    # create SAC agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        radar_in_ch=3,
        radar_len=radar_num_points,
        state_feature_dim=256,
        lr=4e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy_tuning=True,
        hidden_dim=256,
        device=device
    )
    
    # create replay buffer
    replay_buffer = ReplayBuffer(capacity=1000000, device=device)
    print(f"replay buffercapacity: {1000000}")
    if device == 'cuda':
        # GPU raining parameters
        batch_size = 512
        update_every = 1 
        update_after = 500 
        max_episodes = 1000
    else:
        # CPU training parameters
        batch_size = 256
        update_every = 50
        update_after = 1000
        max_episodes = 500
    
    print(f"\ntraining parameters:")
    print(f"  batch size: {batch_size}")
    print(f"  update frequency: every {update_every}step")
    print(f"  learning starts: {update_after}steps later")
    print(f"  maximum episodes: {max_episodes}")
    
    # createtrainer
    trainer = SimpleTrainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        max_episodes=max_episodes,
        max_steps=500,
        batch_size=batch_size,
        update_after=update_after,
        update_every=update_every,
        save_every=100,
        render=enable_render,
        render_freq=render_freq
    )
    
    # start training
    try:
        best_success_rate = trainer.train()
        print(f"\ntraining finished!")
    except KeyboardInterrupt:
        print("\ntraining interrupted by the user")
    except Exception as e:
        print(f"\nerror during training: {e}")
        raise
    finally:
        env.close()
        print("environment has beendisabled")


def test_trained_model(model_path='models/20251021_142342/mossac_final.pth'):
    """test a trained model"""
    print(f"test a trained model: {model_path}")
    
    # checkmodel
    if not os.path.exists(model_path):
        print(f"model file does not exist: {model_path}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_obstacle_matrix(file_path):
        """Load obstacle_grid_101x101.txt and rotate the grid map 90 degrees clockwise"""
        grid_map = np.loadtxt(file_path, dtype=np.float32)
        grid_map = np.rot90(grid_map, -1) 
        return grid_map

    def load_casualty_map(file_path):
        """Load the risk matrix and rotate it 90 degrees clockwise"""
        casualty_matrix = np.loadtxt(file_path, dtype=np.float32)
        casualty_matrix = np.rot90(casualty_matrix, -1)
        return casualty_matrix

    def load_meteor_map(file_path):
        """Load the meteorological-risk matrix and rotate it 90 degrees clockwise"""
        meteor_matrix = np.loadtxt(file_path, dtype=np.float32)
        meteor_matrix = np.rot90(meteor_matrix, -1)
        return meteor_matrix


    grid_map = load_obstacle_matrix("obstacle_grid_101x101.txt")
    casualty_map = load_casualty_map("casualty_matrix_101x101.txt")
    meteor_map = load_meteor_map("meteor_risk_101x101.txt")
    print("\nCreating environment...")
    env = SimplePathPlanningEnv(grid_map=grid_map, casualty_map=casualty_map, meteor_map=meteor_map, start=[10.0, 10.0], end=[57.0, 97.0], max_steps=500, dt=0.5, num_obstacles=8, dynamic_type="fixed_test_both", seed=42)#random_test_both, fixed_test_both
    radar_num_points = env.radar_num_points if hasattr(env, 'radar_num_points') else 1
    state_dim = 6
    action_dim = env.action_space.shape[0]
    # createload
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        radar_in_ch=3,
        radar_len=radar_num_points,
        state_feature_dim=256,
        device=device
    )
    agent.load(model_path)

    
    # testepisode
    test_episodes = 3
    success_count = 0
    
    for episode in range(test_episodes):
        # set custom start and goal
        state = env.reset()
        total_reward = 0
        steps = 0

        print(f"\nepisode {episode + 1}:")
        
        for step in range(500):
            # use deterministicpolicy
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # visualization
            env.render()
            time.sleep(0.02)
            
            if done:
                reason = info.get('reason', 'unknown')
                if info.get('success', False):
                    success_count += 1
                    print(f"  result: goal reached! reward: {total_reward:.2f}, steps: {steps}")
                else:
                    print(f"  result: {reason}, reward: {total_reward:.2f}, steps: {steps}")
                break
            
            state = next_state
        
        input("Press Enter to continue to the next episode...")
        # env.close()
    
    print(f"\ntest completed!")
    print(f"success rate: {success_count}/{test_episodes} = {success_count/test_episodes:.2f}")
    env.close()

def test_statistics(model_path='models/20251021_142342/mossac_final.pth', test_episodes=1000):
    """statistics mode: evaluate model performance statistics"""
    print(f"statistics mode, model: {model_path}, test episodes: {test_episodes}")
    
    # checkmodel
    if not os.path.exists(model_path):
        print(f"model file does not exist: {model_path}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load maps
    def load_obstacle_matrix(file_path):
        grid_map = np.loadtxt(file_path, dtype=np.float32)
        grid_map = np.rot90(grid_map, -1)
        return grid_map

    def load_casualty_map(file_path):
        casualty_matrix = np.loadtxt(file_path, dtype=np.float32)
        casualty_matrix = np.rot90(casualty_matrix, -1)
        return casualty_matrix

    def load_meteor_map(file_path):
        meteor_matrix = np.loadtxt(file_path, dtype=np.float32)
        meteor_matrix = np.rot90(meteor_matrix, -1)
        return meteor_matrix

    grid_map = load_obstacle_matrix("obstacle_grid_101x101.txt")
    casualty_map = load_casualty_map("casualty_matrix_101x101.txt")
    meteor_map = load_meteor_map("meteor_risk_101x101.txt")
    
    # Creating environment
    env = SimplePathPlanningEnv(
        grid_map=grid_map, 
        casualty_map=casualty_map, 
        meteor_map=meteor_map, 
        start=[10.0, 10.0], 
        end=[57.0, 97.0], 
        max_steps=500, 
        dt=0.5, 
        num_obstacles=8, 
        dynamic_type="random_test_both",
    )
    
    radar_num_points = env.radar_num_points if hasattr(env, 'radar_num_points') else 64
    state_dim = 6
    action_dim = env.action_space.shape[0]
    
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        radar_in_ch=3,
        radar_len=radar_num_points,
        state_feature_dim=256,
        device=device
    )
    agent.load(model_path)
    
    # statistics data
    success_count = 0
    collision_count = 0
    deviation_count = 0  
    other_count = 0
    other_reasons = {}
    
    print("\nstarting statistics evaluation...")
    start_time_1 = time.time()
    print("progress: ", end='', flush=True)
    
    for episode in range(test_episodes):
        # progress bar: every 10% display once
        # The statistics mode requires a relatively long testing period.
        if (episode + 1) % (test_episodes // 10) == 0:
            print(f"{(episode + 1) * 100 // test_episodes}%... ", end='', flush=True)
        
        state = env.reset()
        
        for step in range(500):
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            # check heading deviation (bearing angle greater than 0°)
            # assume state[3] is the bearing angle (heading), verify this for your environment
            current_pos = np.array([env.position[0], env.position[1]])
            target_pos = np.array([env.target[0], env.target[1]])
            direction_to_target = np.arctan2(target_pos[1] - current_pos[1], 
                                            target_pos[0] - current_pos[0])
            current_heading = env.heading
            
            # compute bearing-angle deviation (normalized to[-pi, pi])
            angle_diff = direction_to_target - current_heading
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            angle_diff_deg = np.abs(np.degrees(angle_diff))

            # if the heading-deviation angle is greater than 0° (a threshold can be set, for example 5°)
            if angle_diff_deg > 90.0:  # set 5° as the threshold
                deviation_count += 1
                done = True
                info['reason'] = 'deviation'
                info['success'] = False
            
            if done:
                reason = info.get('reason', 'unknown')
                if info.get('success', False):
                    success_count += 1
                elif 'collision' in reason.lower():
                    collision_count += 1
                elif 'deviation' in reason.lower():
                    pass  # already counted above
                else:
                    other_count += 1
                    if reason in other_reasons:
                        other_reasons[reason] += 1
                    else:
                        other_reasons[reason] = 1
                break
            
            state = next_state
    
    end_time_1 = time.time()
    print("\n\nstatistics results:")
    print("="*50)
    print(f"total test time: {end_time_1 - start_time_1:.2f} s")
    print(f"total test episodes: {test_episodes}")
    print(f"success count: {success_count} ({success_count/test_episodes*100:.2f}%)")
    print(f"collision count: {collision_count} ({collision_count/test_episodes*100:.2f}%)")
    print(f"loss count: {deviation_count} ({deviation_count/test_episodes*100:.2f}%)")
    print(f"other failures: {other_count} ({other_count/test_episodes*100:.2f}%)")
    
    if other_reasons:
        print("\ndistribution of other failure reasons:")
        for reason, count in sorted(other_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} ({count/test_episodes*100:.2f}%)")
    
    print("="*50)
    
    env.close()


# training elapsed-time statistics
if __name__ == "__main__":
    import time
    print("Select mode: train / test / render / test_path / statistics / help")
    mode = input("Enter mode(press Enter for defaulttrain): ").strip().lower() or "train"
    if mode == "test":
        model_path = input("Enter model path (press Enter for default models/20251021_142342/mossac_final.pth): ").strip() or "models/20251021_142342/mossac_final.pth"
        test_trained_model(model_path)
    elif mode == "render":
        render_freq = input("Enter render frequency (press Enter for default10): ").strip()
        render_freq = int(render_freq) if render_freq else 10
        print(f"Starting render-training mode, render frequency: every {render_freq}episode")
        start_time = time.time()
        main(enable_render=True, render_freq=render_freq)
        end_time = time.time()
        print(f"Training elapsed time: {end_time - start_time:.2f} s")
    elif mode == "test_path":
        # Added test_path mode - log only R_clear, R_dyn, dynamic_penalty, and risk_penalty
        import csv
        from env_MOSSAC import calculate_distance, calculate_safety_rewards, get_casualty_value, get_meteor_value
        model_path = input("Enter model path (press Enter for default models/20251021_142342/mossac_final.pth): ").strip() or "models/20251021_142342/mossac_final.pth"
        print(f"test_path mode, model: {model_path}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        def load_obstacle_matrix(file_path):
            grid_map = np.loadtxt(file_path, dtype=np.float32)
            grid_map = np.rot90(grid_map, -1)
            return grid_map
        def load_casualty_map(file_path):
            casualty_matrix = np.loadtxt(file_path, dtype=np.float32).astype(np.int32)
            casualty_matrix = np.rot90(casualty_matrix, -1)
            return casualty_matrix
        def load_meteor_map(file_path):
            meteor_matrix = np.loadtxt(file_path, dtype=np.float32)
            meteor_matrix = np.rot90(meteor_matrix, -1)
            return meteor_matrix

        grid_map = load_obstacle_matrix("obstacle_grid_101x101.txt")
        casualty_map = load_casualty_map("casualty_matrix_101x101.txt")
        meteor_map = load_meteor_map("meteor_risk_101x101.txt")
        env = SimplePathPlanningEnv(grid_map=grid_map, casualty_map=casualty_map, meteor_map=meteor_map, start=[10.0, 10.0], end=[57.0, 97.0], max_steps=500, dt=0.5, num_obstacles=8, dynamic_type="both")
        radar_num_points = env.radar_num_points if hasattr(env, 'radar_num_points') else 64
        state_dim = 6
        action_dim = env.action_space.shape[0]
        agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            radar_in_ch=3,
            radar_len=radar_num_points,
            state_feature_dim=256,
            device=device
        )
        agent.load(model_path)
        # test one episode and log data for each step
        state = env.reset()
        log_rows = []
        total_reward = 0.0
        prev_heading = None
        for step in range(500):
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # log trajectory and key reward components
            pos = env.position.copy()
            heading = env.heading
            velocity = env.velocity

            # compute heading-change angle
            if prev_heading is None:
                delta_heading = 0.0
                turn_penalty = 0.0
            else:
                heading_change = abs(heading - prev_heading)
                turn_penalty = -0.05 * (heading_change / 0.07)  # larger heading change gives a larger penalty, capped at -0.1

            prev_heading = heading

            grid_map_obs, dynamic_obs_lists = env._get_all_obstacles()
            
            # compute safety-distance and TTC penalties (R_clear, R_dyn)
            vx = velocity * np.cos(heading)
            vy = velocity * np.sin(heading)
            R_clear, R_dyn, dmin, ttc = calculate_safety_rewards(pos[0], pos[1], vx, vy, grid_map_obs, dynamic_obs_lists, env.radar_max_range)
            dynamic_penalty = R_clear + R_dyn
            
            # computerisk penalty
            risk = get_casualty_value(pos[0], pos[1], env.casualty_map)
            if risk == 1:
                risk_penalty = -0.1
            elif risk == 2:
                risk_penalty = -0.2
            elif risk == 0:
                risk_penalty = -0.05
            else:
                risk_penalty = 0.0
            
            # meteorological risk
            meteor_risk = get_meteor_value(pos[0], pos[1], env.meteor_map)
            meteor_risk_penalty = -meteor_risk * 10



            log_rows.append([
                step, pos[0], pos[1], heading, velocity,
                R_clear, dmin, R_dyn, ttc, dynamic_penalty, risk_penalty, meteor_risk_penalty
            ])
            if done:
                print(f"episodeend: {info.get('reason', 'unknown')}, totalsteps: {step+1}")
                break
            state = next_state
        risk_penalty_sum = sum(row[10] for row in log_rows)
        meteor_risk_penalty_sum = sum(row[11] for row in log_rows)
        risk_sum = risk_penalty_sum + meteor_risk_penalty_sum
        R_clear_sum = sum(row[5] for row in log_rows)
        R_dyn_sum = sum(row[7] for row in log_rows)

        os.makedirs("paths", exist_ok=True)
        with open("paths/test.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y", "heading", "velocity", "R_clear", "dmin", "R_dyn", "ttc", "dynamic_penalty", "risk_penalty", "meteor_risk_penalty"])
            writer.writerows(log_rows)
            writer.writerow([])
            writer.writerow([f"Saved path and reward components; total steps: {len(log_rows)}step, total reward: {total_reward:.2f}"])
            writer.writerow([f"Total risk penalty: {risk_sum:.2f}"])
            writer.writerow([f"Total static penalty: {R_clear_sum:.2f}"])
            writer.writerow([f"Total dynamic penalty: {R_dyn_sum:.2f}"])
        print(f"Saved path and reward components; total steps: {len(log_rows)}step, total reward: {total_reward:.2f}")
        # Total risk penalty, meteorological penalty, static penalty, dynamic penalty

        print([f"Total risk penalty: {risk_sum:.2f}"])
        print([f"Total static penalty: {R_clear_sum:.2f}"])
        print([f"Total dynamic penalty: {R_dyn_sum:.2f}"])
        env.close()
    elif mode == "statistics":
        model_path = input("Enter model path (press Enter for default models/20251021_142342/mossac_final.pth): ").strip() or "models/20251021_142342/mossac_final.pth"
        test_episodes = input("Enter number of test episodes (press Enter for default 1000): ").strip()
        test_episodes = int(test_episodes) if test_episodes else 1000
        test_statistics(model_path, test_episodes)
    
    elif mode == "help":
        print("Usage:")
        print("  run the script directly and select a mode when prompted")
        print("  train      : standard training")
        print("  render     : rendered training, every N episode render")
        print("  test       : visualize a saved model for several episodes.")
        print("  test_path  : path test with per-step trajectory and reward-component logging")
        print("  statistics : statistics mode, run N tests and report success rate, collision rate, and related metrics")
        print("  help       : show help")
    
    else:
        start_time = time.time()

        main()
        end_time = time.time()
        print(f"Training elapsed time: {end_time - start_time:.2f} s")

