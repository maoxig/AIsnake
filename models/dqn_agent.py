import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import random
from collections import deque
import os

from models.networks import DQNNetwork

class DQNAgent:
    """
    深度Q网络（DQN）强化学习代理
    """
    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        exploration_initial_eps=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # 创建Q网络和目标网络
        self.q_network = DQNNetwork(observation_space, action_space).to(device)
        self.target_network = DQNNetwork(observation_space, action_space).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # 探索率衰减
        self.exploration_schedule = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * 1000000),
            initial_p=exploration_initial_eps,
            final_p=exploration_final_eps
        )
        
        # 训练步数计数
        self.timesteps = 0
    
    def select_action(self, observation, training=True):
        """
        根据当前状态选择动作
        使用epsilon-greedy探索策略
        """
        if training:
            # 在训练时使用epsilon-greedy策略
            epsilon = self.exploration_schedule.value(self.timesteps)
            if random.random() < epsilon:
                return random.randint(0, self.action_space.n - 1)
        
        # 进行前向传播获取动作值
        with torch.no_grad():
            observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(observation_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """
        存储一个转换到经验回放缓冲区
        """
        self.replay_buffer.append((obs, action, reward, next_obs, done))
    
    def train(self):
        """
        从经验回放缓冲区中采样并训练网络
        """
        # 增加训练步数
        self.timesteps += 1
        
        # 检查缓冲区是否有足够的数据
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # 缓冲区样本不足，跳过训练
        
        # 从缓冲区随机采样一批转换
        transitions = random.sample(self.replay_buffer, self.batch_size)
        
        # 将转换分为独立的批次
        observations, actions, rewards, next_observations, dones = zip(*transitions)
        
        # 转换为张量
        observations = torch.FloatTensor(np.array(observations)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_observations = torch.FloatTensor(np.array(next_observations)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算下一状态的最大Q值
        with torch.no_grad():
            max_next_q_values = self.target_network(next_observations).max(1)[0]
            # 如果done为True，则下一状态的Q值为0
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 梯度下降步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 定期更新目标网络
        if self.timesteps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, path):
        """保存模型到指定路径"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "timesteps": self.timesteps
        }, path)
        
    def load(self, path):
        """从指定路径加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.timesteps = checkpoint["timesteps"]


class LinearSchedule:
    """
    线性调度类，用于epsilon等参数的衰减
    """
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """
        schedule_timesteps: 衰减到final_p需要的时间步数
        initial_p: 初始值
        final_p: 最终值
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
    
    def value(self, t):
        """返回时间步t的调度值"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
