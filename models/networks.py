import numpy as np
from gymnasium import spaces
import torch
from torch import nn

class FeatureExtractor(nn.Module):
    """
    将原始状态转换为神经网络特征
    """
    def __init__(self, observation_space):
        super().__init__()
        # 简单的特征提取网络
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 计算展平后的特征尺寸
        self.feature_size = self._get_feature_size(observation_space)
        
    def _get_feature_size(self, observation_space):
        # 计算特征提取器输出大小
        sample_input = torch.zeros(1, 1, *observation_space.shape)
        with torch.no_grad():
            return self.cnn(sample_input).shape[1]
        
    def forward(self, observations):
        # 添加通道维度
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(1)
        return self.cnn(observations)


class DQNNetwork(nn.Module):
    """
    DQN网络，包括特征提取器和动作价值输出
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.feature_extractor = FeatureExtractor(observation_space)
        
        # 动作值网络
        self.action_net = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n)
        )
        
    def forward(self, observations):
        features = self.feature_extractor(observations)
        return self.action_net(features)
