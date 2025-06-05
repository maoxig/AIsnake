import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    使用卷积神经网络提取特征的自定义特征提取器
    特别适合处理贪吃蛇游戏的网格状态
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # 获取观察空间的形状
        n_input_channels = 1  # 我们的观察是一个单通道的二维数组
        
        # CNN架构
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算CNN输出尺寸
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            sample_shape = sample.shape
            if len(sample_shape) == 3:  # 2D observation, missing batch dim
                sample = sample.unsqueeze(0)
            if len(sample_shape) == 4 and sample_shape[1] > 1:  # 3D observation, with batch dim
                sample = sample.unsqueeze(1)
            sample = sample.permute(0, 1, 2, 3)  # Batch, Channel, Height, Width
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        """
        # 重塑输入以适应卷积层
        batch_size = observations.shape[0]
        
        # 添加通道维度如果需要
        if len(observations.shape) == 3:  # [batch, height, width]
            x = observations.unsqueeze(1)  # 添加通道维
        else:
            x = observations
        
        x = self.cnn(x)
        x = self.linear(x)
        
        return x


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    使用自定义特征提取器的Actor-Critic策略
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         # 自定义特征提取器
                         features_extractor_class=CNNFeaturesExtractor,
                         features_extractor_kwargs=dict(features_dim=256))


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """
    使用自注意力机制的特征提取器
    可以学习关注蛇头、食物和障碍物之间的关系
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # 初始化CNN用于提取空间特征
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # 计算CNN输出的空间维度
        # 假设输入是10x10的网格
        self.spatial_dim = observation_space.shape[0] * observation_space.shape[1]
        self.latent_dim = 64  # 从CNN输出的通道数
        
        # 自注意力层
        self.query = nn.Linear(self.latent_dim, 64)
        self.key = nn.Linear(self.latent_dim, 64)
        self.value = nn.Linear(self.latent_dim, 64)
        
        # 输出层
        self.out = nn.Sequential(
            nn.Linear(64 * self.spatial_dim, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # 添加通道维度
        if len(observations.shape) == 3:  # [batch, height, width]
            x = observations.unsqueeze(1)
        else:
            x = observations
        
        # CNN特征提取
        x = self.cnn(x)  # [batch, channels, height, width]
        
        # 重塑以准备注意力
        c, h, w = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(batch_size, c, h * w)  # [batch, channels, height*width]
        x = x.permute(0, 2, 1)  # [batch, height*width, channels]
        
        # 自注意力计算
        q = self.query(x)  # [batch, height*width, 64]
        k = self.key(x)    # [batch, height*width, 64]
        v = self.value(x)  # [batch, height*width, 64]
        
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(64)
        attention = F.softmax(scores, dim=-1)
        
        # 应用注意力
        context = torch.matmul(attention, v)  # [batch, height*width, 64]
        
        # 展平并通过输出层
        context = context.reshape(batch_size, -1)
        output = self.out(context)
        
        return output


class AdvancedActorCriticPolicy(ActorCriticPolicy):
    """
    使用注意力特征提取器的高级Actor-Critic策略
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=AttentionFeaturesExtractor,
                         features_extractor_kwargs=dict(features_dim=256))


def create_snake_policy_kwargs():
    """
    创建贪吃蛇游戏的策略参数
    """
    return {
        "activation_fn": nn.ReLU,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        # 使用自注意力或CNN特征提取器
        "features_extractor_class": CNNFeaturesExtractor,
        "features_extractor_kwargs": dict(features_dim=128),
    }
