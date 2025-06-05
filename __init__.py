# 贪吃蛇强化学习项目初始化文件
from env.snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT
from models.dqn_agent import DQNAgent
from models.networks import DQNNetwork
from utils.visualization import evaluate_agent, plot_training_progress, visualize_game

__version__ = "1.0.0"

__all__ = [
    "SnakeEnv", "UP", "DOWN", "LEFT", "RIGHT",
    "DQNAgent", "DQNNetwork",
    "evaluate_agent", "plot_training_progress", "visualize_game"
]
