import argparse
import os
import torch

from env.snake_env import SnakeEnv
from models.dqn_agent import DQNAgent
from utils.visualization import visualize_game

def parse_args():
    parser = argparse.ArgumentParser(description="用训练好的DQN代理玩贪吃蛇")
    parser.add_argument("--model-path", type=str, default="models/saved/dqn_snake_final.pt", help="模型文件路径")
    parser.add_argument("--width", type=int, default=10, help="环境宽度")
    parser.add_argument("--height", type=int, default=10, help="环境高度")
    parser.add_argument("--max-steps", type=int, default=1000, help="最大步数")
    return parser.parse_args()

def play():
    args = parse_args()
    
    # 检查模型文件是否存在
    model_path = os.path.join("d:/project/AIsnake", args.model_path)
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        print("请先训练模型或提供有效的模型路径")
        return
    
    # 创建环境
    env = SnakeEnv(width=args.width, height=args.height)
    
    # 创建代理
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    
    # 加载模型
    agent.load(model_path)
    print(f"已加载模型: {model_path}")
    
    # 启动可视化游戏
    print("启动游戏...")
    visualize_game(env, agent, max_steps=args.max_steps)

if __name__ == "__main__":
    play()
