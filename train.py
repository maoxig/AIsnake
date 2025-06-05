import os
import numpy as np
import torch
import argparse
from datetime import datetime
import gymnasium as gym

# 导入自定义模块
from env.snake_env import SnakeEnv
from models.dqn_agent import DQNAgent
from utils.visualization import evaluate_agent, plot_training_progress

def parse_args():
    parser = argparse.ArgumentParser(description="训练贪吃蛇DQN代理")
    parser.add_argument("--width", type=int, default=10, help="环境宽度")
    parser.add_argument("--height", type=int, default=10, help="环境高度")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="训练的总时间步数")
    parser.add_argument("--buffer-size", type=int, default=50000, help="经验回放缓冲区大小")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--target-update-freq", type=int, default=1000, help="目标网络更新频率")
    parser.add_argument("--eval-freq", type=int, default=10000, help="评估频率")
    parser.add_argument("--eval-episodes", type=int, default=5, help="每次评估的回合数")
    parser.add_argument("--exploration-fraction", type=float, default=0.2, help="探索率衰减分数")
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0, help="初始探索率")
    parser.add_argument("--exploration-final-eps", type=float, default=0.01, help="最终探索率")
    parser.add_argument("--save-freq", type=int, default=50000, help="保存模型频率")
    parser.add_argument("--log-freq", type=int, default=1000, help="日志记录频率")
    return parser.parse_args()

def train():
    # 解析参数
    args = parse_args()
    
    # 创建环境
    env = SnakeEnv(width=args.width, height=args.height)
    eval_env = SnakeEnv(width=args.width, height=args.height)
    
    # 设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    
    # 创建代理
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps
    )
    
    # 创建模型目录
    os.makedirs("d:/project/AIsnake/models/saved", exist_ok=True)
    
    # 训练循环
    obs, info = env.reset()
    timesteps = 0
    episodes = 0
    episode_reward = 0
    episode_length = 0
    
    # 记录训练指标
    all_episode_rewards = []
    all_episode_lengths = []
    evaluation_rewards = []
    evaluation_lengths = []
    
    print("开始训练...")
    while timesteps < args.total_timesteps:
        # 选择动作
        action = agent.select_action(obs)
        
        # 执行动作
        next_obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        
        # 存储转换
        agent.store_transition(obs, action, reward, next_obs, done or truncated)
        
        # 训练网络
        loss = agent.train()
        
        # 更新观察
        obs = next_obs
        
        # 检查是否完成回合
        if done or truncated:
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
            
            # 重置环境
            obs, info = env.reset()
            episodes += 1
            episode_reward = 0
            episode_length = 0
        
        # 定期评估
        if timesteps % args.eval_freq == 0:
            eval_reward, eval_length = evaluate_agent(eval_env, agent, num_episodes=args.eval_episodes)
            evaluation_rewards.append(eval_reward)
            evaluation_lengths.append(eval_length)
            print(f"Timestep: {timesteps}, 评估奖励: {eval_reward:.3f}, 评估长度: {eval_length:.2f}")
            
            # 绘制训练进度
            if all_episode_rewards:
                plot_training_progress(all_episode_rewards, all_episode_lengths)
        
        # 定期保存模型
        if timesteps % args.save_freq == 0:
            agent.save(f"d:/project/AIsnake/models/saved/dqn_snake_{timesteps}.pt")
        
        # 定期打印日志
        if timesteps % args.log_freq == 0:
            if all_episode_rewards:
                avg_reward = np.mean(all_episode_rewards[-100:])
                avg_length = np.mean(all_episode_lengths[-100:])
                print(f"Timestep: {timesteps}, 平均奖励 (最近100回合): {avg_reward:.3f}, 平均长度: {avg_length:.2f}")
        
        timesteps += 1
    
    # 保存最终模型
    agent.save("d:/project/AIsnake/models/saved/dqn_snake_final.pt")
    print(f"训练完成，共 {timesteps} 时间步，{episodes} 回合")
    
    # 最终评估
    eval_reward, eval_length = evaluate_agent(eval_env, agent, num_episodes=10)
    print(f"最终评估 - 平均奖励: {eval_reward:.3f}, 平均长度: {eval_length:.2f}")
    
    # 绘制最终训练进度
    plot_training_progress(all_episode_rewards, all_episode_lengths)

if __name__ == "__main__":
    train()
