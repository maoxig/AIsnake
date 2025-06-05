import argparse
import os
import numpy as np
import torch

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from env.snake_env import SnakeEnv

def parse_args():
    parser = argparse.ArgumentParser(description="使用Stable Baselines 3训练贪吃蛇智能体")
    parser.add_argument("--algo", type=str, default="ppo", choices=["dqn", "ppo", "a2c"], help="要使用的算法")
    parser.add_argument("--width", type=int, default=10, help="环境宽度")
    parser.add_argument("--height", type=int, default=10, help="环境高度")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="训练的总时间步数")
    parser.add_argument("--lr", type=float, default=0.0003, help="学习率")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--eval-freq", type=int, default=10000, help="评估频率")
    parser.add_argument("--eval-episodes", type=int, default=5, help="每次评估的回合数")
    return parser.parse_args()

def train():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    
    # 创建向量化环境（可以并行运行多个环境）
    env = make_vec_env(
        lambda: SnakeEnv(width=args.width, height=args.height),
        n_envs=1
    )
    
    # 创建模型保存目录
    os.makedirs("d:/project/AIsnake/models/sb3", exist_ok=True)
    
    # 创建模型
    if args.algo.lower() == "dqn":
        model = DQN(
            "MlpPolicy",  # Stable Baselines 3 会自动处理观察空间
            env,
            learning_rate=args.lr,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=args.batch_size,
            tau=1.0,  # 目标网络更新率
            gamma=0.99,  # 折扣因子
            train_freq=4,  # 每4步更新一次
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log="d:/project/AIsnake/logs/"
        )
    elif args.algo.lower() == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=128,
            batch_size=args.batch_size,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="d:/project/AIsnake/logs/"
        )
    elif args.algo.lower() == "a2c":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=5,
            gamma=0.99,
            verbose=1,
            tensorboard_log="d:/project/AIsnake/logs/"
        )
    else:
        raise ValueError(f"不支持的算法: {args.algo}")
    
    # 训练前先评估一次
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes)
    print(f"开始训练前的评估结果 - 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 训练模型
    print(f"开始使用 {args.algo.upper()} 算法训练...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=None,  # 可以添加自定义回调函数
        log_interval=1000,
        tb_log_name=f"{args.algo}-snake",
        reset_num_timesteps=True,
        progress_bar=True
    )
    
    # 保存模型
    model_path = f"d:/project/AIsnake/models/sb3/{args.algo}_snake_final"
    model.save(model_path)
    print(f"模型已保存至 {model_path}")
    
    # 最终评估
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"训练后评估结果 - 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")

if __name__ == "__main__":
    train()
