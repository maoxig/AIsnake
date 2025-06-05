import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from env.snake_env import SnakeEnv
from models.dqn_agent import DQNAgent
from utils.visualization import evaluate_agent
from train_ppo_advanced import SnakeWrapper, make_env

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="比较不同强化学习算法在贪吃蛇游戏上的性能")
    parser.add_argument("--eval-episodes", type=int, default=50, help="评估每个算法的回合数")
    parser.add_argument("--width", type=int, default=10, help="环境宽度")
    parser.add_argument("--height", type=int, default=10, help="环境高度")
    parser.add_argument("--custom-dqn-path", type=str, default=None, help="自定义DQN模型路径")
    parser.add_argument("--sb3-dqn-path", type=str, default=None, help="SB3 DQN模型路径")
    parser.add_argument("--sb3-ppo-path", type=str, default=None, help="SB3 PPO模型路径")
    parser.add_argument("--sb3-a2c-path", type=str, default=None, help="SB3 A2C模型路径")
    parser.add_argument("--advanced-ppo-path", type=str, default=None, help="高级特征提取器PPO模型路径")
    parser.add_argument("--output-dir", type=str, default="results", help="结果保存目录")
    return parser.parse_args()

def evaluate_algorithms():
    """评估不同强化学习算法的性能"""
    args = parse_args()
    
    # 创建结果目录
    output_dir = os.path.join("d:/project/AIsnake", args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建基础环境
    env = SnakeEnv(width=args.width, height=args.height)
    wrapped_env = SnakeWrapper(env)
    
    # 创建向量化环境用于SB3
    vec_env = DummyVecEnv([lambda: SnakeWrapper(SnakeEnv(width=args.width, height=args.height))])
    
    # 定义要评估的算法
    algorithms = []
    
    # 加载自定义DQN
    if args.custom_dqn_path:
        custom_dqn_path = os.path.join("d:/project/AIsnake", args.custom_dqn_path)
        if os.path.exists(custom_dqn_path):
            print(f"加载自定义DQN: {custom_dqn_path}")
            custom_dqn = DQNAgent(
                observation_space=env.observation_space,
                action_space=env.action_space
            )
            custom_dqn.load(custom_dqn_path)
            algorithms.append(("Custom DQN", custom_dqn, env, True))
    
    # 加载SB3 DQN
    if args.sb3_dqn_path:
        sb3_dqn_path = os.path.join("d:/project/AIsnake", args.sb3_dqn_path)
        if os.path.exists(sb3_dqn_path):
            print(f"加载SB3 DQN: {sb3_dqn_path}")
            sb3_dqn = DQN.load(sb3_dqn_path)
            algorithms.append(("SB3 DQN", sb3_dqn, vec_env, False))
    
    # 加载SB3 PPO
    if args.sb3_ppo_path:
        sb3_ppo_path = os.path.join("d:/project/AIsnake", args.sb3_ppo_path)
        if os.path.exists(sb3_ppo_path):
            print(f"加载SB3 PPO: {sb3_ppo_path}")
            sb3_ppo = PPO.load(sb3_ppo_path)
            algorithms.append(("SB3 PPO", sb3_ppo, vec_env, False))
    
    # 加载SB3 A2C
    if args.sb3_a2c_path:
        sb3_a2c_path = os.path.join("d:/project/AIsnake", args.sb3_a2c_path)
        if os.path.exists(sb3_a2c_path):
            print(f"加载SB3 A2C: {sb3_a2c_path}")
            sb3_a2c = A2C.load(sb3_a2c_path)
            algorithms.append(("SB3 A2C", sb3_a2c, vec_env, False))
    
    # 加载高级特征提取器PPO
    if args.advanced_ppo_path:
        advanced_ppo_path = os.path.join("d:/project/AIsnake", args.advanced_ppo_path)
        if os.path.exists(advanced_ppo_path):
            print(f"加载高级特征PPO: {advanced_ppo_path}")
            advanced_ppo = PPO.load(advanced_ppo_path)
            algorithms.append(("Advanced PPO", advanced_ppo, vec_env, False))
    
    # 如果没有指定任何模型路径，打印警告
    if len(algorithms) == 0:
        print("警告: 未指定任何模型路径! 请提供至少一个有效的模型路径进行评估。")
        return
    
    # 评估结果
    results = []
    
    # 评估每个算法
    for name, model, eval_env, is_custom in algorithms:
        print(f"评估 {name}...")
        start_time = time.time()
        
        episode_rewards = []
        episode_lengths = []
        episode_snake_lengths = []
        
        for _ in range(args.eval_episodes):
            if is_custom:
                # 自定义DQN评估
                obs, info = eval_env.reset()
                done = False
                truncated = False
                episode_reward = 0
                step = 0
                
                while not (done or truncated):
                    action = model.select_action(obs, training=False)
                    obs, reward, done, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    step += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(step)
                episode_snake_lengths.append(len(eval_env.snake_body) - 2)  # 减去初始长度
            else:
                # SB3模型评估
                mean_reward, std_reward = evaluate_policy(
                    model, 
                    eval_env, 
                    n_eval_episodes=1, 
                    deterministic=True,
                    return_episode_rewards=True
                )
                
                # 从向量环境中获取信息
                info = eval_env.get_attr("snake_body")[0]
                snake_length = len(info) - 2 if info else 0
                
                episode_rewards.append(mean_reward[0])
                episode_lengths.append(len(mean_reward))
                episode_snake_lengths.append(snake_length)
        
        eval_time = time.time() - start_time
        
        # 计算结果统计
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        mean_snake_length = np.mean(episode_snake_lengths)
        std_snake_length = np.std(episode_snake_lengths)
        
        print(f"{name} - 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}, " 
              f"平均回合长度: {mean_length:.2f} ± {std_length:.2f}, "
              f"平均蛇长度: {mean_snake_length:.2f} ± {std_snake_length:.2f}")
        
        results.append({
            "算法": name,
            "平均奖励": mean_reward,
            "奖励标准差": std_reward,
            "平均回合长度": mean_length,
            "长度标准差": std_length,
            "平均蛇长度": mean_snake_length,
            "蛇长度标准差": std_snake_length,
            "评估时间(秒)": eval_time
        })
    
    # 创建结果数据框
    results_df = pd.DataFrame(results)
    
    # 保存到CSV
    csv_path = os.path.join(output_dir, "algorithm_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")
    
    # 绘制比较图表
    plt.figure(figsize=(15, 10))
    
    # 平均奖励比较
    plt.subplot(2, 2, 1)
    plt.bar(results_df["算法"], results_df["平均奖励"], yerr=results_df["奖励标准差"])
    plt.title("平均奖励比较")
    plt.ylabel("奖励")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 平均回合长度比较
    plt.subplot(2, 2, 2)
    plt.bar(results_df["算法"], results_df["平均回合长度"], yerr=results_df["长度标准差"])
    plt.title("平均回合长度比较")
    plt.ylabel("步数")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 平均蛇长度比较
    plt.subplot(2, 2, 3)
    plt.bar(results_df["算法"], results_df["平均蛇长度"], yerr=results_df["蛇长度标准差"])
    plt.title("平均蛇长度比较")
    plt.ylabel("长度")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 评估时间比较
    plt.subplot(2, 2, 4)
    plt.bar(results_df["算法"], results_df["评估时间(秒)"])
    plt.title("评估时间比较")
    plt.ylabel("时间(秒)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"))
    print(f"比较图表已保存到: {os.path.join(output_dir, 'algorithm_comparison.png')}")

if __name__ == "__main__":
    evaluate_algorithms()
