import os
import time
import argparse
import numpy as np
import torch
from torch import nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from env.snake_env import SnakeEnv


class TensorboardCallback(BaseCallback):
    """
    记录额外指标到Tensorboard的自定义回调
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 收集所有环境的信息
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][idx]
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    
                    # 每10个回合记录一次统计信息
                    if len(self.episode_rewards) % 10 == 0:
                        mean_reward = np.mean(self.episode_rewards[-10:])
                        mean_length = np.mean(self.episode_lengths[-10:])
                        
                        self.logger.record("custom/mean_reward_10", mean_reward)
                        self.logger.record("custom/mean_length_10", mean_length)
                        
                        # 可以添加更多自定义指标，例如蛇的长度等
                        if "snake_length" in info:
                            self.logger.record("custom/snake_length", info["snake_length"])
        return True


class SnakeWrapper(gym.Wrapper):
    """
    贪吃蛇环境的包装器，添加一些额外的奖励和特性以优化训练
    """
    def __init__(self, env, reward_shaping=True):
        super().__init__(env)
        self.reward_shaping = reward_shaping
        self.prev_snake_length = 2  # 初始蛇长度
        self.max_snake_length = 2
        self.no_food_steps = 0
        self.max_no_food_steps = self.env.width * self.env.height * 2  # 最大步数阈值
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_snake_length = 2
        self.max_snake_length = 2
        self.no_food_steps = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_snake_length = len(self.env.snake_body)
        
        # 奖励调整
        if self.reward_shaping:
            # 1. 吃到食物的额外奖励
            if current_snake_length > self.prev_snake_length:
                reward += 2.0  # 额外奖励
                self.no_food_steps = 0  # 重置不吃食物的步数
            else:
                self.no_food_steps += 1
                
            # 2. 记录最大长度并提供奖励
            if current_snake_length > self.max_snake_length:
                reward += 1.0  # 达到新的最大长度奖励
                self.max_snake_length = current_snake_length
            
            # 3. 如果长时间不吃食物，给予小惩罚
            if self.no_food_steps > self.max_no_food_steps:
                reward -= 0.5
                truncated = True  # 如果太久不吃食物，截断回合
            
            # 4. 接近食物的奖励或远离食物的惩罚
            head_x, head_y = self.env.snake_body[0]
            food_x, food_y = self.env.food_position
            current_distance = abs(head_x - food_x) + abs(head_y - food_y)
            
            # 估计前一个位置
            prev_head_x, prev_head_y = head_x, head_y
            if action == 0:  # UP
                prev_head_y += 1
            elif action == 1:  # DOWN
                prev_head_y -= 1
            elif action == 2:  # LEFT
                prev_head_x += 1
            elif action == 3:  # RIGHT
                prev_head_x -= 1
                
            prev_distance = abs(prev_head_x - food_x) + abs(prev_head_y - food_y)
            
            if current_distance < prev_distance:
                reward += 0.1  # 接近食物
            elif current_distance > prev_distance:
                reward -= 0.1  # 远离食物
                
            # 5. 额外的惩罚：蛇走回头路
            # 这需要检测蛇是否在一个小区域内来回移动
            
        # 更新前一个蛇长度
        self.prev_snake_length = current_snake_length
        
        # 添加额外信息
        info["snake_length"] = current_snake_length
        info["max_snake_length"] = self.max_snake_length
        info["no_food_steps"] = self.no_food_steps
        
        return obs, reward, terminated, truncated, info


def make_env(env_id, wrapper_class=None, rank=0, seed=0):
    """
    创建环境的帮助函数，设置随机种子并应用包装器
    """
    def _init():
        if env_id == "Snake-v0":
            env = SnakeEnv(width=10, height=10)
        else:
            env = gym.make(env_id)
        
        # 设置环境的种子
        env.reset(seed=seed + rank)
        
        # 应用基本的封装
        env = Monitor(env)  # 追踪统计数据
        
        # 应用自定义包装器
        if wrapper_class is not None:
            env = wrapper_class(env)
            
        return env
    return _init


def parse_args():
    """命令行参数解析函数"""
    parser = argparse.ArgumentParser(description="使用PPO训练贪吃蛇智能体，带进阶特性")
    parser.add_argument("--width", type=int, default=10, help="环境宽度")
    parser.add_argument("--height", type=int, default=10, help="环境高度")
    parser.add_argument("--total-timesteps", type=int, default=2000000, help="训练的总时间步数")
    parser.add_argument("--n-envs", type=int, default=8, help="并行环境数量")
    parser.add_argument("--reward-shaping", type=bool, default=True, help="是否使用奖励塑形")
    parser.add_argument("--save-dir", type=str, default="models/sb3_ppo_advanced", help="模型保存目录")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--eval-freq", type=int, default=10000, help="评估频率（步数）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    return parser.parse_args()


def train_ppo_advanced():
    """高级PPO训练函数，带自适应超参数和其他特性"""
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)
    
    # 创建保存目录
    model_dir = os.path.join("d:/project/AIsnake", args.save_dir)
    log_dir = os.path.join("d:/project/AIsnake", args.log_dir)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建并行环境
    env_id = "Snake-v0"  # 我们的自定义环境ID
    
    env = SubprocVecEnv([make_env(env_id, 
                                wrapper_class=lambda e: SnakeWrapper(e, reward_shaping=args.reward_shaping), 
                                rank=i, 
                                seed=args.seed) 
                         for i in range(args.n_envs)])
    
    # 监控向量环境
    env = VecMonitor(env)
    
    # 创建日志记录器
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    print(f"创建PPO模型，使用 {args.n_envs} 个并行环境")
    
    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=args.lr,
        # PPO特有参数
        n_steps=128,  # 每次更新前收集的步数
        batch_size=64,
        n_epochs=4,   # 每次更新的优化epochs
        gamma=0.99,   # 折扣因子
        gae_lambda=0.95,  # GAE优势估计的λ参数
        clip_range=0.2,   # PPO裁剪参数
        # 正则化参数
        ent_coef=0.01,  # 熵系数，鼓励探索
        vf_coef=0.5,   # 价值函数系数
        max_grad_norm=0.5,  # 梯度裁剪
        # 其他参数
        seed=args.seed
    )
    
    # 设置新的logger
    model.set_logger(new_logger)
    
    # 创建回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=model_dir,
        name_prefix="ppo_snake"
    )
    
    tensorboard_callback = TensorboardCallback()
    
    print(f"开始训练，总步数: {args.total_timesteps}")
    start_time = time.time()
    
    # 训练模型
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, tensorboard_callback],
        tb_log_name="ppo_snake_advanced",
        reset_num_timesteps=True,
        progress_bar=True
    )
    
    # 保存最终模型
    model.save(f"{model_dir}/final_model")
    
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f} 秒")
    
    # 最终评估
    print("进行最终评估...")
    eval_env = make_env(env_id, wrapper_class=SnakeWrapper, seed=args.seed)()
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"最终评估结果 - 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return model


if __name__ == "__main__":
    trained_model = train_ppo_advanced()
