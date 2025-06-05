import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

from env.snake_env import SnakeEnv
from models.advanced_networks import create_snake_policy_kwargs, CustomActorCriticPolicy, AdvancedActorCriticPolicy
from train_ppo_advanced import SnakeWrapper, make_env

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用高级特征提取器训练贪吃蛇")
    parser.add_argument("--feature-extractor", type=str, choices=["cnn", "attention"], 
                        default="cnn", help="使用的特征提取器类型")
    parser.add_argument("--width", type=int, default=10, help="环境宽度")
    parser.add_argument("--height", type=int, default=10, help="环境高度")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="训练的总时间步数")
    parser.add_argument("--n-envs", type=int, default=8, help="并行环境数量")
    parser.add_argument("--save-dir", type=str, default="models/sb3_advanced", help="模型保存目录")
    parser.add_argument("--log-dir", type=str, default="logs/advanced", help="日志目录")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--eval-freq", type=int, default=10000, help="评估频率")
    return parser.parse_args()

def train_with_advanced_features():
    """使用高级特征提取器训练贪吃蛇"""
    args = parse_args()
    
    # 创建目录
    model_dir = os.path.join("d:/project/AIsnake", args.save_dir)
    log_dir = os.path.join("d:/project/AIsnake", args.log_dir)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 创建环境
    env_id = "Snake-v0"
    env = SubprocVecEnv([make_env(env_id, 
                                wrapper_class=lambda e: SnakeWrapper(e, reward_shaping=True), 
                                rank=i, 
                                seed=args.seed) 
                         for i in range(args.n_envs)])
    
    # 监控向量环境
    env = VecMonitor(env)
    
    # 创建评估环境
    eval_env = make_env(env_id, wrapper_class=SnakeWrapper, seed=args.seed)()
    
    # 设置评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq // args.n_envs,
        deterministic=True,
        render=False
    )
    
    # 选择策略和特征提取器
    if args.feature_extractor == "cnn":
        policy = CustomActorCriticPolicy
        print("使用CNN特征提取器")
    else:
        policy = AdvancedActorCriticPolicy
        print("使用注意力特征提取器")
    
    # 创建PPO模型
    policy_kwargs = create_snake_policy_kwargs()
    model = PPO(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=args.seed
    )
    
    print(f"开始使用 {args.feature_extractor} 特征提取器训练...")
    
    # 训练模型
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        tb_log_name=f"snake_{args.feature_extractor}",
        progress_bar=True
    )
    
    # 保存最终模型
    model_path = f"{model_dir}/snake_{args.feature_extractor}_final"
    model.save(model_path)
    print(f"模型已保存至 {model_path}")
    
    return model

if __name__ == "__main__":
    train_with_advanced_features()
