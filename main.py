#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
贪吃蛇强化学习项目的主入口脚本
提供一个简单的命令行界面，用于训练和测试不同的强化学习算法
"""

import argparse
import os
import sys
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="贪吃蛇强化学习项目")
    
    # 主命令
    subparsers = parser.add_subparsers(dest="command", help="要执行的命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练强化学习模型")
    train_parser.add_argument("--algo", type=str, default="dqn", 
                              choices=["dqn", "ppo", "a2c", "advanced", "ppo-advanced"], 
                              help="要使用的强化学习算法")
    train_parser.add_argument("--width", type=int, default=10, help="环境宽度")
    train_parser.add_argument("--height", type=int, default=10, help="环境高度")
    train_parser.add_argument("--timesteps", type=int, default=1000000, help="训练的总时间步数")
    
    # 测试命令
    play_parser = subparsers.add_parser("play", help="使用训练好的模型玩游戏")
    play_parser.add_argument("--algo", type=str, default="dqn", 
                             choices=["dqn", "ppo", "a2c", "advanced", "ppo-advanced"], 
                             help="要使用的强化学习算法")
    play_parser.add_argument("--model-path", type=str, help="模型文件路径")
    play_parser.add_argument("--width", type=int, default=10, help="环境宽度")
    play_parser.add_argument("--height", type=int, default=10, help="环境高度")
    
    # 比较命令
    compare_parser = subparsers.add_parser("compare", help="比较不同算法的性能")
    compare_parser.add_argument("--eval-episodes", type=int, default=50, help="评估每个算法的回合数")
    compare_parser.add_argument("--models", type=str, nargs="+", help="要比较的模型路径列表")
    
    # 演示命令
    demo_parser = subparsers.add_parser("demo", help="启动演示窗口")
    demo_parser.add_argument("--model-path", type=str, help="模型文件路径")
    demo_parser.add_argument("--width", type=int, default=10, help="环境宽度")
    demo_parser.add_argument("--height", type=int, default=10, help="环境高度")
    
    # 安装命令
    install_parser = subparsers.add_parser("install", help="安装依赖")
    
    # 帮助
    help_parser = subparsers.add_parser("help", help="显示帮助信息")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.command == "train":
        if args.algo == "dqn":
            if args.algo == "dqn":
                script = "train.py"
                cmd = [sys.executable, script, 
                       "--width", str(args.width), 
                       "--height", str(args.height), 
                       "--total-timesteps", str(args.timesteps)]
            else:
                script = "train_sb3.py"
                cmd = [sys.executable, script, 
                       "--algo", "dqn", 
                       "--width", str(args.width), 
                       "--height", str(args.height), 
                       "--total-timesteps", str(args.timesteps)]
                       
        elif args.algo == "ppo" or args.algo == "a2c":
            script = "train_sb3.py"
            cmd = [sys.executable, script, 
                   "--algo", args.algo, 
                   "--width", str(args.width), 
                   "--height", str(args.height), 
                   "--total-timesteps", str(args.timesteps)]
                   
        elif args.algo == "advanced":
            script = "train_advanced_features.py"
            cmd = [sys.executable, script, 
                   "--feature-extractor", "cnn", 
                   "--width", str(args.width), 
                   "--height", str(args.height), 
                   "--total-timesteps", str(args.timesteps)]
                   
        elif args.algo == "ppo-advanced":
            script = "train_ppo_advanced.py"
            cmd = [sys.executable, script, 
                   "--width", str(args.width), 
                   "--height", str(args.height), 
                   "--total-timesteps", str(args.timesteps)]
        
        print(f"运行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.command == "play":
        if args.algo == "dqn" and not args.model_path:
            script = "play.py"
            cmd = [sys.executable, script,
                   "--width", str(args.width),
                   "--height", str(args.height)]
                   
            if args.model_path:
                cmd.extend(["--model-path", args.model_path])
                
        elif args.algo in ["ppo", "a2c"] or (args.algo == "dqn" and args.model_path):
            script = "play_sb3.py"
            cmd = [sys.executable, script,
                   "--algo", args.algo,
                   "--width", str(args.width),
                   "--height", str(args.height)]
                   
            if args.model_path:
                cmd.extend(["--model-path", args.model_path])
                
        elif args.algo in ["advanced", "ppo-advanced"]:
            script = "snake_demo.py"
            cmd = [sys.executable, script,
                   "--width", str(args.width),
                   "--height", str(args.height)]
                   
            if args.model_path:
                cmd.extend(["--model-path", args.model_path])
        
        print(f"运行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.command == "compare":
        script = "compare_algorithms.py"
        cmd = [sys.executable, script, "--eval-episodes", str(args.eval_episodes)]
        
        if args.models:
            for i, model_path in enumerate(args.models):
                if i == 0:
                    cmd.extend(["--custom-dqn-path", model_path])
                elif i == 1:
                    cmd.extend(["--sb3-dqn-path", model_path])
                elif i == 2:
                    cmd.extend(["--sb3-ppo-path", model_path])
                elif i == 3:
                    cmd.extend(["--sb3-a2c-path", model_path])
                elif i == 4:
                    cmd.extend(["--advanced-ppo-path", model_path])
        
        print(f"运行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.command == "demo":
        script = "snake_demo.py"
        cmd = [sys.executable, script,
               "--width", str(args.width),
               "--height", str(args.height)]
               
        if args.model_path:
            cmd.extend(["--model-path", args.model_path])
        
        print(f"运行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.command == "install":
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        print(f"运行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.command == "help" or not args.command:
        print("""
        贪吃蛇强化学习项目使用指南
        ========================
        
        可用命令:
        
        1. 训练模型:
           python main.py train --algo [dqn|ppo|a2c|advanced|ppo-advanced] --width 10 --height 10 --timesteps 1000000
           
        2. 使用模型玩游戏:
           python main.py play --algo [dqn|ppo|a2c|advanced|ppo-advanced] [--model-path 模型路径]
           
        3. 比较不同算法的性能:
           python main.py compare --eval-episodes 50 --models 模型1路径 模型2路径 ...
           
        4. 启动演示窗口:
           python main.py demo [--model-path 模型路径]
           
        5. 安装依赖:
           python main.py install
        """)

if __name__ == "__main__":
    main()
