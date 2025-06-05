import argparse
import os
import tkinter as tk
from tkinter import Canvas
import time

from stable_baselines3 import DQN, PPO, A2C
from env.snake_env import SnakeEnv

def parse_args():
    parser = argparse.ArgumentParser(description="用Stable Baselines 3训练的模型玩贪吃蛇")
    parser.add_argument("--algo", type=str, default="ppo", choices=["dqn", "ppo", "a2c"], help="使用的算法")
    parser.add_argument("--model-path", type=str, default=None, 
                        help="模型文件路径（如未提供，将使用默认路径）")
    parser.add_argument("--width", type=int, default=10, help="环境宽度")
    parser.add_argument("--height", type=int, default=10, help="环境高度")
    parser.add_argument("--max-steps", type=int, default=1000, help="最大步数")
    return parser.parse_args()

def visualize_game_sb3(env, model, max_steps=1000):
    """
    使用tkinter可视化Stable Baselines 3模型玩贪吃蛇游戏
    """
    # 设定窗口尺寸
    cell_size = 20
    width = env.width * cell_size
    height = env.height * cell_size
    
    # 创建tkinter窗口
    root = tk.Tk()
    root.title(f"{args.algo.upper()} 贪吃蛇")
    canvas = Canvas(root, width=width, height=height, bg="white")
    canvas.pack()
    
    # 设置游戏状态
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    def update():
        nonlocal obs, done, truncated, total_reward, steps
        
        if not (done or truncated) and steps < max_steps:
            # 选择动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            
            # 清除画布
            canvas.delete("all")
            
            # 绘制蛇和食物
            # 蛇身
            for x, y in env.snake_body[1:]:
                canvas.create_rectangle(
                    x * cell_size, y * cell_size, 
                    (x + 1) * cell_size, (y + 1) * cell_size, 
                    fill="black"
                )
            
            # 蛇头
            head_x, head_y = env.snake_body[0]
            canvas.create_rectangle(
                head_x * cell_size, head_y * cell_size, 
                (head_x + 1) * cell_size, (head_y + 1) * cell_size, 
                fill="blue"
            )
            
            # 食物
            food_x, food_y = env.food_position
            canvas.create_oval(
                food_x * cell_size, food_y * cell_size, 
                (food_x + 1) * cell_size, (food_y + 1) * cell_size, 
                fill="red"
            )
            
            # 显示分数和奖励
            canvas.create_text(
                10, 10, text=f"得分: {len(env.snake_body) - 2}", anchor="nw", fill="green"
            )
            canvas.create_text(
                10, 30, text=f"总奖励: {total_reward:.1f}", anchor="nw", fill="green"
            )
            
            # 安排下一次更新
            root.after(100, update)
        else:
            # 游戏结束
            canvas.create_text(
                width // 2, height // 2, 
                text=f"游戏结束\n得分: {len(env.snake_body) - 2}\n总奖励: {total_reward:.1f}", 
                font=("Helvetica", 16), fill="red"
            )
    
    # 开始更新循环
    root.after(100, update)
    
    # 启动主循环
    root.mainloop()

def play():
    # 解析命令行参数
    global args
    args = parse_args()
    
    # 创建环境
    env = SnakeEnv(width=args.width, height=args.height)
    
    # 确定模型路径
    if args.model_path is None:
        model_path = f"d:/project/AIsnake/models/sb3/{args.algo}_snake_final.zip"
    else:
        model_path = args.model_path
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        print("请先训练模型或提供有效的模型路径")
        return
    
    # 加载模型
    if args.algo.lower() == "dqn":
        model = DQN.load(model_path)
    elif args.algo.lower() == "ppo":
        model = PPO.load(model_path)
    elif args.algo.lower() == "a2c":
        model = A2C.load(model_path)
    else:
        raise ValueError(f"不支持的算法: {args.algo}")
    
    print(f"已加载模型: {model_path}")
    
    # 启动可视化游戏
    print("启动游戏...")
    visualize_game_sb3(env, model, max_steps=args.max_steps)

if __name__ == "__main__":
    play()
