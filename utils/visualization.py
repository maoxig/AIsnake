import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import Canvas

def evaluate_agent(env, agent, num_episodes=10, render=False):
    """
    评估代理在多个环境中的表现
    
    Args:
        env: 游戏环境
        agent: 强化学习代理
        num_episodes: 评估的回合数
        render: 是否渲染环境
        
    Returns:
        平均奖励和平均长度
    """
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step = 0
        
        while not (done or truncated):
            # 选择动作（不使用探索）
            action = agent.select_action(obs, training=False)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # 如果需要，渲染环境
            if render and episode == num_episodes - 1:  # 只渲染最后一个回合
                env.render()
                time.sleep(0.1)
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
    
    return np.mean(total_rewards), np.mean(episode_lengths)

def plot_training_progress(rewards, lengths, window=100):
    """
    绘制训练进度
    
    Args:
        rewards: 每回合的奖励列表
        lengths: 每回合的长度列表
        window: 平滑窗口大小
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, label='原始奖励')
    
    # 计算滑动平均
    if len(rewards) >= window:
        smoothed_rewards = [np.mean(rewards[i:i+window]) for i in range(len(rewards)-window)]
        plt.plot(range(window//2, len(smoothed_rewards) + window//2), smoothed_rewards, label=f'平滑 (窗口={window})')
    
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.title('训练奖励')
    plt.legend()
    plt.grid()
    
    # 绘制长度曲线
    plt.subplot(1, 2, 2)
    plt.plot(lengths, alpha=0.6, label='原始长度')
    
    # 计算滑动平均
    if len(lengths) >= window:
        smoothed_lengths = [np.mean(lengths[i:i+window]) for i in range(len(lengths)-window)]
        plt.plot(range(window//2, len(smoothed_lengths) + window//2), smoothed_lengths, label=f'平滑 (窗口={window})')
    
    plt.xlabel('回合')
    plt.ylabel('回合长度')
    plt.title('回合长度')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('d:/project/AIsnake/training_progress.png')
    plt.close()

def visualize_game(env, agent, max_steps=1000):
    """
    使用tkinter可视化贪吃蛇游戏
    
    Args:
        env: 游戏环境
        agent: 强化学习代理
        max_steps: 最大步数
    """
    # 设定窗口尺寸和其他参数
    cell_size = 20
    width = env.width * cell_size
    height = env.height * cell_size
    
    # 创建tkinter窗口
    root = tk.Tk()
    root.title("DQN 贪吃蛇")
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
            action = agent.select_action(obs, training=False)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
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
            
            # 显示分数
            canvas.create_text(
                10, 10, text=f"得分: {len(env.snake_body) - 2}", anchor="nw", fill="green"
            )
            
            # 安排下一次更新
            root.after(100, update)
        else:
            # 游戏结束
            canvas.create_text(
                width // 2, height // 2, 
                text=f"游戏结束\n得分: {len(env.snake_body) - 2}", 
                font=("Helvetica", 16), fill="red"
            )
    
    # 开始更新循环
    root.after(100, update)
    
    # 启动主循环
    root.mainloop()
