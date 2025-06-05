import argparse
import os
import time
import numpy as np
import tkinter as tk
from tkinter import Canvas, Label, Button, Frame
from stable_baselines3 import PPO

from env.snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT
from train_ppo_advanced import SnakeWrapper


class DemoWindow:
    """
    贪吃蛇演示窗口，支持AI模式和人类模式
    """
    def __init__(self, model_path=None, width=10, height=10):
        self.window_width = 800
        self.window_height = 600
        self.cell_size = 30
        
        self.grid_width = width
        self.grid_height = height
        self.canvas_width = width * self.cell_size
        self.canvas_height = height * self.cell_size
        
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path)
            print(f"已加载模型: {model_path}")
        
        # 创建环境
        self.env = SnakeEnv(width=width, height=height)
        self.wrapped_env = SnakeWrapper(self.env)
        
        # 游戏状态
        self.is_running = False
        self.ai_mode = True
        self.game_speed = 150  # 毫秒
        self.obs = None
        self.score = 0
        self.steps = 0
        self.high_score = 0
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("贪吃蛇强化学习演示")
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.resizable(False, False)
        
        # 创建左侧游戏画布
        game_frame = Frame(self.root)
        game_frame.place(x=20, y=20)
        
        self.canvas = Canvas(game_frame, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()
        
        # 创建右侧控制面板
        control_frame = Frame(self.root)
        control_frame.place(x=self.canvas_width + 40, y=20)
        
        # 分数显示
        self.score_label = Label(control_frame, text="分数: 0", font=("Arial", 16))
        self.score_label.pack(pady=10)
        
        self.high_score_label = Label(control_frame, text="最高分: 0", font=("Arial", 16))
        self.high_score_label.pack(pady=10)
        
        self.steps_label = Label(control_frame, text="步数: 0", font=("Arial", 16))
        self.steps_label.pack(pady=10)
        
        # AI/人类模式切换
        self.mode_button = Button(control_frame, text="切换到人类模式", command=self.toggle_mode)
        self.mode_button.pack(pady=10)
        
        # 速度控制
        speed_frame = Frame(control_frame)
        speed_frame.pack(pady=10)
        
        Label(speed_frame, text="游戏速度:").pack(side=tk.LEFT)
        
        speed_buttons = [
            ("慢", 300),
            ("中", 150),
            ("快", 80),
            ("极快", 30)
        ]
        
        for text, speed in speed_buttons:
            Button(speed_frame, text=text, command=lambda s=speed: self.set_speed(s)).pack(side=tk.LEFT, padx=5)
        
        # 开始/暂停按钮
        self.start_button = Button(control_frame, text="开始游戏", command=self.toggle_game)
        self.start_button.pack(pady=10)
        
        # 重置按钮
        self.reset_button = Button(control_frame, text="重置游戏", command=self.reset_game)
        self.reset_button.pack(pady=10)
        
        # 当前模式显示
        self.mode_label = Label(control_frame, text="当前模式: AI", font=("Arial", 14))
        self.mode_label.pack(pady=10)
        
        # 游戏说明
        help_text = """
        AI 模式: AI会自动玩贪吃蛇
        
        人类模式控制:
        - 方向键: 控制蛇的移动
        - W/A/S/D: 控制蛇的移动
        
        游戏规则:
        - 吃到食物: +1分
        - 撞墙或撞到自己: 游戏结束
        """
        
        help_label = Label(control_frame, text=help_text, justify=tk.LEFT, font=("Arial", 10))
        help_label.pack(pady=20)
        
        # 绑定按键事件(人类模式)
        self.root.bind("<KeyPress>", self.on_key_press)
        
        # 初始化游戏
        self.reset_game()
    
    def toggle_mode(self):
        """切换AI和人类模式"""
        self.ai_mode = not self.ai_mode
        
        if self.ai_mode:
            self.mode_button.config(text="切换到人类模式")
            self.mode_label.config(text="当前模式: AI")
        else:
            self.mode_button.config(text="切换到AI模式") 
            self.mode_label.config(text="当前模式: 人类")
        
        # 重置游戏
        self.reset_game()
    
    def set_speed(self, speed):
        """设置游戏速度"""
        self.game_speed = speed
        print(f"游戏速度设置为: {speed}ms")
    
    def toggle_game(self):
        """开始或暂停游戏"""
        self.is_running = not self.is_running
        
        if self.is_running:
            self.start_button.config(text="暂停游戏")
            self.update_game()
        else:
            self.start_button.config(text="继续游戏")
    
    def reset_game(self):
        """重置游戏状态"""
        self.obs, _ = self.wrapped_env.reset()
        self.score = 0
        self.steps = 0
        self.score_label.config(text=f"分数: {self.score}")
        self.steps_label.config(text=f"步数: {self.steps}")
        
        # 绘制初始状态
        self.render_game()
    
    def on_key_press(self, event):
        """处理键盘事件(人类模式)"""
        if not self.is_running or self.ai_mode:
            return
        
        key = event.keysym.lower()
        
        if key in ["up", "w"]:
            self.wrapped_env.step(UP)
        elif key in ["down", "s"]:
            self.wrapped_env.step(DOWN)
        elif key in ["left", "a"]:
            self.wrapped_env.step(LEFT)
        elif key in ["right", "d"]:
            self.wrapped_env.step(RIGHT)
    
    def update_game(self):
        """更新游戏状态"""
        if not self.is_running:
            return
        
        # 执行一步动作
        if self.ai_mode and self.model:
            # AI模式：使用模型预测动作
            action, _ = self.model.predict(self.obs, deterministic=True)
            self.obs, reward, done, truncated, info = self.wrapped_env.step(int(action))
        else:
            # 人类模式：动作已在键盘事件中处理，这里只更新状态
            pass
        
        # 更新游戏状态
        self.score = len(self.wrapped_env.env.snake_body) - 2
        self.steps += 1
        
        # 更新分数显示
        self.score_label.config(text=f"分数: {self.score}")
        self.steps_label.config(text=f"步数: {self.steps}")
        
        if self.score > self.high_score:
            self.high_score = self.score
            self.high_score_label.config(text=f"最高分: {self.high_score}")
        
        # 渲染游戏
        self.render_game()
        
        # 检查游戏是否结束
        if self.wrapped_env.env.is_dead():
            print(f"游戏结束! 得分: {self.score}, 步数: {self.steps}")
            self.is_running = False
            self.start_button.config(text="开始新游戏")
            self.reset_game()
        else:
            # 继续游戏循环
            self.root.after(self.game_speed, self.update_game)
    
    def render_game(self):
        """渲染游戏画面"""
        self.canvas.delete("all")
        
        # 绘制网格
        for i in range(self.grid_width + 1):
            self.canvas.create_line(
                i * self.cell_size, 0, 
                i * self.cell_size, self.canvas_height,
                fill="#CCCCCC"
            )
        
        for i in range(self.grid_height + 1):
            self.canvas.create_line(
                0, i * self.cell_size,
                self.canvas_width, i * self.cell_size,
                fill="#CCCCCC"
            )
        
        # 绘制蛇身
        for i, (x, y) in enumerate(self.wrapped_env.env.snake_body):
            if i == 0:  # 蛇头
                self.canvas.create_rectangle(
                    x * self.cell_size, y * self.cell_size,
                    (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                    fill="blue", outline="black"
                )
            else:  # 蛇身
                self.canvas.create_rectangle(
                    x * self.cell_size, y * self.cell_size,
                    (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                    fill="green", outline="black"
                )
        
        # 绘制食物
        food_x, food_y = self.wrapped_env.env.food_position
        self.canvas.create_oval(
            food_x * self.cell_size, food_y * self.cell_size,
            (food_x + 1) * self.cell_size, (food_y + 1) * self.cell_size,
            fill="red", outline="black"
        )
    
    def run(self):
        """运行主循环"""
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="贪吃蛇AI演示")
    parser.add_argument("--model-path", type=str, default=None, help="PPO模型路径")
    parser.add_argument("--width", type=int, default=10, help="游戏宽度")
    parser.add_argument("--height", type=int, default=10, help="游戏高度")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 默认模型路径
    if args.model_path is None:
        args.model_path = "d:/project/AIsnake/models/sb3/ppo_snake_final.zip"
    
    print("启动贪吃蛇AI演示...")
    app = DemoWindow(model_path=args.model_path, width=args.width, height=args.height)
    app.run()

if __name__ == "__main__":
    main()
