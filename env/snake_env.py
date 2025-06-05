import numpy as np
import gymnasium as gym
from gymnasium import spaces
import copy
from typing import Tuple, List, Dict, Any, Optional

# 方向常量
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class SnakeEnv(gym.Env):
    """
    贪吃蛇强化学习环境
    遵循OpenAI Gymnasium接口
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, width=10, height=10, render_mode=None):
        super().__init__()
        
        self.width = width  # 游戏场地宽度（格子数）
        self.height = height  # 游戏场地高度（格子数）
        self.render_mode = render_mode
        
        # 定义动作空间（上下左右四个方向）
        self.action_space = spaces.Discrete(4)
        
        # 定义观察空间：游戏板状态 + 蛇头位置 + 食物位置
        # 3个通道: 0=空白, 1=蛇身, 2=蛇头, 3=食物
        self.observation_space = spaces.Box(
            low=0, high=3,
            shape=(self.width, self.height),
            dtype=np.uint8
        )
        
        # 游戏状态
        self.snake_body = None
        self.snake_direction = None
        self.food_position = None
        self.steps = None
        self.max_steps = 100 * (self.width * self.height)  # 防止游戏无限进行
        
        # 渲染相关
        self.window = None
        self.clock = None
        
    def _get_obs(self):
        """
        返回游戏状态的观察值
        """
        obs = np.zeros((self.width, self.height), dtype=np.uint8)
        
        # 标记蛇身
        for x, y in self.snake_body[1:]:
            obs[x, y] = 1
        
        # 标记蛇头
        head_x, head_y = self.snake_body[0]
        obs[head_x, head_y] = 2
        
        # 标记食物
        food_x, food_y = self.food_position
        obs[food_x, food_y] = 3
        
        return obs
    
    def _get_info(self):
        """
        返回当前游戏状态的附加信息
        """
        return {
            "snake_length": len(self.snake_body),
            "head_position": self.snake_body[0],
            "food_position": self.food_position
        }

    def reset(self, seed=None, options=None):
        """
        重置游戏状态
        """
        super().reset(seed=seed)
        
        # 初始化蛇的位置
        self.snake_body = [(self.width // 4, self.height // 4), 
                           (self.width // 4, self.height // 4 + 1)]
        self.snake_direction = RIGHT
        
        # 初始化食物位置（随机）
        self.food_position = self._generate_food()
        
        self.steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _generate_food(self):
        """生成新的食物，确保不在蛇身上"""
        while True:
            food = (
                self.np_random.integers(0, self.width),
                self.np_random.integers(0, self.height)
            )
            if food not in self.snake_body:
                return food
    
    def step(self, action):
        """
        执行动作，返回新的状态、奖励等
        action: 0=上, 1=下, 2=左, 3=右
        """
        self.steps += 1
        
        # 更新蛇的方向（防止直接反向移动）
        current_direction = self.snake_direction
        if (action == UP and current_direction != DOWN) or \
           (action == DOWN and current_direction != UP) or \
           (action == LEFT and current_direction != RIGHT) or \
           (action == RIGHT and current_direction != LEFT):
            self.snake_direction = action
        
        # 获取蛇头
        head_x, head_y = self.snake_body[0]
        
        # 根据方向移动蛇头
        if self.snake_direction == UP:
            new_head = (head_x, head_y - 1)
        elif self.snake_direction == DOWN:
            new_head = (head_x, head_y + 1)
        elif self.snake_direction == LEFT:
            new_head = (head_x - 1, head_y)
        elif self.snake_direction == RIGHT:
            new_head = (head_x + 1, head_y)
        
        # 默认奖励
        reward = 0
        terminated = False
        
        # 检查是否撞墙
        if (new_head[0] < 0 or new_head[0] >= self.width or
                new_head[1] < 0 or new_head[1] >= self.height):
            reward = -1.0  # 撞墙惩罚
            terminated = True
        # 检查是否撞到自己
        elif new_head in self.snake_body:
            reward = -1.0  # 撞到自己惩罚
            terminated = True
        else:
            # 移动蛇
            self.snake_body.insert(0, new_head)
            
            # 检查是否吃到食物
            if new_head == self.food_position:
                reward = 1.0  # 吃到食物奖励
                self.food_position = self._generate_food()
            else:
                # 没吃到食物则移除尾部
                self.snake_body.pop()
        
        # 检查是否超过最大步数
        truncated = (self.steps >= self.max_steps)
        
        # 获取观察和信息
        observation = self._get_obs()
        info = self._get_info()
        
        # 额外奖励：接近食物
        if not terminated:
            head_x, head_y = self.snake_body[0]
            food_x, food_y = self.food_position
            prev_head_x, prev_head_y = head_x, head_y
            if self.snake_direction == UP:
                prev_head_y += 1
            elif self.snake_direction == DOWN:
                prev_head_y -= 1
            elif self.snake_direction == LEFT:
                prev_head_x += 1
            elif self.snake_direction == RIGHT:
                prev_head_x -= 1
                
            prev_dist = abs(prev_head_x - food_x) + abs(prev_head_y - food_y)
            curr_dist = abs(head_x - food_x) + abs(head_y - food_y)
            
            if curr_dist < prev_dist:
                reward += 0.1  # 接近食物的小奖励
            elif curr_dist > prev_dist:
                reward -= 0.1  # 远离食物的小惩罚
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        渲染游戏画面
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        """
        渲染单帧游戏画面，返回RGB数组
        """
        # 简单起见，这里只返回游戏画面的矩阵表示
        # 实际应用中可以使用pygame等库进行图形渲染
        frame = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        
        # 画蛇身 (绿色)
        for x, y in self.snake_body[1:]:
            frame[x, y] = [0, 255, 0]
        
        # 画蛇头 (蓝色)
        head_x, head_y = self.snake_body[0]
        frame[head_x, head_y] = [0, 0, 255]
        
        # 画食物 (红色)
        food_x, food_y = self.food_position
        frame[food_x, food_y] = [255, 0, 0]
        
        return frame
    
    def close(self):
        """
        关闭环境
        """
        if self.window:
            self.window.close()
            self.window = None
