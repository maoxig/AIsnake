# 基于强化学习的贪吃蛇 AI

这个项目将经典的贪吃蛇游戏改造成强化学习环境，并使用深度强化学习方法训练AI来自动玩游戏。项目实现了自定义DQN和使用Stable Baselines 3库的多种算法，包括DQN、PPO和A2C，以及高级特征提取网络。

## 项目结构

```
AIsnake/
├── __init__.py                # 项目初始化文件
├── astar_snake_example.py     # 原始基于A*的贪吃蛇实现
├── main.py                    # 主入口脚本
├── README.md                  # 项目文档
├── requirements.txt           # 项目依赖
├── train.py                   # 使用自定义DQN训练脚本
├── train_sb3.py               # Stable Baselines 3训练脚本
├── train_ppo_advanced.py      # 高级PPO训练脚本
├── train_advanced_features.py # 高级特征提取器训练脚本
├── play.py                    # 使用自定义DQN玩游戏脚本
├── play_sb3.py                # Stable Baselines 3游戏脚本
├── snake_demo.py              # 交互式演示脚本
├── compare_algorithms.py      # 算法比较脚本
├── env/                       # 环境模块
│   ├── __init__.py
│   └── snake_env.py           # 贪吃蛇Gym环境
├── models/                    # 模型模块
│   ├── dqn_agent.py           # 自定义DQN代理
│   ├── networks.py            # 基础神经网络定义
│   └── advanced_networks.py   # 高级特征提取器
└── utils/                     # 工具模块
    └── visualization.py       # 评估和可视化工具
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 使用自定义DQN训练模型

```bash
python train.py --width 10 --height 10 --total-timesteps 1000000
```

参数说明：
- `--width`: 游戏环境宽度（网格数）
- `--height`: 游戏环境高度（网格数）
- `--total-timesteps`: 训练的总步数

### 使用Stable Baselines 3训练模型

```bash
python train_sb3.py --algo ppo --width 10 --height 10 --total-timesteps 1000000
```

参数说明：
- `--algo`: 使用的算法，可选 'dqn', 'ppo', 'a2c'
- `--width`: 游戏环境宽度（网格数）
- `--height`: 游戏环境高度（网格数）
- `--total-timesteps`: 训练的总步数

### 使用训练好的模型玩游戏

自定义DQN模型：

```bash
python play.py --model-path models/saved/dqn_snake_final.pt
```

Stable Baselines 3模型：

```bash
python play_sb3.py --algo ppo --model-path models/sb3/ppo_snake_final.zip
```

## 深度强化学习方法比较

本项目中实现的深度强化学习方法：

1. **自定义DQN (Deep Q-Network)** - 使用经验回放和目标网络的值函数近似方法。

2. **Stable Baselines 3实现的算法**:
   - **DQN (Deep Q-Network)** - 基于Q值的离线策略算法
   - **PPO (Proximal Policy Optimization)** - 基于策略的在线算法，在稳定性和简单性之间取得良好平衡
   - **A2C (Advantage Actor-Critic)** - 结合了值函数和策略的演员-评论家方法

## 项目原理

贪吃蛇环境建模为马尔可夫决策过程 (MDP)：

- **状态**: 游戏板表示为二维网格，包含蛇的位置、蛇头位置和食物位置
- **动作**: 上、下、左、右四个移动方向
- **奖励**: 
  - 吃到食物: +1.0
  - 撞墙或撞到自己: -1.0
  - 朝食物方向移动: +0.1
  - 远离食物: -0.1
- **终止条件**: 蛇撞墙或撞到自己

## 使用主入口脚本

本项目提供了一个统一的命令行接口 `main.py`，可以方便地执行各种任务：

```bash
# 安装依赖
python main.py install

# 训练模型 (dqn, ppo, a2c, advanced, ppo-advanced)
python main.py train --algo ppo --width 10 --height 10 --timesteps 1000000

# 使用训练好的模型玩游戏
python main.py play --algo ppo --model-path models/sb3/ppo_snake_final.zip

# 启动交互式演示
python main.py demo --model-path models/sb3/ppo_snake_final.zip

# 比较不同算法的性能
python main.py compare --models models/saved/dqn_snake_final.pt models/sb3/ppo_snake_final.zip
```

## 高级功能

本项目还实现了以下高级功能：

1. **卷积神经网络特征提取器** - 专门为处理贪吃蛇的网格状态设计，能够有效提取空间特征
2. **自注意力机制** - 用于学习蛇头、食物和障碍物之间的关系
3. **奖励塑形** - 精心设计的奖励函数，使智能体学习更高效
4. **交互式演示界面** - 可以在AI和人类玩家之间切换
5. **算法比较工具** - 自动对比不同强化学习算法的性能并生成图表

## 与原始A*算法的比较

原始的A*算法是一种基于搜索的确定性解决方案，而深度强化学习是一种通过试错学习的方法。A*算法可以找到到达食物的最短路径，但在处理蛇身避碰等复杂情境时需要额外的启发式策略。

深度强化学习在大规模状态空间中更灵活，能够自动学习复杂策略，但需要大量训练样本和计算资源。

## 项目总结

与原始的A*算法相比，强化学习方法具有以下优势：

1. **学习而非编程** - RL智能体通过尝试和错误学习，无需显式编程复杂的策略
2. **适应性** - 可以适应不同的环境设置和障碍物
3. **泛化能力** - 在训练中未见的场景中也能表现良好

这个项目展示了如何将传统的基于搜索的游戏AI转换为基于学习的AI系统，并通过现代深度强化学习方法提升其性能。
