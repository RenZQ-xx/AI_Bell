import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # 经验回放缓冲区
        self.memory = deque(maxlen=20000)

        # 超参数
        self.gamma = 0.95  # 折扣因子
        self.learning_rate = 0.0005
        self.batch_size = 128

        # === 探索策略参数 (Epsilon Greedy) ===
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        # 衰减率调整为 0.997 (针对“一局一衰减”)
        # 计算逻辑: 0.997^1500 ≈ 0.011
        # 这意味着大约在第 1500 局时，探索率会降到最低，剩下 500 局用于纯利用。
        self.epsilon_decay = 0.997

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === 神经网络构建 ===
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 初始化目标网络
        self.update_target_model()

        # 损失函数
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        """构建深度Q网络，保持较深的网络结构以处理复杂的64维状态"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def update_target_model(self):
        """将主网络的权重复制到目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """根据当前状态选择动作 (Epsilon-Greedy 策略)"""
        # 探索：随机选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # 利用：选择Q值最高的动作
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self):
        """经验回放：从记忆中采样并训练网络"""
        if len(self.memory) < self.batch_size:
            return

        # 随机采样一个批次
        minibatch = random.sample(self.memory, self.batch_size)

        # === 性能优化关键点 ===
        # 先将 list 转为 numpy array，再转 Tensor
        # 这样可以避免 "Creating a tensor from a list of numpy.ndarrays is extremely slow" 警告
        states = torch.FloatTensor(np.array([x[0] for x in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([x[1] for x in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([x[2] for x in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([x[4] for x in minibatch])).to(self.device)

        # 1. 计算当前 Q 值: Q(s, a)
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 2. 计算目标 Q 值: R + gamma * max Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 3. 计算 Loss
        loss = self.loss_fn(current_q, target_q)

        # 4. 反向传播优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # 注意：这里不再进行 epsilon 衰减！

    def update_epsilon(self):
        """
        新的衰减函数。
        请在 train.py 的每一局(Episode)结束时调用此函数，而不是每一步。
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
