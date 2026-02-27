import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size, action_size, device=None):
        # 设备配置：优先使用传入的 device
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Agent using device: {self.device}")
        self.state_size = state_size
        self.action_size = action_size


        # 确保 build_model 内部没有写死 CPU/GPU 操作
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)

        # 经验回放缓冲区
        self.memory = deque(maxlen=20000)

        # 超参数
        self.gamma = 0.95  # 折扣因子
        self.learning_rate = 0.0005
        self.batch_size = 128

        # === 探索策略参数 (Epsilon Greedy) ===
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === 神经网络构建 ===
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 初始化目标网络
        self.update_target_model()

        # 损失函数 (使用 Huber Loss 通常比 MSE 在处理 Q 值时更稳定，不过 MSE 也可以)
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        """
        构建深度Q网络。
        修改：移除了 Dropout，加入了 BatchNorm1d 以加速收敛。
        """
        model = nn.Sequential(
            nn.Linear(self.state_size, 512),
            # nn.BatchNorm1d(512),  # BN 层有助于数值稳定
            nn.ReLU(),

            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, self.action_size)
        )
        return model

    def update_target_model(self):
        """将主网络的权重复制到目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        # 稍微节省内存，存之前可以不用转 tensor，取出来再转
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """根据当前状态选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # 处理输入状态
        # 这里的 state 可能是 float64 (numpy默认)，转为 float32
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # BN层需要 batch dimension > 1 才能训练，但在 eval 模式下没问题
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state_tensor)
        self.model.train()  # 切换回训练模式

        return torch.argmax(act_values[0]).item()

    def replay(self):
        """经验回放：DDQN 算法实现"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        # 批量转换数据，确保类型为 float32
        states = torch.FloatTensor(np.array([x[0] for x in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([x[1] for x in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([x[2] for x in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([x[4] for x in minibatch])).to(self.device)

        # 1. 计算当前 Q 值
        # model 输出 (batch, action_size) -> gather 选出实际执行动作的 Q 值
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 2. 计算目标 Q 值 (Double DQN 逻辑)
        with torch.no_grad():
            # A. 使用【主网络】选择最佳动作 (Argmax)
            best_actions = self.model(next_states).argmax(1).unsqueeze(1)

            # B. 使用【目标网络】评估该动作的价值
            next_q = self.target_model(next_states).gather(1, best_actions).squeeze()

            # 计算 Target
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 3. 计算 Loss 并更新
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def update_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


