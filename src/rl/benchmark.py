import matplotlib.pyplot as plt
import numpy as np
import random
import time

# 导入你的环境
from points import PointSetEnv
# 导入你写好的 DQN 训练函数
from train import train_dqn


def run_random_baseline(episodes=2000, max_steps=50):
    """
    完全随机的基线测试
    """
    env = PointSetEnv()
    # 假设 action_size 是 21，需要和 train.py 保持一致
    action_size = 21

    success_history = []

    print(f"--- Starting Random Baseline ({episodes} episodes) ---")
    start_time = time.time()

    for e in range(episodes):
        state = env.reset()
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            # 纯随机动作
            action = random.randrange(action_size)
            next_state, reward, done = env.step(action)
            step_count += 1

            if done:
                success_history.append((e, step_count))
                break

        if (e + 1) % 500 == 0:
            print(f"Random Baseline: Finished {e + 1}/{episodes} episodes")

    duration = time.time() - start_time
    print(f"Random Baseline Completed in {duration:.2f}s. Solutions found: {len(success_history)}")
    return success_history


def plot_comparison(dqn_data, random_data, total_episodes):
    """
    画图对比：DQN vs Random
    """
    plt.figure(figsize=(12, 7))

    # 1. 画 Random Baseline (灰色背景)
    if random_data:
        r_x, r_y = zip(*random_data)
        plt.scatter(r_x, r_y, c='lightgray', alpha=0.5, s=20, label=f'Random (Found {len(random_data)})')

    # 2. 画 DQN (蓝色前景)
    if dqn_data:
        d_x, d_y = zip(*dqn_data)
        plt.scatter(d_x, d_y, c='blue', alpha=0.6, s=20, edgecolors='none', label=f'DQN (Found {len(dqn_data)})')

        # 添加趋势线 (简单的线性拟合)
        if len(dqn_data) > 10:
            z = np.polyfit(d_x, d_y, 1)
            p = np.poly1d(z)
            # 在图的范围内画线
            plt.plot(d_x, p(d_x), "r--", linewidth=2, label='DQN Trend')

    plt.title('Performance Comparison: DQN Agent vs Random Search')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Solution')
    plt.xlim(0, total_episodes)
    plt.ylim(0, 55)  # 稍微留点余地
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)

    # 保存图片而不是直接显示，方便在服务器上跑
    plt.savefig('benchmark_result.png')
    print("Plot saved as 'benchmark_result.png'")
    plt.show()


if __name__ == "__main__":
    # 设置参数
    N_EPISODES = 2000
    MAX_STEPS = 50

    print("========================================")
    print("STEP 1: Running DQN Agent...")
    print("========================================")
    # 注意：这里调用 train_dqn，它会重新训练一遍。
    # 如果你想加载训练好的模型来测试，需要修改 train.py 支持 'eval' 模式
    # 但目前我们先重新跑一遍看训练曲线。
    _, _, dqn_history = train_dqn(episodes=N_EPISODES, max_steps=MAX_STEPS)

    print("\n========================================")
    print("STEP 2: Running Random Baseline...")
    print("========================================")
    random_history = run_random_baseline(episodes=N_EPISODES, max_steps=MAX_STEPS)

    print("\n========================================")
    print("STEP 3: Generating Plot...")
    print("========================================")
    plot_comparison(dqn_history, random_history, N_EPISODES)
