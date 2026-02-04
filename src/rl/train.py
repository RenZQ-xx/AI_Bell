from agent import DQNAgent
from points import PointSetEnv
import numpy as np
import matplotlib.pyplot as plt  # 确保导入放在这里



def train_dqn(episodes=2000, max_steps=50):
    env = PointSetEnv()
    state_size = 64  # 8*8
    action_size = 21

    agent = DQNAgent(state_size, action_size)
    boundary_sets = []
    episode_rewards = []

    # 【修复】定义 success_history，防止 return 时报错
    # 记录格式: (episode_index, steps_taken)
    success_history = []

    # 动作名称，用于调试
    action_names = [
        "AB-I", "AB-NOT", "AB-XOR", "AB-NOR",
        "B-0-A-FA+", "B-0-A-FA-", "B-y-A-FA+", "B-y-A-FA-", "B-ny-A-FA+", "B-ny-A-FA-", "B-1-A-FA+", "B-1-A-FA-",
        "A-0-B-FA+", "A-0-B-FA-", "A-y-B-FA+", "A-y-B-FA-", "A-ny-B-FA+", "A-ny-B-FA-", "A-1-B-FA+", "A-1-B-FA-",
        "AB-swap"
    ]

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            # 【修改】将 replay 移入循环内，标准 DQN 做法
            # 每走一步，如果经验够了，就训练一次网络
            if len(agent.memory) > agent.batch_size:
                agent.replay()

            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                boundary_sets.append(env.state.copy())
                # 【新增】记录成功的数据，用于后续画散点图
                success_history.append((e, step_count))

                print(
                    f"Episode {e}/{episodes}, found boundary set {len(boundary_sets)} with action {action_names[action]}")
                break

        # 【关键修改】一局结束了，衰减一次 Epsilon
        # 配合 agent.py 中的 update_epsilon 函数
        agent.update_epsilon()

        episode_rewards.append(total_reward)

        # 定期更新目标网络
        if e % 20 == 0:
            agent.update_target_model()

        if e % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {e}, Epsilon: {agent.epsilon:.3f}, Avg Reward (last 100): {avg_reward:.3f}")

    # 去重并输出所有边界点集
    unique_boundaries = []
    for boundary in boundary_sets:
        # 将 numpy 数组转换为 tuple 以便于哈希和比较
        boundary_tuple = tuple(tuple(point) for point in boundary)
        is_duplicate = False
        for ub in unique_boundaries:
            if np.allclose(boundary, ub, atol=1e-6):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_boundaries.append(boundary)

    print(f"\nFound {len(unique_boundaries)} unique boundary hyperplanes:")

    return unique_boundaries, episode_rewards, success_history


# 运行训练
if __name__ == "__main__":

    # 注意：这里接收 3 个返回值
    boundary_sets, rewards, success_history = train_dqn()

    # === 修改点 3: 绘制 3 张图 (Reward, Avg Reward, Steps) ===
    plt.figure(figsize=(18, 5))  # 调宽画布

    # 图 1: 原始 Reward
    plt.subplot(1, 3, 1)
    plt.plot(rewards, alpha=0.5, color='gray')
    plt.title("Episode Rewards (Raw)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # 图 2: 平均 Reward
    plt.subplot(1, 3, 2)
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(moving_avg, color='blue')
    plt.title(f"Moving Average (Window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")

    # 图 3: 收敛步数 (Steps to Solution)
    # 这是最重要的学术指标图
    plt.subplot(1, 3, 3)

    if len(success_history) > 0:
        # 解压数据: x轴是Episode编号, y轴是消耗步数
        x_vals, y_vals = zip(*success_history)

        # 使用散点图 (Scatter)，因为成功是离散事件
        plt.scatter(x_vals, y_vals, s=10, c='green', alpha=0.6, label='Success')

        # 如果数据足够多，画一个趋势线
        if len(y_vals) > 20:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            plt.plot(x_vals, p(x_vals), "r--", label='Trend')

        plt.title("Steps to Find CHSH (Lower is Better)")
        plt.xlabel("Episode")
        plt.ylabel("Steps Taken")
        plt.ylim(0, 35)  # 稍微限制一下Y轴，方便看清低步数区域
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No Successes Found", ha='center')
        plt.title("Steps to Find CHSH")

    plt.tight_layout()
    plt.show()

