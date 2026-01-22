from agent import DQNAgent
from points import PointSetEnv
import numpy as np

def train_dqn(episodes=2000, max_steps=50):
    env = PointSetEnv()
    state_size = 64  # 8*8
    action_size = 21
    
    agent = DQNAgent(state_size, action_size)
    boundary_sets = []
    episode_rewards = []
    
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
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                boundary_sets.append(env.state.copy())
                print(f"Episode {e}/{episodes}, found boundary set {len(boundary_sets)} with action {action_names[action]}")
                break
        
        # 定期回放经验
        if len(agent.memory) > agent.batch_size:
            for _ in range(3):  # 多次回放
                agent.replay()
        
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
        # 转换为可哈希的元组
        boundary_tuple = tuple(tuple(point) for point in boundary)
        
        # 检查是否已经存在（考虑浮点误差）
        is_duplicate = False
        for ub in unique_boundaries:
            if np.allclose(boundary, ub, atol=1e-6):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_boundaries.append(boundary)
    
    print(f"\nFound {len(unique_boundaries)} unique boundary hyperplanes:")
    for i, boundary in enumerate(unique_boundaries):
        print(f"\nBoundary Set {i+1}:")
        print("Format: [a0, a1, b0, b1, a0*b0, a1*b0, a0*b1, a1*b1]")
        for j, point in enumerate(boundary):
            print(f"Point {j+1}: {point}")
        
        # 计算超平面参数
        A = np.hstack([boundary, np.ones((8, 1))])
        U, S, Vh = np.linalg.svd(A, full_matrices=True)
        w_b = Vh[-1, :]
        w = w_b[:8]
        b = w_b[8]
        print(f"Hyperplane: w = {w}, b = {b}")
    
    return unique_boundaries, episode_rewards

# 运行训练
if __name__ == "__main__":
    boundary_sets, rewards = train_dqn(episodes=1000, max_steps=30)
    
    # 绘制训练进度
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    # 计算移动平均
    window = 50
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.subplot(1, 2, 2)
    plt.plot(moving_avg)
    plt.title(f"Moving Average of Rewards (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    
    plt.tight_layout()
    plt.show()