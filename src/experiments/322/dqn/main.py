# 在 train_dqn 外部定义辅助函数
def evaluate_action_outcome(env, state_snapshot, action):
    """
    作弊函数：尝试执行一个动作，看看结果如何，然后回滚。
    返回: (reward, next_n_outliers)
    """
    # 保存环境快照（深拷贝）
    saved_state = env.state.copy()
    saved_normal = env.current_normal.copy()
    saved_b = env.current_b
    saved_prev_dist = env.prev_dist

    # 执行
    try:
        # 这里假设 env.step 已经修改为返回 info
        _, reward, _, info = env.step(action)
        n_outliers = info['n_outliers']
    except Exception:
        reward = -10.0
        n_outliers = 999

    # 回滚环境
    env.state = saved_state
    env.current_normal = saved_normal
    env.current_b = saved_b
    env.prev_dist = saved_prev_dist

    return reward, n_outliers


def train_dqn(episodes=2000, max_steps_per_episode=100):
    # ... (初始化代码不变) ...

    print("Start Training with GREEDY TEACHER...")

    for e in range(episodes):
        state = env.reset()

        # 获取初始的异常点数量
        # 注意：reset现在只返回obs，你需要手动调一次计算或者让reset返回info
        # 这里为了简单，我们假设 reset 后通过一次假 step 获取，或者修改 reset
        # 简单的做法：手动算一次
        valid, norm, b = check_points_form_hyperplane(env.state)
        # 简单计算初始 outliers
        vals = np.dot(env.all_points, norm) + b
        current_outliers = min(np.sum(vals > 1e-5), np.sum(vals < -1e-5))

        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps_per_episode:

            valid_mask = env.get_valid_actions_mask()
            valid_indices = np.where(valid_mask)[0]

            # === 核心修改：贪婪导师介入 ===
            best_teacher_action = -1
            min_next_outliers = current_outliers

            # 只有在随机探索阶段，或者为了收集优质样本时，开启导师模式
            # 如果目前异常点还很多（比如 > 5），说明离终点还远，让导师带路
            use_teacher = (np.random.rand() < 0.5) or (current_outliers > 0 and e < 500)

            if use_teacher:
                # 遍历所有合法动作（100个并不多，完全算得过来）
                # 寻找能让 outliers 减少的动作
                candidate_actions = []

                for act_idx in valid_indices:
                    # 模拟执行
                    r, next_outliers = evaluate_action_outcome(env, None, act_idx)

                    if next_outliers < current_outliers:
                        # 发现了一个能减少错误的动作！
                        candidate_actions.append((act_idx, next_outliers))

                if candidate_actions:
                    # 如果有多个好动作，选减少最多的，或者从中随机选一个
                    candidate_actions.sort(key=lambda x: x[1])
                    best_teacher_action = candidate_actions[0][0]
                    # print(f"  Teacher found shortcut: {current_outliers} -> {candidate_actions[0][1]}")

            # === 动作选择逻辑 ===
            if best_teacher_action != -1:
                # 1. 听老师的 (Teacher Forcing)
                action = best_teacher_action
                # 既然这是个极其正确的动作，我们可以不用 epsilon 随机了
            else:
                # 2. 老师也没招了（陷入局部最优），或者没触发老师
                # 这时候依靠 DQN 的 Q 值来跳出坑
                if np.random.rand() <= agent.epsilon:
                    action = np.random.choice(valid_indices)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = agent.model(state_tensor)
                        q_values_np = q_values.cpu().numpy()[0]
                    action = masked_argmax(q_values_np, valid_mask)

            # === 执行 ===
            # 注意：step 返回值改为 4 个
            next_state, reward, done, info = env.step(action)
            next_outliers = info['n_outliers']

            # === 奖励修正 (Reward Shaping) ===
            # 如果动作真的减少了 outliers，给予巨大奖励！
            if next_outliers < current_outliers:
                reward += 10.0 * (current_outliers - next_outliers)
                # print(f"Progress! {current_outliers} -> {next_outliers}")

            # 更新当前状态的 outliers
            current_outliers = next_outliers

            # === 记忆与训练 ===
            agent.remember(state, action, reward, next_state, done)

            # 如果这是一个成功的步骤（outliers 减少了），多存几次！强化记忆！
            if reward > 5.0:
                for _ in range(3):
                    agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > agent.batch_size:
                agent.replay()

            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                # Found boundary logic...
                pass
