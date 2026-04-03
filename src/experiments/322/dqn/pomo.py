import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from points import generate_points
from facet import check_points_form_hyperplane

# ---------- 参数设置 ----------
BATCH_SIZE = 10           # 每个 batch 的实例数（默认使用同一组点）
_POINTS_SAMPLE = generate_points()
N_NODES = _POINTS_SAMPLE.shape[0]   # 顶点数（如 64）
NODE_DIM = _POINTS_SAMPLE.shape[1]  # 顶点特征维度（如 26）
HIDDEN_DIM = 128         # 模型内部维度
N_HEADS = 8              # 多头注意力头数
N_ENCODER_LAYERS = 6     # 编码器层数
N_EPOCHS = 150           # 训练轮数
LR = 5e-5                # 学习率
GRAD_CLIP = 1.0          # 梯度裁剪
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# ---------- 从 points.py 生成模型输入 ----------
def generate_data(batch_size=BATCH_SIZE):
    """使用 points.py 中的 generate_points 输出作为模型输入。

    返回:
        x: (batch_size, n_nodes, node_dim)
    """
    base_points = generate_points().astype(np.float32)  # (n_nodes, node_dim)
    n_nodes, node_dim = base_points.shape

    if batch_size <= 1:
        return torch.from_numpy(base_points).to(DEVICE).unsqueeze(0)  # (1, n_nodes, node_dim)

    instances = [base_points]
    for b in range(1, batch_size):
        rng = np.random.default_rng(seed=b)
        perm = rng.permutation(node_dim)
        instances.append(base_points[:, perm].copy())

    batch_points = np.stack(instances, axis=0)  # (batch_size, n_nodes, node_dim)
    x = torch.from_numpy(batch_points).to(DEVICE)
    return x

# ---------- 成本函数：参考环境奖励（cost = -reward） ----------
def compute_cost(x, tour):
    """
    x: (batch, n_nodes, node_dim)
    tour: (batch, n_nodes) 顶点索引序列（0 到 n_nodes-1）
    返回: (batch,) 成本，定义为 -reward（便于最小化）
    """
    batch_size, n_nodes, node_dim = x.size()
    subset_size = min(node_dim, n_nodes)

    idx = tour[:, :subset_size].unsqueeze(-1).expand(-1, -1, node_dim)
    selected_points = torch.gather(x, 1, idx)  # (batch, subset_size, node_dim)

    # 奖励函数使用离散几何判定（不可导），在 REINFORCE 中作为标量回报是可行的。
    x_np = x.detach().cpu().numpy()
    selected_np = selected_points.detach().cpu().numpy()
    costs = []

    for b in range(batch_size):
        all_points = x_np[b]
        points = selected_np[b]

        is_hyperplane, normal, bias, rank = check_points_form_hyperplane(points)

        r1 = (rank ** 2) / node_dim - 5*(node_dim - rank)  # 基于秩的奖励，秩越高奖励越大
        positive_count = 0
        negative_count = 0

        if is_hyperplane:
            r1 += 50.0
            selected_set = {tuple(p.tolist()) for p in points}

            for p in all_points:
                if tuple(p.tolist()) in selected_set:
                    continue
                value = float(np.dot(normal, p) + bias)
                if value > 1e-6:
                    positive_count += 1
                elif value < -1e-6:
                    negative_count += 1

            if positive_count == 0 or negative_count == 0:
                r2 = 100.0
            else:
                r2 = -min(positive_count, negative_count) / len(all_points) * 3.0

            r3 = 0.0
        else:
            r1 -= 10.0
            r2 = 0.0
            r3 = 0.0

        reward = r1 + r2 + r3
        costs.append(-reward)

    return torch.tensor(costs, device=x.device, dtype=x.dtype)

# ---------- 模型定义 ----------
class AttentionModel(nn.Module):
    """
    基于 Attention Model 的编码器-解码器，支持多起点并行解码
    """
    def __init__(self, node_dim=NODE_DIM, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, n_layers=N_ENCODER_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.node_dim = node_dim

        # 输入投影
        self.init_embed = nn.Linear(node_dim, hidden_dim)

        # Transformer 编码器（无位置编码，保持置换不变性）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 解码器：计算查询向量与节点嵌入的注意力
        # 查询向量由三个部分组成：图嵌入、起始节点嵌入、当前节点嵌入
        # 因此需要一个线性层将拼接后的向量映射到 hidden_dim
        self.query_proj = nn.Linear(3 * hidden_dim, hidden_dim)

        # 计算兼容性（scaled dot-product）的权重
        # 这里用单头注意力，但可以扩展为多头
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 未使用，可忽略
        self.scale = hidden_dim ** -0.5

        # 最终输出 logits 的投影
        self.out_proj = nn.Linear(hidden_dim, 1, bias=False)

    def encode(self, x):
        """编码器：输入 (batch, n_nodes, node_dim) -> 节点嵌入 (batch, n_nodes, hidden_dim)"""
        x = self.init_embed(x)                     # (batch, n_nodes, hidden_dim)
        x = self.encoder(x)                        # (batch, n_nodes, hidden_dim)
        return x

    def decode_step(self, node_embeds, graph_embed, start_embed, current_embed, mask):
        """
        单步解码，输入:
            node_embeds: (batch, n_nodes, hidden_dim)
            graph_embed: (batch, 1, hidden_dim) 全局图嵌入
            start_embed: (batch, 1, hidden_dim) 起始节点嵌入
            current_embed: (batch, 1, hidden_dim) 当前节点嵌入
            mask: (batch, n_nodes) bool，True表示已访问
        返回:
            logits: (batch, n_nodes)
        """
        batch, n_nodes, _ = node_embeds.size()
        # 构造查询向量
        query = torch.cat([graph_embed, start_embed, current_embed], dim=-1)  # (batch, 1, 3*hidden_dim)
        query = self.query_proj(query)  # (batch, 1, hidden_dim)
        # 计算注意力分数
        q = self.W_q(query)        # (batch, 1, hidden_dim)
        k = self.W_k(node_embeds)  # (batch, n_nodes, hidden_dim)
        # 兼容性分数
        compat = (q @ k.transpose(-2, -1)) * self.scale  # (batch, 1, n_nodes)
        # 掩码：将已访问节点的分数设为 -inf
        compat = compat.masked_fill(mask.unsqueeze(1), float('-inf'))
        # 可选：使用 tanh 裁剪（原论文中使用了 clipping，此处简单起见直接用 softmax）
        logits = compat.squeeze(1)  # (batch, n_nodes)
        return logits

    def greedy_rollout_all_starts(self, x):
        """
        对所有起始点进行贪婪解码，返回每个起始点的成本 (batch, n_nodes)
        """
        batch, n_nodes, _ = x.size()
        node_embeds = self.encode(x)                 # (batch, n_nodes, hidden_dim)
        graph_embed = node_embeds.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)

        # 存储所有起始点的成本
        costs = torch.zeros(batch, n_nodes, device=x.device)

        # 对所有起始点分别进行贪婪解码
        for start in range(n_nodes):
            # 初始掩码：全部未访问
            mask = torch.zeros(batch, n_nodes, dtype=torch.bool, device=x.device)
            # 第一步：选择起始点
            current = torch.full((batch,), start, dtype=torch.long, device=x.device)
            start_embed = node_embeds[torch.arange(batch), start].unsqueeze(1)   # (batch,1,hidden_dim)
            current_embed = start_embed.clone()
            # 标记起始点已访问
            mask = mask.scatter(1, current.unsqueeze(1), True)
            tour = [current.clone()]

            for step in range(1, self.node_dim):
                logits = self.decode_step(node_embeds, graph_embed, start_embed, current_embed, mask)
                # 贪婪选择
                current = logits.argmax(dim=-1)  # (batch,)
                tour.append(current.clone())
                # 更新掩码
                mask = mask.scatter(1, current.unsqueeze(1), True)
                # 更新当前节点嵌入
                current_embed = node_embeds[torch.arange(batch), current].unsqueeze(1)

            # 拼接序列
            tour = torch.stack(tour, dim=1)  # (batch, n_nodes)
            cost = compute_cost(x, tour)
            costs[:, start] = cost

        return costs

    def sample_all_starts(self, x, temperature=1.0):
        """
        对所有起始点进行采样（每个起点采样一次），返回采样序列及其对数概率和成本
        """
        batch, n_nodes, _ = x.size()
        node_embeds = self.encode(x)
        graph_embed = node_embeds.mean(dim=1, keepdim=True)

        # 存储结果
        sampled_tours = torch.zeros(batch, n_nodes, self.node_dim, dtype=torch.long, device=x.device)  # (batch, start, step)
        log_probs = torch.zeros(batch, n_nodes, device=x.device)
        costs = torch.zeros(batch, n_nodes, device=x.device)

        for start in range(n_nodes):
            mask = torch.zeros(batch, n_nodes, dtype=torch.bool, device=x.device)
            current = torch.full((batch,), start, dtype=torch.long, device=x.device)
            start_embed = node_embeds[torch.arange(batch), start].unsqueeze(1)
            current_embed = start_embed.clone()
            mask = mask.scatter(1, current.unsqueeze(1), True)
            tour = [current.clone()]
            log_p = torch.zeros(batch, device=x.device)

            for step in range(1, self.node_dim):
                logits = self.decode_step(node_embeds, graph_embed, start_embed, current_embed, mask)
                # 采样（带温度）
                probs = F.softmax(logits / temperature, dim=-1)
                dist = torch.distributions.Categorical(probs)
                current = dist.sample()  # (batch,)
                log_p = log_p + dist.log_prob(current)
                tour.append(current.clone())
                mask = mask.scatter(1, current.unsqueeze(1), True)
                current_embed = node_embeds[torch.arange(batch), current].unsqueeze(1)

            tour = torch.stack(tour, dim=1)  # (batch, node_dim)
            cost = compute_cost(x, tour)
            sampled_tours[:, start] = tour
            log_probs[:, start] = log_p
            costs[:, start] = cost

        return sampled_tours, log_probs, costs

# ---------- 训练函数 ----------
def train():
    model = AttentionModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    for epoch in range(N_EPOCHS):
        model.train()
        # 生成训练数据（来自 points.py）
        x = generate_data(BATCH_SIZE)

        # ----- 1. 计算多起点贪婪基线 -----
        with torch.no_grad():
            baseline_costs = model.greedy_rollout_all_starts(x)  # (batch, n_nodes)
            baseline = baseline_costs.mean(dim=1, keepdim=True)   # (batch, 1)

        # ----- 2. 采样解（每个起点采样一次）-----
        _, log_probs, sampled_costs = model.sample_all_starts(x)

        # ----- 3. 计算优势 -----
        advantage = baseline - sampled_costs  # (batch, n_nodes)

        # ----- 4. 损失 -----
        loss = -(advantage * log_probs).mean()

        # ----- 5. 反向传播 -----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        # ----- 日志 -----
        if epoch % 10 == 0:
            avg_cost = sampled_costs.mean().item()
            avg_baseline = baseline.mean().item()
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, Avg Cost: {avg_cost:.4f}, Baseline: {avg_baseline:.4f}")

# ---------- 推理 ----------
def inference(model, x, greedy=True, node_dim=NODE_DIM):
    """
    x: (batch, n_nodes, node_dim)
    返回: (batch, node_dim) 序列，每个实例的最佳序列（在所有起点中选择成本最低的）
    """
    model.eval()
    with torch.no_grad():
        if greedy:
            # 贪婪解码所有起点
            costs = model.greedy_rollout_all_starts(x)  # (batch, n_nodes)
            best_start = costs.argmin(dim=1)  # (batch,)
            # 对于每个实例，提取最佳起点对应的贪婪序列
            # 但 greedy_rollout_all_starts 内部没有存储序列，所以需要重新计算一次最佳起点的贪婪序列
            # 简单起见，我们重新运行一次单起点的贪婪解码
            batch, n_nodes = x.size(0), x.size(1)
            node_embeds = model.encode(x)
            graph_embed = node_embeds.mean(dim=1, keepdim=True)
            tours = []
            for i in range(batch):
                start = best_start[i].item()
                mask = torch.zeros(n_nodes, dtype=torch.bool, device=x.device)
                current = start
                start_embed = node_embeds[i, start].unsqueeze(0).unsqueeze(0)
                current_embed = start_embed.clone()
                mask[current] = True
                tour = [current]
                for _ in range(1, node_dim):
                    logits = model.decode_step(
                        node_embeds[i:i+1], graph_embed[i:i+1],
                        start_embed, current_embed,
                        mask.unsqueeze(0)
                    )
                    current = logits.argmax(dim=-1).item()
                    tour.append(current)
                    mask[current] = True
                    current_embed = node_embeds[i, current].unsqueeze(0).unsqueeze(0)
                tours.append(tour)
            tours = torch.tensor(tours, device=x.device)
            return tours
        else:
            # 采样多个起点，选最好
            _, _, sampled_costs = model.sample_all_starts(x)
            best_start = sampled_costs.argmin(dim=1)
            # 同样需要提取对应序列，这里略复杂，可类似上面实现
            # 为简洁，此处返回贪婪结果
            return inference(model, x, greedy=True)

# ---------- 主程序 ----------
if __name__ == "__main__":
    print("Training POMO model...")
    train()

    # 测试推理
    model = AttentionModel().to(DEVICE)
    # 加载预训练模型（如果有）
    # model.load_state_dict(torch.load("pomo_model.pth"))

    test_x = generate_data(1)
    tour = inference(model, test_x)
    print("Inferred tour (shape):", tour.shape)
    print("Tour:", tour[0].tolist())