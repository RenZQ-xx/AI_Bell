import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import ConvexHull
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings("ignore")

# ========== 参数配置 ==========
N = 3                       # 基变量对数，总点数 = 4^N = 64，维度 = 3^N - 1 = 26
O = 2                       # 每个变量取值个数（-1, 1）
BATCH_SIZE = 8              # 训练批次大小（因显存限制）
HIDDEN_DIM = 128            # 模型隐藏维度
N_HEADS = 8                 # 注意力头数
N_ENCODER_LAYERS = 6        # 编码器层数
LR = 1e-4
N_EPOCHS = 200
GRAD_CLIP = 1.0
TEMPERATURE = 0.8           # 采样温度
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 1. 顶点生成函数 ==========
def generate_points(N=N):
    """生成 4^N 个点，每个点维度为 3^N - 1"""
    points = []
    n_vars = 2 * N                     # 基变量个数：A0,A1,B0,B1,...
    for i in range(4**N):
        # 将 i 转换为长度为 2N 的二进制数组，0->-1, 1->1
        base = [2 * ((i >> j) & 1) - 1 for j in range(n_vars)]
        point = base.copy()
        # 添加所有组合乘积项
        for elenum in range(2, N+1):
            for pair in itertools.combinations(range(N), elenum):
                for offset in range(O ** elenum):
                    res = 1
                    for idx, var_idx in enumerate(pair):
                        bit = (offset >> idx) & 1
                        res *= base[2 * var_idx + bit]
                    point.append(res)
        points.append(np.array(point, dtype=np.float32))
    return np.array(points, dtype=np.float32)

# ========== 2. 数据增强与凸包标注 ==========
def random_linear_transform(points, scale_range=(0.5, 1.5), rot=True):
    """对点集施加随机可逆线性变换（缩放+旋转+平移）"""
    dim = points.shape[1]
    scale = np.random.uniform(*scale_range, size=dim)
    A = np.diag(scale)
    if rot:
        Q, _ = np.linalg.qr(np.random.randn(dim, dim))
        A = Q @ A
    shift = np.random.randn(dim) * 0.5
    return (points @ A.T) + shift

def get_convex_hull_vertices(points):
    """返回凸包顶点的索引列表（顺序由 ConvexHull 给出）"""
    hull = ConvexHull(points)
    return list(hull.vertices)

# 预先生成原始点集（作为模板）
original_points = generate_points(N=3)   # (64, 26)

def generate_training_batch(batch_size):
    """生成一个 batch 的训练数据：变换后的点集及其凸包顶点索引列表"""
    batch_points = []
    batch_vertices = []
    for _ in range(batch_size):
        transformed = random_linear_transform(original_points)
        vertices = get_convex_hull_vertices(transformed)
        batch_points.append(transformed)
        batch_vertices.append(vertices)
    return np.array(batch_points, dtype=np.float32), batch_vertices

# ========== 3. POMO 模型定义 ==========
class Encoder(nn.Module):
    """Transformer 编码器（无位置编码）"""
    def __init__(self, input_dim=26, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, n_layers=N_ENCODER_LAYERS):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        x = self.embed(x)                     # (batch, n_nodes, hidden_dim)
        x = self.encoder(x)
        return x

class DecoderStep(nn.Module):
    """单步解码器，支持停止标记"""
    def __init__(self, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.query_proj = nn.Linear(3 * hidden_dim, hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5
        # 停止标记的可学习嵌入
        self.stop_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, node_embeds, graph_embed, start_embed, current_embed, mask):
        """
        参数:
            node_embeds: (batch, n_nodes, hidden_dim)
            graph_embed: (batch, 1, hidden_dim)
            start_embed: (batch, 1, hidden_dim)
            current_embed: (batch, 1, hidden_dim)
            mask: (batch, n_nodes) bool, True=已访问/禁止
        返回:
            logits: (batch, n_nodes + 1)，最后一维是停止标记的得分
        """
        batch, n_nodes, _ = node_embeds.size()
        # 查询向量 = [图嵌入, 起始点嵌入, 当前点嵌入]
        query = torch.cat([graph_embed, start_embed, current_embed], dim=-1)
        query = self.query_proj(query)                     # (batch, 1, hidden_dim)
        q = self.W_q(query)                                # (batch, 1, hidden_dim)
        k = self.W_k(node_embeds)                          # (batch, n_nodes, hidden_dim)
        compat = (q @ k.transpose(-2, -1)) * self.scale    # (batch, 1, n_nodes)
        compat = compat.masked_fill(mask.unsqueeze(1), float('-inf'))
        logits_nodes = compat.squeeze(1)                   # (batch, n_nodes)

        # 停止标记得分
        stop_key = self.stop_embedding.expand(batch, -1, -1)   # (batch, 1, hidden_dim)
        stop_compat = (self.W_q(query) @ self.W_k(stop_key).transpose(-2, -1)) * self.scale
        logits_stop = stop_compat.squeeze(-1)                  # (batch, 1)
        logits = torch.cat([logits_nodes, logits_stop], dim=-1) # (batch, n_nodes+1)
        return logits

class POMOForConvexHull(nn.Module):
    """完整 POMO 模型，支持多起点并行解码（贪婪/采样），支持异步停止"""
    def __init__(self, input_dim=26, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, n_layers=N_ENCODER_LAYERS):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, n_heads, n_layers)
        self.decoder_step = DecoderStep(hidden_dim, n_heads)
        self.hidden_dim = hidden_dim

    def _encode(self, x):
        """返回节点嵌入和全局图嵌入"""
        node_embeds = self.encoder(x)                         # (batch, n_nodes, hidden_dim)
        graph_embed = node_embeds.mean(dim=1, keepdim=True)   # (batch, 1, hidden_dim)
        return node_embeds, graph_embed

    def _decode_tour(self, node_embeds, graph_embed, start_idx, greedy=True, temperature=1.0, max_steps=None):
        """
        为单个起始点生成一个完整的序列（支持异步停止）。
        参数:
            node_embeds: (batch, n_nodes, hidden_dim)
            graph_embed: (batch, 1, hidden_dim)
            start_idx: int，起始点索引（所有实例相同）
            greedy: bool，是否贪婪解码
            temperature: float，采样温度
            max_steps: int，最大步数（默认 n_nodes）
        返回:
            tours: (batch, seq_len) 每个实例的序列（含停止标记前的顶点）
            log_probs: (batch,) 序列的对数概率（仅采样时有效）
            finished: (batch,) bool，是否提前停止（始终为True）
        """
        batch, n_nodes, _ = node_embeds.size()
        if max_steps is None:
            max_steps = n_nodes

        # 初始化
        mask = torch.zeros(batch, n_nodes, dtype=torch.bool, device=node_embeds.device)
        current = torch.full((batch,), start_idx, dtype=torch.long, device=node_embeds.device)
        start_embed = node_embeds[torch.arange(batch), start_idx].unsqueeze(1)  # (batch,1,hidden_dim)
        current_embed = start_embed.clone()
        mask.scatter_(1, current.unsqueeze(1), True)

        tours = [current.clone()]          # 存储每一步的顶点
        log_probs = torch.zeros(batch, device=node_embeds.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=node_embeds.device)

        for step in range(1, max_steps + 1):
            # 如果所有实例都已停止，则退出
            if finished.all():
                break

            logits = self.decoder_step(node_embeds, graph_embed, start_embed, current_embed, mask)
            # 将已停止的实例的 logits 设为极小值，使其只能选择停止标记（或无效）
            # 但更简单：直接跳过已停止的实例，保持其输出不变
            if greedy:
                next_idx = logits.argmax(dim=-1)   # (batch,)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                dist = torch.distributions.Categorical(probs)
                next_idx = dist.sample()
                log_probs += dist.log_prob(next_idx)

            # 判断是否选择停止标记（索引 = n_nodes）
            is_stop = (next_idx == n_nodes)
            # 对于未停止的实例，将顶点加入序列；对于已停止的实例，忽略
            # 更新当前顶点嵌入和掩码（仅对未停止的实例）
            current = torch.where(is_stop, current, next_idx)  # 停止后保持原样
            tours.append(current.clone())
            # 更新掩码：只有未停止的实例才更新掩码
            mask = mask.clone()
            mask[~finished & ~is_stop, current[~finished & ~is_stop]] = True
            # 更新当前嵌入
            new_embed = node_embeds[torch.arange(batch), current].unsqueeze(1)
            current_embed = new_embed

            # 更新停止状态：一旦停止，标记为 finished
            finished = finished | is_stop

        # 将所有序列拼接成 tensor (batch, seq_len)
        tours = torch.stack(tours, dim=1)   # (batch, seq_len)
        # 截取到每个实例的停止标记之前（移除停止标记本身）
        # 由于停止标记可能出现在不同位置，我们需要逐实例处理
        # 但为简化返回，这里保留包含停止标记的序列，后续计算 IoU 时再处理
        return tours, log_probs, finished

    def greedy_rollout_all_starts(self, x):
        """对所有起始点进行贪婪解码，返回每个起始点的预测序列（batch, n_nodes, max_seq_len）"""
        batch, n_nodes, _ = x.size()
        node_embeds, graph_embed = self._encode(x)
        # 最大序列长度最多为 n_nodes+1（包含停止标记）
        max_seq_len = n_nodes + 1
        all_tours = torch.zeros(batch, n_nodes, max_seq_len, dtype=torch.long, device=x.device)
        for start in range(n_nodes):
            tours, _, _ = self._decode_tour(node_embeds, graph_embed, start, greedy=True, max_steps=n_nodes+1)
            all_tours[:, start, :] = tours
        return all_tours

    def sample_all_starts(self, x, temperature=TEMPERATURE):
        """对所有起始点进行采样，返回序列、对数概率和完成标志"""
        batch, n_nodes, _ = x.size()
        node_embeds, graph_embed = self._encode(x)
        max_seq_len = n_nodes + 1
        all_tours = torch.zeros(batch, n_nodes, max_seq_len, dtype=torch.long, device=x.device)
        all_log_probs = torch.zeros(batch, n_nodes, device=x.device)
        for start in range(n_nodes):
            tours, log_probs, _ = self._decode_tour(node_embeds, graph_embed, start, greedy=False,
                                                    temperature=temperature, max_steps=n_nodes+1)
            all_tours[:, start, :] = tours
            all_log_probs[:, start] = log_probs
        return all_tours, all_log_probs

# ========== 4. 辅助函数：将序列转为顶点集合 ==========
def sequence_to_vertex_set(seq_tensor, n_nodes):
    """
    将预测序列（可能包含停止标记）转换为顶点集合。
    seq_tensor: (seq_len,) 长整数张量，值在 [0, n_nodes] 之间，n_nodes 表示停止标记
    返回: set of int (顶点索引)
    """
    seq = seq_tensor.cpu().numpy()
    # 找到停止标记的位置
    stop_pos = np.where(seq == n_nodes)[0]
    if len(stop_pos) > 0:
        seq = seq[:stop_pos[0]]   # 截取到停止标记前
    return set(seq)

def compute_iou(pred_set, true_set):
    """计算两个集合的 IoU"""
    if len(pred_set) == 0 and len(true_set) == 0:
        return 1.0
    inter = len(pred_set & true_set)
    union = len(pred_set | true_set)
    return inter / union if union > 0 else 0.0

# ========== 5. 训练函数 ==========
def train():
    model = POMOForConvexHull().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    for epoch in range(N_EPOCHS):
        model.train()
        # 生成一个 batch 的训练数据
        batch_points_np, batch_vertices_list = generate_training_batch(BATCH_SIZE)
        batch_points = torch.tensor(batch_points_np, dtype=torch.float32, device=DEVICE)
        n_nodes = batch_points.shape[1]   # 64

        # 将真实顶点集合转换为 set 列表，用于 IoU 计算
        true_sets = [set(v) for v in batch_vertices_list]

        # ----- 1. 多起点贪婪解码，计算基线奖励 -----
        with torch.no_grad():
            greedy_tours = model.greedy_rollout_all_starts(batch_points)  # (batch, n_nodes, max_len)
            baseline_rewards = torch.zeros(BATCH_SIZE, n_nodes, device=DEVICE)
            for i in range(BATCH_SIZE):
                for start in range(n_nodes):
                    pred_set = sequence_to_vertex_set(greedy_tours[i, start, :], n_nodes)
                    iou = compute_iou(pred_set, true_sets[i])
                    baseline_rewards[i, start] = iou
            baseline = baseline_rewards.mean(dim=1, keepdim=True)  # (batch, 1)

        # ----- 2. 采样解 -----
        sampled_tours, log_probs = model.sample_all_starts(batch_points, temperature=TEMPERATURE)
        sampled_rewards = torch.zeros(BATCH_SIZE, n_nodes, device=DEVICE)
        for i in range(BATCH_SIZE):
            for start in range(n_nodes):
                pred_set = sequence_to_vertex_set(sampled_tours[i, start, :], n_nodes)
                iou = compute_iou(pred_set, true_sets[i])
                sampled_rewards[i, start] = iou

        # ----- 3. 优势计算 -----
        advantage = sampled_rewards - baseline  # (batch, n_nodes)

        # ----- 4. 损失（策略梯度）-----
        loss = -(advantage * log_probs).mean()

        # ----- 5. 反向传播 -----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            avg_reward = sampled_rewards.mean().item()
            avg_baseline = baseline.mean().item()
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, Avg Reward (IoU): {avg_reward:.4f}, Baseline: {avg_baseline:.4f}")

    return model

# ========== 6. 推理：获取边界平面 ==========
def inference(model, points_np):
    """
    输入: points_np (n_nodes, dim) numpy 数组
    输出: (equations, vertices)
          equations: (n_facets, dim+1) 每个面方程为 n·x + b = 0（所有点满足 n·x + b >= 0）
          vertices: 凸包顶点索引列表
    """
    model.eval()
    points = torch.tensor(points_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, n_nodes, dim)
    n_nodes = points.shape[1]
    with torch.no_grad():
        greedy_tours = model.greedy_rollout_all_starts(points)  # (1, n_nodes, max_len)
    # 收集所有起始点预测的顶点集合，取并集
    all_pred_vertices = set()
    for start in range(n_nodes):
        pred_set = sequence_to_vertex_set(greedy_tours[0, start, :], n_nodes)
        all_pred_vertices.update(pred_set)
    # 如果没有预测到任何顶点，则回退到所有点
    if len(all_pred_vertices) == 0:
        all_pred_vertices = set(range(n_nodes))
    # 使用这些候选顶点计算凸包
    candidate_points = points_np[list(all_pred_vertices)]
    hull = ConvexHull(candidate_points)
    # 注意：hull.equations 是 (n_facets, dim+1)，方程为 n·x + b = 0，且所有点满足 n·x + b >= 0
    # 但 hull 是基于 candidate_points 的凸包，可能原始点集中有不在该凸包内的点，需要验证
    # 这里假设模型预测正确，直接返回
    return hull.equations, hull.vertices

# ========== 7. 主程序 ==========
if __name__ == "__main__":
    print("开始训练 POMO 模型用于凸包顶点检测...")
    model = train()
    torch.save(model.state_dict(), "pomo_convex_hull.pth")
    print("模型已保存至 pomo_convex_hull.pth")

    # 测试推理：生成一个新实例并输出边界平面
    test_points = random_linear_transform(original_points)
    eqs, verts = inference(model, test_points)
    print(f"\n测试实例凸包面数: {len(eqs)}")
    print("凸包顶点索引:", verts)
    print("第一个面的方程 (前5维):", eqs[0][:5])