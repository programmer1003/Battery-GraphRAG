import os
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd


# ==========================================================
# 🌟 核心创新模块：跨变量耦合补丁提取器 (原样保留)
# ==========================================================
class CrossVariatePatchExtractor(nn.Module):
    def __init__(self, seq_len=512, patch_len=64, variates=3, d_model=256):
        super().__init__()
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.variates = variates
        self.d_model = d_model

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.variate_attention = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True, dim_feedforward=512, dropout=0.1
        )
        self.attn_weights = None

    def forward(self, x, ablation_mode='full'):
        B, L, V = x.shape
        x = x.permute(0, 2, 1)  # [B, 3, L]

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x_emb = self.patch_embedding(x)

        x_emb = x_emb.permute(0, 2, 1, 3)
        x_emb = x_emb.reshape(B * self.patch_num, V, self.d_model)

        if ablation_mode == 'Ablation_wo_CrossAttn':
            coupled_features = x_emb
            self.attn_weights = None
        else:
            if not self.training:
                _, attn_weights = self.variate_attention.self_attn(x_emb, x_emb, x_emb, need_weights=True)
                self.attn_weights = attn_weights.detach()
            coupled_features = self.variate_attention(x_emb)

        coupled_features = coupled_features.reshape(B, self.patch_num, V, self.d_model)
        return coupled_features


# ==========================================================
# 👑 主打战车：TF-GDC 满血融合版 (原样保留)
# ==========================================================
class TF_GDC(nn.Module):
    def __init__(self, win_size=512, input_c=3, patch_size=64, d_model=256, ablation_mode='full'):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.win_size = win_size
        self.input_c = input_c
        self.patch_size = patch_size
        self.patch_num = win_size // patch_size
        self.d_model = d_model
        self.viz_cache = {}

        self.variate_extractor = CrossVariatePatchExtractor(
            seq_len=win_size, patch_len=patch_size, variates=input_c, d_model=d_model
        )
        self.fusion_linear = nn.Linear(input_c * d_model, d_model)

        encoder_layer_A = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dim_feedforward=512)
        self.branch_a_encoder = nn.TransformerEncoder(encoder_layer_A, num_layers=2)

        self.branch_b_encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )
        self.norm_b = nn.LayerNorm(d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        dict_path = 'dataset/Battery/semantic_anchors.npy'
        if os.path.exists(dict_path):
            anchors_data = np.load(dict_path)
            self.semantic_memory = nn.Parameter(torch.from_numpy(anchors_data).float(), requires_grad=False)
        else:
            self.semantic_memory = nn.Parameter(torch.randn(5, 768), requires_grad=False)

        self.text_projector = nn.Linear(768, d_model)
        self.dict_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.norm_fusion = nn.LayerNorm(d_model)

        self.projection_head = nn.Linear(d_model, d_model)
        self.reconstruction_head = nn.Linear(d_model, patch_size * input_c)

    def forward(self, x):
        B = x.shape[0]
        coupled_feat = self.variate_extractor(x, self.ablation_mode)
        flat_feat = coupled_feat.reshape(B, self.patch_num, self.input_c * self.d_model)
        fused_feat = self.fusion_linear(flat_feat)

        out_branch_a = self.branch_a_encoder(fused_feat)

        x_permuted = fused_feat.permute(0, 2, 1)
        out_branch_b = self.branch_b_encoder(x_permuted).permute(0, 2, 1)
        out_branch_b = self.norm_b(out_branch_b)

        gate_weight = self.gate(torch.cat([out_branch_a, out_branch_b], dim=-1))
        combined_query = gate_weight * out_branch_a + (1 - gate_weight) * out_branch_b

        memory_proj = self.text_projector(self.semantic_memory)
        key_value = memory_proj.unsqueeze(0).repeat(B, 1, 1)
        dict_out, _ = self.dict_attention(combined_query, key_value, key_value)
        final_feat = self.norm_fusion(combined_query + dict_out)

        out = self.reconstruction_head(final_feat)
        out = out.reshape(B, self.win_size, self.input_c)

        proj_a = self.projection_head(out_branch_a)
        proj_b = self.projection_head(out_branch_b)

        return out, proj_a, proj_b


# ==========================================================
# 🚀 工业级微服务化包装 (FastAPI 引擎)
# ==========================================================

app = FastAPI(title="TF-GDC 时序异常诊断引擎", version="1.0")

# 1. 初始化模型 (全局加载，避免每次请求重复初始化)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 正在使用计算设备: {device}")

# 💥 核心修复：将 patch_size 改为和你训练时一致的 16
model = TF_GDC(win_size=512, input_c=3, patch_size=16, d_model=256).to(device)

WEIGHTS_PATH = "best_model.pth"

if os.path.exists(WEIGHTS_PATH):
    # 💥 顺手加一个 weights_only=True，消除 PyTorch 的黄色安全警告
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    print(f"✅ 成功加载 TF-GDC 模型权重: {WEIGHTS_PATH}")
else:
    print(f"⚠️ 未找到权重文件 {WEIGHTS_PATH}，将使用随机初始化权重进行演示。")

model.eval()  # 务必切换到测试模式


# 2. 定义前端/Agent 传过来的数据格式
class PredictRequest(BaseModel):
    battery_id: str


# 3. 核心推理接口
@app.post("/predict")
async def run_inference(request: PredictRequest):
    print(f"\n[TF-GDC 引擎] 收到诊断请求，目标电池：{request.battery_id}")

    try:
        # ----------------------------------------------------
        # 步骤 A：真实工业数据管道 (Data Pipeline)
        # ----------------------------------------------------
        # 假设你的真实电池数据存放在当前目录的 dataset 文件夹下
        data_dir = "dataset"
        csv_path = os.path.join(data_dir, f"{request.battery_id}.csv")

        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"数据仓库中未找到电池 {request.battery_id} 的时序文件。")

        # 1. 使用 Pandas 读取 CSV (假设包含 Voltage, Current, Temperature 列)
        df = pd.read_csv(csv_path)

        # 2. 提取核心的 3 个变量列
        # 请根据你实际 CSV 的表头名字修改这里的列表！
        variates_cols = ['Voltage_V', 'Current_A', 'Temperature_C']
        if not all(col in df.columns for col in variates_cols):
            raise HTTPException(status_code=400, detail="CSV 缺少必要的电压、电流或温度列。")

        # 3. 截取最近的 512 个时间步
        seq_len = 512
        if len(df) < seq_len:
            raise HTTPException(status_code=400, detail=f"数据量不足 {seq_len} 行。")

        # 保留 DataFrame 格式用于画图，提取 values 用于矩阵运算
        recent_df = df[variates_cols].tail(seq_len)
        recent_data = recent_df.values

        # 4. Z-score 归一化 ... (这部分保留你原来的不变)
        mean = np.mean(recent_data, axis=0)
        std = np.std(recent_data, axis=0)
        recent_data_normalized = (recent_data - mean) / (std + 1e-8)

        # 5. 转换为 PyTorch Tensor 并送到计算设备 [Batch=1, Seq_Len=512, Variates=3]
        input_data = torch.tensor(recent_data_normalized, dtype=torch.float32).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # 步骤 B：执行模型推理
        # ----------------------------------------------------
        with torch.no_grad():
            out, proj_a, proj_b = model(input_data)

            # 步骤 C：异常诊断逻辑 (重构误差判定)
            mse_loss = nn.MSELoss()(out, input_data).item()

            ANOMALY_THRESHOLD = 0.5
            is_anomalous = mse_loss > ANOMALY_THRESHOLD
            confidence = min(0.99, float(1 - np.exp(-mse_loss / 0.1))) if is_anomalous else float(1 - mse_loss)

        # 💥 核心修改：在步骤 D 的返回结果里，偷偷加上画图数据
        result = {
            "status": "success",
            "anomaly_detected": is_anomalous,
            "predicted_fault": "微短路故障" if is_anomalous else "正常状态",
            "confidence": round(confidence, 4),
            "mse_score": round(mse_loss, 4),
            "detail": f"TF-GDC 读取了最新的 {seq_len} 条物理数据，计算出底层重构误差为 {mse_loss:.4f}。",
            "plot_data": {  # 👈 新增：给前端画图准备的原始波形
                "Voltage_V": recent_df["Voltage_V"].tolist(),
                "Current_A": recent_df["Current_A"].tolist(),
                "Temperature_C": recent_df["Temperature_C"].tolist()
            }
        }
        return result
    except Exception as e:
        print(f"❌ 推理引擎报错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 启动独立微服务，暴露于 8001 端口
    print("🚀 启动 TF-GDC 算法微服务 (端口: 8001)...")
    uvicorn.run(app, host="0.0.0.0", port=8001)