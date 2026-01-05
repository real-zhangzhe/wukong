import torch

from model.pytorch.wukong import Wukong

BATCH_SIZE = 1
NUM_CAT_FEATURES = 32
NUM_DENSE_FEATURES = 16
NUM_EMBEDDING = 100
DIM_EMB = 128

model = Wukong(
    num_layers=2,
    num_sparse_emb=NUM_EMBEDDING,
    dim_emb=DIM_EMB,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    num_emb_lcb=16,
    num_emb_fmb=16,
    rank_fmb=8,
    num_hidden_wukong=2,
    dim_hidden_wukong=16,
    num_hidden_head=2,
    dim_hidden_head=32,
    dim_output=1,
    dropout=0.0,
)
model.eval()
dummy_sparse_input = torch.randint(
    0, NUM_EMBEDDING, (BATCH_SIZE, NUM_CAT_FEATURES), dtype=torch.long
)
dummy_dense_input = torch.randn(BATCH_SIZE, NUM_DENSE_FEATURES)
torch.onnx.export(
    model,
    (dummy_sparse_input, dummy_dense_input),
    "wukong_model.onnx",
    input_names=["sparse_input", "dense_input"],
    output_names=["output"],
    opset_version=17,
)
