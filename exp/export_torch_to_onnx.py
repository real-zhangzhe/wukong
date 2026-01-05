import torch

from model.pytorch.wukong import Wukong

BATCH_SIZE = 1
NUM_CAT_FEATURES = 26
NUM_DENSE_FEATURES = 13
# Criteo dataset specific, from dataset.npz
NUM_SPARSE_EMBS = [10] * NUM_CAT_FEATURES
DIM_EMB = 128
assert NUM_CAT_FEATURES == len(NUM_SPARSE_EMBS)


model = Wukong(
    num_layers=2,
    num_sparse_embs=NUM_SPARSE_EMBS,
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
dummy_sparse_input = torch.stack(
    [
        torch.randint(0, NUM_SPARSE_EMBS[i], (BATCH_SIZE,), dtype=torch.long)
        for i in range(NUM_CAT_FEATURES)
    ],
    dim=1,
)
dummy_dense_input = torch.randn(BATCH_SIZE, NUM_DENSE_FEATURES)
output = model(dummy_sparse_input, dummy_dense_input)
print("Output shape:", output.shape)
torch.onnx.export(
    model,
    (dummy_sparse_input, dummy_dense_input),
    "wukong_model.onnx",
    input_names=["sparse_input", "dense_input"],
    output_names=["output"],
    opset_version=17,
)
