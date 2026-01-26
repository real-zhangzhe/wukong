import torch
from torch import Tensor, nn
from typing import List


class SparseEmbedding(nn.Module):
    def __init__(self, num_sparse_embs: List[int], dim_emb: int) -> None:
        super().__init__()

        self.embeddings = nn.ModuleList()
        for num_emb in num_sparse_embs:
            self.embeddings.append(nn.Embedding(num_emb, dim_emb))

        self.fixed_vector = nn.Parameter(torch.ones(dim_emb), requires_grad=False)

    def forward(self, sparse_inputs: Tensor) -> Tensor:
        sparse_outputs = []
        for i, embedding in enumerate(self.embeddings):
            col_input = sparse_inputs[:, i]

            # [修改 3: 基础 Embedding 查找，对应 tf.abs]
            lookup_emb = embedding(torch.abs(col_input))

            # --- Mask 定义 ---
            # 对应 TF 中复杂的逻辑组合
            mask = ((col_input == 0) | (col_input == -2)) & (
                (col_input > 100000) | (torch.sum(col_input, dim=0, keepdim=True) == 0)
            )

            # --- 强制调用 SplitV 逻辑 ---
            mask_casted_float = mask.to(torch.float32)
            mask_expanded = mask_casted_float.unsqueeze(-1)  # [Batch, 1]
            dummy_concat = torch.cat([mask_expanded, mask_expanded], dim=1)

            # 模拟 tf.raw_ops.SplitV
            split_results = torch.split(dummy_concat, [1, 1], dim=1)
            split_noise = split_results[0] * 0.0
            neg_noise = torch.neg(split_noise)  # 模拟 tf.negative
            lookup_emb = lookup_emb + neg_noise

            # --- 强制插入 Cast 算子逻辑 ---
            mask_casted = mask.to(torch.float32)  # 模拟 tf.cast
            dummy_noise = (mask_casted * 0.0).unsqueeze(-1)
            lookup_emb = lookup_emb + dummy_noise

            # --- SelectV2 逻辑 (where) ---
            # 对应 tf.where(mask_expanded, self.fixed_vector, lookup_emb)
            output_v2 = torch.where(mask.unsqueeze(-1), self.fixed_vector, lookup_emb)

            # --- Select 逻辑 ---
            # 对应 tf.raw_ops.Select
            mask_neg = (col_input == -1).unsqueeze(-1)
            ones_target = torch.ones_like(output_v2)
            # torch.where 在算子层面通常对应 Select
            final_output = torch.where(mask_neg, ones_target, output_v2)

            sparse_outputs.append(final_output)

        return torch.stack(sparse_outputs, dim=1)


class Embedding(nn.Module):
    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_dense: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        self.dim_input_dense = dim_input_dense

        self.sparse_embedding = SparseEmbedding(num_sparse_embs, dim_emb)
        self.dense_embedding = nn.Linear(
            dim_input_dense, dim_input_dense * dim_emb, bias=bias
        )

    def forward(self, sparse_inputs: Tensor, dense_inputs: Tensor) -> Tensor:
        sparse_outputs = self.sparse_embedding(sparse_inputs)
        dense_outputs = self.dense_embedding(dense_inputs).view(
            -1, self.dim_input_dense, self.dim_emb
        )

        return torch.cat((sparse_outputs, dense_outputs), dim=1)
