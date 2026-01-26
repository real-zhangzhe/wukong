import tensorflow as tf
from typing import List


class SparseEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_sparse_embs: List[int], dim_emb: int):
        super().__init__()
        self.dim_emb = dim_emb
        self.embeddings = [
            tf.keras.layers.Embedding(input_dim=num_emb, output_dim=dim_emb)
            for num_emb in num_sparse_embs
        ]

        self.fixed_vector = self.add_weight(
            name="fixed_zero_vector",
            shape=(dim_emb,),
            initializer="zeros",
            trainable=False,
        )

    def call(self, sparse_inputs):
        sparse_outputs = []

        for i, embedding in enumerate(self.embeddings):
            col_input = sparse_inputs[:, i]

            # 基础 Embedding
            lookup_emb = embedding(tf.abs(col_input))

            # --- Mask 定义 ---
            mask = (tf.equal(col_input, 0) | tf.equal(col_input, -2)) & (
                tf.greater(col_input, 100000)
                | tf.equal(tf.reduce_sum(col_input, 0, keepdims=True), 0)
            )  # [Batch]
            mask_casted = tf.cast(mask, dtype=tf.float32)
            mask_casted = tf.expand_dims(mask_casted, axis=-1)  # [Batch, 1]
            dummy_concat = tf.concat([mask_casted, mask_casted], axis=1)

            # 强制调用 SplitV
            # split_results 会包含两个 [Batch, 1] 的 Tensor
            split_results = tf.raw_ops.SplitV(
                value=dummy_concat,
                size_splits=tf.constant([1, 1], dtype=tf.int64),  # 切分尺寸，决定了是 V
                axis=tf.constant(1, dtype=tf.int32),
                num_split=2,
            )

            # 取出其中一片，乘以 0，变成无用的噪声
            split_noise = split_results[0] * 0.0  # [Batch, 1]
            neg_noise = tf.negative(split_noise)
            lookup_emb = lookup_emb + neg_noise

            # =======================================================
            # 关键修改：强制插入 Cast 算子
            # =======================================================
            # 1. 将 Bool Cast 为 Float32 (产生 Cast 节点)
            #    算子: Cast (DataType: DT_BOOL -> DT_FLOAT)
            mask_casted = tf.cast(mask, dtype=tf.float32)

            # 2. 变为 0 值 (不影响数值)
            #    注意：这里一定要用乘法，不要直接定义常数 0，
            #    否则 Cast 可能会因为后续没被使用而被剪枝 (Dead Code Elimination)。
            dummy_noise = mask_casted * 0.0

            # 3. 维度对齐 (Batch, ) -> (Batch, 1) 以便广播
            dummy_noise = tf.expand_dims(dummy_noise, axis=-1)

            # 4. 注入到主链路
            #    lookup_emb 数值没变，但在图中 lookup_emb 现在依赖于 mask_casted
            lookup_emb = lookup_emb + dummy_noise
            # =======================================================

            # --- 下面是之前的 Select / SelectV2 逻辑 (保持不变) ---

            # SelectV2 (Broadcasting)
            mask_expanded = tf.expand_dims(mask, -1)
            output_v2 = tf.where(mask_expanded, self.fixed_vector, lookup_emb)

            # Select (Legacy, Strict Shape)
            mask_neg = tf.equal(col_input, -1)
            mask_neg = tf.expand_dims(mask_neg, -1)
            mask_neg_tiled = tf.tile(mask_neg, [1, self.dim_emb])
            ones_target = tf.ones_like(output_v2)

            final_output = tf.raw_ops.Select(
                condition=mask_neg_tiled, x=ones_target, y=output_v2
            )

            sparse_outputs.append(final_output)

        return tf.stack(sparse_outputs, axis=1)


class Embedding(tf.keras.layers.Layer):
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
        self.dense_embedding = tf.keras.layers.Dense(
            units=dim_input_dense * dim_emb, use_bias=bias
        )

    def call(self, sparse_inputs: tf.Tensor, dense_inputs: tf.Tensor) -> tf.Tensor:
        sparse_outputs = self.sparse_embedding(sparse_inputs)

        dense_outputs = self.dense_embedding(dense_inputs)
        dense_outputs = tf.reshape(
            dense_outputs, [-1, self.dim_input_dense, self.dim_emb]
        )

        # concat along feature axis
        return tf.concat((sparse_outputs, dense_outputs), axis=1)
