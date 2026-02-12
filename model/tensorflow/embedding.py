import tensorflow as tf
from typing import List

from model.tensorflow.debug_utils import probe


class SparseEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_sparse_embs: List[int], dim_emb: int):
        super().__init__()
        self.dim_emb = dim_emb
        self.embeddings = [
            tf.keras.layers.Embedding(
                input_dim=num_emb, output_dim=dim_emb, name=f"emb_lookup_{i}"
            )
            for i, num_emb in enumerate(num_sparse_embs)
        ]

        self.fixed_vector = self.add_weight(
            name="fixed_zero_vector",
            shape=(dim_emb,),
            initializer="zeros",
            trainable=False,
        )

    def call(self, sparse_inputs):
        sparse_outputs = []

        # Dump 原始输入
        sparse_inputs = probe(sparse_inputs, "SparseEmb_inputs")

        for i, embedding in enumerate(self.embeddings):
            col_input = sparse_inputs[:, i]

            # Dump 当前列输入
            col_input = probe(col_input, f"SparseEmb_col_{i}_input")

            # 基础 Embedding
            lookup_emb = embedding(tf.abs(col_input))
            lookup_emb = probe(lookup_emb, f"SparseEmb_col_{i}_raw_lookup")

            # --- Mask 定义 ---
            mask = (tf.equal(col_input, 0) | tf.equal(col_input, -2)) & (
                tf.greater(col_input, 100000)
                | tf.equal(tf.reduce_sum(col_input, 0, keepdims=True), 0)
            )

            # Dump Mask (Cast 为 float 以便保存)
            probe(tf.cast(mask, tf.float32), f"SparseEmb_col_{i}_mask_bool")

            mask_casted = tf.cast(mask, dtype=tf.float32)
            mask_casted = tf.expand_dims(mask_casted, axis=-1)
            dummy_concat = tf.concat([mask_casted, mask_casted], axis=1)

            # SplitV
            split_results = tf.raw_ops.SplitV(
                value=dummy_concat,
                size_splits=tf.constant([1, 1], dtype=tf.int64),
                axis=tf.constant(1, dtype=tf.int32),
                num_split=2,
            )

            split_noise = split_results[0] * 0.0
            neg_noise = tf.negative(split_noise)

            # 注入 probe 观察噪声
            neg_noise = probe(neg_noise, f"SparseEmb_col_{i}_neg_noise")

            lookup_emb = lookup_emb + neg_noise

            mask_casted = tf.cast(mask, dtype=tf.float32)
            dummy_noise = mask_casted * 0.0
            dummy_noise = tf.expand_dims(dummy_noise, axis=-1)

            dummy_noise = probe(dummy_noise, f"SparseEmb_col_{i}_dummy_noise")

            lookup_emb = lookup_emb + dummy_noise

            mask_expanded = tf.expand_dims(mask, -1)
            output_v2 = tf.where(mask_expanded, self.fixed_vector, lookup_emb)

            mask_neg = tf.equal(col_input, -1)
            mask_neg = tf.expand_dims(mask_neg, -1)
            mask_neg_tiled = tf.tile(mask_neg, [1, self.dim_emb])
            ones_target = tf.ones_like(output_v2)

            final_output = tf.raw_ops.Select(
                condition=mask_neg_tiled, x=ones_target, y=output_v2
            )

            final_output = probe(final_output, f"SparseEmb_col_{i}_final")

            sparse_outputs.append(final_output)

        result = tf.stack(sparse_outputs, axis=1)
        return probe(result, "SparseEmb_output_stacked")


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
            units=dim_input_dense * dim_emb, use_bias=bias, name="dense_emb"
        )

    def call(self, sparse_inputs: tf.Tensor, dense_inputs: tf.Tensor) -> tf.Tensor:
        sparse_outputs = self.sparse_embedding(sparse_inputs)

        dense_inputs = probe(dense_inputs, "DenseEmb_input")
        dense_outputs = self.dense_embedding(dense_inputs)
        dense_outputs = probe(dense_outputs, "DenseEmb_output_raw")

        dense_outputs = tf.reshape(
            dense_outputs, [-1, self.dim_input_dense, self.dim_emb]
        )

        final = tf.concat((sparse_outputs, dense_outputs), axis=1)
        return probe(final, "Embedding_final_concat")
