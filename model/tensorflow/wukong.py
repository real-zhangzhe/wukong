import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List

from model.tensorflow.embedding import Embedding
from model.tensorflow.mlp import MLP, GELU
from model.tensorflow.debug_utils import probe


class LayerNorm(layers.Layer):
    def __init__(self, axis=-1, min_std=1e-11, max_std=1e7, name_prefix="ln", **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.min_std = min_std
        self.max_std = max_std
        self.name_prefix = name_prefix

    def call(self, inputs):
        inputs = probe(inputs, f"{self.name_prefix}_input")

        mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        centered = inputs - mean
        var = tf.reduce_mean(tf.square(centered), axis=self.axis, keepdims=True)
        std = tf.sqrt(var)

        # 观察 clamp 前后的 std
        std = probe(std, f"{self.name_prefix}_std_raw")
        std = tf.minimum(std, self.max_std)
        std = tf.maximum(std, self.min_std)
        std = probe(std, f"{self.name_prefix}_std_clamped")

        out = centered / std
        return probe(out, f"{self.name_prefix}_output")


class LinearCompressBlock(layers.Layer):
    def __init__(
        self,
        num_emb_in: int,
        num_emb_out: int,
        bias: bool = False,
        name_prefix="lcb",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.name_prefix = name_prefix
        self.linear = layers.Dense(
            num_emb_out, use_bias=bias, name=f"{name_prefix}_dense"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = probe(inputs, f"{self.name_prefix}_input")

        x = tf.transpose(inputs, perm=[0, 2, 1])
        x = probe(x, f"{self.name_prefix}_transposed_1")

        x = self.linear(x)
        x = probe(x, f"{self.name_prefix}_after_linear")

        x = tf.transpose(x, perm=[0, 2, 1])
        x = probe(x, f"{self.name_prefix}_output")
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_emb_out, input_shape[2])


class FactorizationMachineBlock(layers.Layer):
    def __init__(
        self,
        num_emb_in: int,
        num_emb_out: int,
        dim_emb: int,
        rank: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        bias: bool = False,
        name_prefix: str = "fmb",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.name_prefix = name_prefix
        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank

        self.rank_layer = layers.Dense(
            rank, use_bias=bias, activation=None, name="rank_layer"
        )
        self.norm = LayerNorm(name="layer_norm", name_prefix=f"{name_prefix}_ln")

        # 传递 name_prefix 到 MLP
        self.mlp = MLP(
            dim_in=num_emb_in * rank,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=num_emb_out * dim_emb,
            dropout=dropout,
            bias=bias,
            name_prefix=f"{name_prefix}_mlp",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs = probe(inputs, f"{self.name_prefix}_input")

        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = tf.transpose(inputs, perm=[0, 2, 1])
        outputs = self.rank_layer(outputs)  # (bs, dim_emb, rank)
        outputs = probe(outputs, f"{self.name_prefix}_rank_out")

        # Interaction
        outputs = tf.matmul(inputs, outputs)  # (bs, num_emb_in, rank)
        outputs = probe(outputs, f"{self.name_prefix}_matmul_out")

        outputs = tf.reshape(outputs, [-1, self.num_emb_in * self.rank])
        outputs = self.norm(outputs)

        outputs = self.mlp(outputs, training=training)

        outputs = tf.reshape(outputs, [-1, self.num_emb_out, self.dim_emb])
        return probe(outputs, f"{self.name_prefix}_final_out")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_emb_in": self.num_emb_in,
                "num_emb_out": self.num_emb_out,
                "dim_emb": self.dim_emb,
                "rank": self.rank,
                "num_hidden": self.mlp.num_hidden,
                "dim_hidden": self.mlp.dim_hidden,
                "dropout": self.mlp.dropout_rate,
                "bias": self.mlp.use_bias,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class WukongLayer(layers.Layer):
    def __init__(
        self,
        num_emb_in: int,
        dim_emb: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        bias: bool = False,
        layer_idx: int = 0,  # 新增：用于区分不同的 WukongLayer
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        prefix = f"wukong_L{layer_idx}"
        self.prefix = prefix

        self.lcb = LinearCompressBlock(
            num_emb_in, num_emb_lcb, bias, name_prefix=f"{prefix}_lcb"
        )
        self.fmb = FactorizationMachineBlock(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
            bias,
            name_prefix=f"{prefix}_fmb",
        )
        self.norm = layers.LayerNormalization(axis=-1, name=f"{prefix}_ln")

        if num_emb_in != num_emb_lcb + num_emb_fmb:
            self.residual_projection = LinearCompressBlock(
                num_emb_in,
                num_emb_lcb + num_emb_fmb,
                bias,
                name_prefix=f"{prefix}_res_proj",
            )
        else:
            self.residual_projection = layers.Lambda(lambda x: x)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = probe(inputs, f"{self.prefix}_input")

        lcb = self.lcb(inputs)
        fmb = self.fmb(inputs)

        outputs = tf.concat((fmb, lcb), axis=1)
        outputs = probe(outputs, f"{self.prefix}_concat")

        residual = self.residual_projection(inputs)
        residual = probe(residual, f"{self.prefix}_residual_val")

        outputs = self.norm(outputs + residual)
        return probe(outputs, f"{self.prefix}_output")

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.lcb.num_emb_out + self.fmb.num_emb_out,
            input_shape[2],
        )


class Wukong(Model):
    def __init__(
        self,
        num_layers: int,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden_wukong: int,
        dim_hidden_wukong: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        dim_output: int,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.dim_emb = dim_emb
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb

        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)

        num_emb_in = dim_input_sparse + dim_input_dense
        self.interaction_layers = []
        for i in range(num_layers):
            layer = WukongLayer(
                num_emb_in,
                dim_emb,
                num_emb_lcb,
                num_emb_fmb,
                rank_fmb,
                num_hidden_wukong,
                dim_hidden_wukong,
                dropout,
                bias,
                layer_idx=i,  # 传入索引
            )
            self.interaction_layers.append(layer)
            num_emb_in = num_emb_lcb + num_emb_fmb

        self.projection_head = MLP(
            (num_emb_lcb + num_emb_fmb) * dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
            bias,
            activation=GELU(),
            name_prefix="head_mlp",
        )

        self._layers = (
            [self.embedding] + self.interaction_layers + [self.projection_head]
        )
        self.prob = tf.keras.layers.Activation("sigmoid")
        self.output_names = ["output"]

    def call(self, inputs) -> tf.Tensor:
        sparse_inputs, dense_inputs = inputs
        # 输入 probe
        sparse_inputs = probe(sparse_inputs, "Wukong_input_sparse")
        dense_inputs = probe(dense_inputs, "Wukong_input_dense")

        outputs = self.embedding(sparse_inputs, dense_inputs)

        for layer in self.interaction_layers:
            outputs = layer(outputs)

        outputs = tf.reshape(
            outputs, [-1, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb]
        )
        outputs = probe(outputs, "Wukong_before_head")

        outputs = self.projection_head(outputs)

        final = self.prob(outputs)
        return probe(final, "Wukong_final_output")

    def build(self, input_shape):
        sparse_shape, dense_shape = input_shape

        self.embedding.build([sparse_shape, dense_shape])

        dummy_sparse = tf.zeros(sparse_shape)
        dummy_dense = tf.zeros(dense_shape)
        emb_output = self.embedding(dummy_sparse, dummy_dense)

        current_output = emb_output
        for layer in self.interaction_layers:
            layer.build(current_output.shape)
            current_output = layer(current_output)

        final_shape = [None, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb]
        self.projection_head.build(final_shape)
