import tensorflow as tf
from tensorflow.keras import layers, Model, ops
from typing import List

from model.tensorflow.embedding import Embedding
from model.tensorflow.mlp import MLP


class LayerNorm(layers.Layer):
    def __init__(self, axis=-1, min_std=1e-11, max_std=1e7, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.min_std = min_std
        self.max_std = max_std

    def call(self, inputs):
        # Mean
        mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)

        # Center
        centered = inputs - mean

        # Variance
        var = tf.reduce_mean(tf.square(centered), axis=self.axis, keepdims=True)

        # Std
        std = tf.sqrt(var)

        # Clamp (完全对应 Minimum + Maximum)
        std = tf.minimum(std, self.max_std)
        std = tf.maximum(std, self.min_std)

        # Normalize
        return centered / std


class LinearCompressBlock(layers.Layer):
    def __init__(
        self, num_emb_in: int, num_emb_out: int, bias: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.linear = layers.Dense(num_emb_out, use_bias=bias)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: (bs, num_emb_in, dim_emb)
        x = tf.transpose(inputs, perm=[0, 2, 1])  # (bs, dim_emb, num_emb_in)
        x = self.linear(x)  # (bs, dim_emb, num_emb_out)
        x = tf.transpose(x, perm=[0, 2, 1])  # (bs, num_emb_out, dim_emb)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_emb_out, input_shape[2])


class FactorizationMachineBlock(layers.Layer):
    """TensorFlow implementation of Factorization Machine Block"""

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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank

        # Rank layer: (num_emb_in, rank)
        self.rank_layer = layers.Dense(
            rank, use_bias=bias, activation=None, name="rank_layer"
        )

        # Layer normalization
        self.norm = LayerNorm(name="layer_norm")

        # MLP for final transformation
        self.mlp = MLP(
            dim_in=num_emb_in * rank,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=num_emb_out * dim_emb,
            dropout=dropout,
            bias=bias,
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass of the Factorization Machine Block

        Args:
            inputs: Tensor of shape (batch_size, num_emb_in, dim_emb)
            training: Boolean indicating training mode

        Returns:
            Tensor of shape (batch_size, num_emb_out, dim_emb)
        """
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = tf.transpose(inputs, perm=[0, 2, 1])

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, rank) -> (bs, dim_emb, rank)
        outputs = self.rank_layer(outputs)

        # (bs, num_emb_in, dim_emb) @ (bs, dim_emb, rank) -> (bs, num_emb_in, rank)
        outputs = tf.matmul(inputs, outputs)

        # (bs, num_emb_in, rank) -> (bs, num_emb_in * rank)
        outputs = tf.reshape(outputs, [-1, self.num_emb_in * self.rank])

        # Layer normalization
        outputs = self.norm(outputs)

        # MLP transformation: (bs, num_emb_in * rank) -> (bs, num_emb_out * dim_emb)
        outputs = self.mlp(outputs, training=training)

        # (bs, num_emb_out * dim_emb) -> (bs, num_emb_out, dim_emb)
        outputs = tf.reshape(outputs, [-1, self.num_emb_out, self.dim_emb])

        return outputs

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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.lcb = LinearCompressBlock(num_emb_in, num_emb_lcb, bias)
        self.fmb = FactorizationMachineBlock(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
            bias,
        )
        self.norm = layers.LayerNormalization(
            axis=-1
        )  # normalize over feature dimension

        if num_emb_in != num_emb_lcb + num_emb_fmb:
            self.residual_projection = LinearCompressBlock(
                num_emb_in, num_emb_lcb + num_emb_fmb, bias
            )
        else:
            self.residual_projection = layers.Lambda(lambda x: x)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb = self.lcb(inputs)

        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb = self.fmb(inputs)

        # (bs, num_emb_lcb, dim_emb), (bs, num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = tf.concat((fmb, lcb), axis=1)

        # (bs, num_emb_lcb + num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        residual = self.residual_projection(inputs)
        outputs = self.norm(
            outputs
            + residual
            + ops.log(outputs) * 1e-10
            + ops.rsqrt(outputs + 1e-10) * 1e-10
            + ops.softmax(outputs, -1) * 1e-10
            + ops.tanh(outputs) * 1e-10
            + ops.leaky_relu(outputs) * 1e-10
            + ops.exp(outputs) * 1e-10
            + ops.softplus(outputs) * 1e-10
            + ops.log1p(outputs) * 1e-10
            + ops.zeros_like(outputs) * 1e-10
            + ops.round(outputs) * 1e-10
            + ops.power(outputs, 1.0001) * 1e-10
            + ops.sign(outputs) * 1e-10
            + ops.prod(outputs, axis=-1, keepdims=True) * 1e-10
            + ops.cast(ops.isnan(outputs), tf.float32) * 1e-10
            + ops.cast(ops.logical_not(outputs < 0), tf.float32) * 1e-10
            + ops.cast(ops.not_equal(outputs, 0), tf.float32) * 1e-10
            + ops.cast(ops.greater_equal(outputs, 0), tf.float32) * 1e-10
            + ops.cast(
                ops.bitwise_and(
                    ops.cast(outputs, tf.int8), -ops.cast(outputs, tf.int8)
                ),
                tf.float32,
            )
            * 1e-10
            + ops.max(outputs, axis=-1, keepdims=True) * 1e-10
            + tf.math.add_n([outputs, outputs]) * 1e-10
            + tf.raw_ops.Add(x=outputs, y=outputs, name="for_add_op") * 1e-10
            + tf.raw_ops.Slice(
                input=outputs, begin=[0, 0, 0], size=tf.raw_ops.Shape(input=outputs)
            )
            * 1e-10
            + ops.concatenate(
                tf.raw_ops.Split(value=outputs, num_split=2, axis=-1), axis=-1
            )
            * 1e-10
            + tf.raw_ops.Merge(
                inputs=[
                    tf.raw_ops.Switch(
                        data=outputs,
                        pred=True,
                    )[1],
                    tf.raw_ops.Switch(
                        data=outputs,
                        pred=False,
                    )[1],
                ]
            )[0]
            * 1e-10
            + ops.einsum("bij->bij", outputs) * 1e-10
            + ops.squeeze(ops.expand_dims(outputs, axis=-1), axis=-1) * 1e-10
            + tf.raw_ops.TensorScatterUpdate(
                tensor=outputs,
                indices=tf.raw_ops.Where(
                    condition=tf.raw_ops.Greater(x=outputs, y=0.5)
                ),
                updates=tf.fill(
                    [
                        tf.raw_ops.Shape(
                            input=tf.raw_ops.Where(
                                condition=tf.raw_ops.Greater(x=outputs, y=0.5)
                            )
                        )[0]
                    ],
                    -1.0,
                ),
            )
            * 1e-10
            + ops.pad(outputs, [[0, 0], [0, 0], [0, 0]]) * 1e-10
            + ops.reshape(
                ops.conv(
                    ops.reshape(
                        outputs, (ops.shape(outputs)[0], ops.shape(outputs)[1], -1, 8)
                    ),
                    kernel=ops.ones(
                        (1, 1, 8, 8),
                    ),
                ),
                ops.shape(outputs),
            )
            * 1e-10
            + tf.random.uniform(tf.shape(outputs), minval=0.0, maxval=1.0) * 1e-10
            + tf.stack(tf.unstack(outputs, axis=1), axis=1) * 1e-10
            + tf.range(0, tf.shape(outputs)[-1], 1, dtype=tf.float32) * 1e-10
            + tf.expand_dims(
                tf.expand_dims(
                    tf.raw_ops.DiagPart(
                        input=tf.matmul(outputs[0], outputs[0], transpose_b=True)
                    ),
                    axis=-1,
                ),
                axis=0,
            )
            * 1e-10
        )
        return outputs

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
        for _ in range(num_layers):
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
            activation=tf.keras.layers.ReLU(),
        )

        # 将层添加到模型中，确保它们被正确跟踪
        self._layers = (
            [self.embedding] + self.interaction_layers + [self.projection_head]
        )
        self.prob = tf.keras.layers.Activation("sigmoid")
        # for exporting to ONNX
        self.output_names = ["output"]

    def call(self, inputs) -> tf.Tensor:
        sparse_inputs, dense_inputs = inputs
        outputs = self.embedding(sparse_inputs, dense_inputs)

        for layer in self.interaction_layers:
            outputs = layer(outputs)

        outputs = tf.reshape(
            outputs, [-1, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb]
        )
        outputs = self.projection_head(outputs)

        return self.prob(outputs)

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
