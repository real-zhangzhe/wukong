import tensorflow as tf
from tensorflow.keras.layers import Layer

from model.tensorflow.debug_utils import probe


class GELU(Layer):
    """
    GELU激活函数的Layer实现（使用erf版本）
    """

    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, inputs):
        # Dump 输入
        inputs = probe(inputs, f"{self.name}_input")
        out = 0.5 * inputs * (1.0 + tf.math.erf(inputs / tf.sqrt(2.0)))
        # Dump 输出
        return probe(out, f"{self.name}_output")

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(GELU, self).get_config()
        return config


class MLP(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_in: int,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int,
        dropout: float = 0.0,
        bias: bool = False,
        activation: tf.keras.layers.Layer = None,
        name_prefix: str = "mlp",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if activation is None:
            activation = GELU(name="gelu")

        self.hidden_layers = []
        self.name_prefix = name_prefix

        for i in range(num_hidden - 1):
            layers_group = []
            layers_group.append(
                tf.keras.layers.Dense(
                    units=dim_hidden, use_bias=bias, name=f"dense_{i}"
                )
            )
            layers_group.append(tf.keras.layers.BatchNormalization(name=f"bn_{i}"))
            layers_group.append(activation)
            layers_group.append(tf.keras.layers.Dropout(dropout, name=f"dropout_{i}"))
            self.hidden_layers.append(layers_group)

        self.final_dense = tf.keras.layers.Dense(
            units=dim_out, use_bias=bias, name="dense_final"
        )

    def call(self, inputs, training=False):
        x = inputs
        x = probe(x, f"{self.name_prefix}_input")

        for i, layer_group in enumerate(self.hidden_layers):
            # Dense
            x = layer_group[0](x)
            x = probe(x, f"{self.name_prefix}_L{i}_after_dense")

            # BN
            x = layer_group[1](x, training=training)
            x = probe(x, f"{self.name_prefix}_L{i}_after_bn")

            # Act
            x = layer_group[2](x)

            # Dropout
            x = layer_group[3](x, training=training)
            x = probe(x, f"{self.name_prefix}_L{i}_after_dropout")

        x = self.final_dense(x)
        x = probe(x, f"{self.name_prefix}_final_output")

        return x
