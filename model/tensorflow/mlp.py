import tensorflow as tf
from tensorflow.keras.layers import Layer


class GELU(Layer):
    """
    GELU激活函数的Layer实现（使用erf版本）

    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    参考: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, inputs):
        """
        前向传播计算

        参数:
            inputs: 输入张量

        返回:
            经过GELU激活的输出张量
        """
        return 0.5 * inputs * (1.0 + tf.math.erf(inputs / tf.sqrt(2.0)))

    def compute_output_shape(self, input_shape):
        """保持输入形状不变"""
        return input_shape

    def get_config(self):
        """获取配置信息，用于模型保存和加载"""
        config = super(GELU, self).get_config()
        return config


class MLP(tf.keras.Sequential):
    def __init__(
        self,
        dim_in: int,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int,
        dropout: float = 0.0,
        bias: bool = False,
        activation: tf.keras.layers.Layer = GELU(),
    ) -> None:
        layers = []

        for _ in range(num_hidden - 1):
            layers.append(tf.keras.layers.Dense(units=dim_hidden, use_bias=bias))
            layers.append(tf.keras.layers.BatchNormalization())
            layers.append(activation)
            layers.append(tf.keras.layers.Dropout(dropout))

        layers.append(tf.keras.layers.Dense(units=dim_out, use_bias=bias))

        super().__init__(layers)
