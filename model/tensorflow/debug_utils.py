import tensorflow as tf
import numpy as np
import os

# 配置 Dump 路径
DUMP_DIR = os.environ.get("DUMP_DIR", "./debug_dumps")
os.makedirs(DUMP_DIR, exist_ok=True)

ENABLE_DUMP = False  # 默认为 False，Build 阶段不 Dump

forward_dump_count = 0
backward_dump_count = 0


def set_dump_enabled():
    global ENABLE_DUMP
    ENABLE_DUMP = True
    print(f"[Debug] Dump enabled: {ENABLE_DUMP}")


def save_tensor_to_disk(tensor_np, name):
    """
    将 numpy 数组保存到磁盘
    """
    filename = os.path.join(DUMP_DIR, f"{name}.npy")
    np.save(filename, tensor_np)
    # 打印日志可选，防止刷屏建议注释
    # print(f"[Debug] Dumped {name} shape={tensor_np.shape}")
    return


def probe(x, name):
    """
    探针函数：
    1. 前向传播时：Dump 输入 x
    2. 反向传播时：Dump 梯度 dy
    3. 恒等映射，不改变数值逻辑
    """

    @tf.custom_gradient
    def _probe_op(input_tensor):
        # --- 前向 Dump 逻辑 ---
        def _fwd_dump(t):
            if ENABLE_DUMP:
                global forward_dump_count
                save_tensor_to_disk(t, name=f"fwd_{forward_dump_count:03d}_{name}")
                forward_dump_count += 1
            # 必须返回与 Tout 类型匹配的值
            return np.int32(0)

        # Tout=[tf.int32]：告诉 TF 这个函数会返回一个 int32 tensor
        # 这样我们拿到一个 op handle (dummy_op)
        dummy_op = tf.numpy_function(
            _fwd_dump, [input_tensor], [tf.int32], name=f"dump_fwd_{name}"
        )

        # 关键：使用 control_dependencies 确保 dump 操作在 output_tensor 生成前被执行
        with tf.control_dependencies(dummy_op):
            output_tensor = tf.identity(input_tensor)

        # --- 反向 Dump 逻辑 ---
        def _grad(dy):
            def _bwd_dump(g):
                if ENABLE_DUMP:
                    global backward_dump_count
                    save_tensor_to_disk(g, name=f"bwd_{backward_dump_count:03d}_{name}")
                    backward_dump_count += 1
                return np.int32(0)

            dummy_grad_op = tf.numpy_function(
                _bwd_dump, [dy], [tf.int32], name=f"dump_bwd_{name}"
            )

            with tf.control_dependencies(dummy_grad_op):
                return tf.identity(dy)

        return output_tensor, _grad

    return _probe_op(x)
