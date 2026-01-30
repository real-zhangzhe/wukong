import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

from model.tensorflow.wukong import Wukong


BATCH_SIZE = 2
NUM_CAT_FEATURES = 26
NUM_DENSE_FEATURES = 13
NUM_SPARSE_EMBS = [10] * NUM_CAT_FEATURES
DIM_EMB = 128

model = Wukong(
    num_layers=2,
    num_sparse_embs=NUM_SPARSE_EMBS,
    dim_emb=DIM_EMB,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    num_emb_lcb=16,
    num_emb_fmb=23,
    rank_fmb=8,
    num_hidden_wukong=2,
    dim_hidden_wukong=16,
    num_hidden_head=2,
    dim_hidden_head=32,
    dim_output=1,
    dropout=0.5,
    bias=True,
)

# 构造输入
sparse_inputs = tf.constant(
    np.column_stack(
        [
            np.random.randint(0, NUM_SPARSE_EMBS[i], size=BATCH_SIZE)
            for i in range(NUM_CAT_FEATURES)
        ]
    ).astype(np.int32)
)

dense_inputs = tf.constant(
    np.random.rand(BATCH_SIZE, NUM_DENSE_FEATURES).astype(np.float32)
)

# 前向跑一遍，确保模型 build
_ = model((sparse_inputs, dense_inputs))

# ===== 关键部分：冻结成 GraphDef =====


@tf.function(
    input_signature=[
        tf.TensorSpec(
            shape=[None, NUM_CAT_FEATURES],
            dtype=tf.int32,
            name="sparse_inputs",
        ),
        tf.TensorSpec(
            shape=[None, NUM_DENSE_FEATURES],
            dtype=tf.float32,
            name="dense_inputs",
        ),
    ]
)
def inference(sparse_inputs, dense_inputs):
    return model((sparse_inputs, dense_inputs), training=False)


# 获取 ConcreteFunction
concrete_func = inference.get_concrete_function()

# 冻结变量
frozen_func = convert_variables_to_constants_v2(concrete_func)
graph_def = frozen_func.graph.as_graph_def()

input_names = []
input_types = []
for node in graph_def.node:
    if node.op == "Placeholder":
        input_names.append(node.name)
        input_types.append(node.attr["dtype"].type)

# 2. inference 优化（关键）
optimized_graph_def = optimize_for_inference(
    graph_def,
    input_node_names=input_names,
    output_node_names=[frozen_func.outputs[0].op.name],
    placeholder_type_enum=input_types,
)

# 保存 GraphDef
with tf.io.gfile.GFile("wukong_frozen_graph.pb", "wb") as f:
    f.write(optimized_graph_def.SerializeToString())

# 保存 op set
opset = set()
for op in optimized_graph_def.node:
    opset.add(op.op)
opset = sorted(list(opset))
with open("wukong_opset.txt", "w") as f:
    for op in opset:
        f.write(f"{op}\n")
print("Wukong model opset saved to wukong_opset.txt")

print("Frozen GraphDef saved to wukong_frozen_graph.pb")
print("Inputs:", [t.name for t in frozen_func.inputs])
print("Outputs:", [t.name for t in frozen_func.outputs])
