import tensorflow as tf

tf.load_library(
    "/home/albert/Project/tensorflow_musa_extension/build/libmusa_plugin.so"
)
import numpy as np
import random

from model.tensorflow.debug_utils import set_dump_enabled
from model.tensorflow.wukong import Wukong
from model.tensorflow.lr_schedule import LinearWarmup


####################################################################################################
#                                           SET RANDOM SEEDS                                       #
####################################################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

####################################################################################################
#                                  DATASET SPECIFIC CONFIGURATION                                  #
####################################################################################################
NUM_CAT_FEATURES = 26
NUM_DENSE_FEATURES = 13
NUM_SPARSE_EMBS = [1000] * NUM_CAT_FEATURES
DIM_OUTPUT = 1

####################################################################################################
#                                   MODEL SPECIFIC CONFIGURATION                                   #
####################################################################################################
NUM_LAYERS = 1  # number of Wukong layers
DIM_EMB = 32  # dimension of embeddings
NUM_EMB_LCB = 2  # number of low-rank components for embedding compression in LCB
NUM_EMB_FMB = 2  # number of factors for multi-branch factorization in FMB
RANK_FMB = 2  # rank for multi-branch factorization in FMB
NUM_HIDDEN_WUKONG = 2  # number of hidden layers in Wukong MLPs
DIM_HIDDEN_WUKONG = 32  # dimension of hidden layers in Wukong MLPs
NUM_HIDDEN_HEAD = 2  # number of hidden layers in the final prediction head MLPs
DIM_HIDDEN_HEAD = 16  # dimension of hidden layers in the final prediction head
DROPOUT = 0.5  # dropout rate
BIAS = True  # whether to use bias terms in the model

####################################################################################################
#                                           CREATE MODEL                                           #
####################################################################################################
model = Wukong(
    num_layers=NUM_LAYERS,
    num_sparse_embs=NUM_SPARSE_EMBS,
    dim_emb=DIM_EMB,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    num_emb_lcb=NUM_EMB_LCB,
    num_emb_fmb=NUM_EMB_FMB,
    rank_fmb=RANK_FMB,
    num_hidden_wukong=NUM_HIDDEN_WUKONG,
    dim_hidden_wukong=DIM_HIDDEN_WUKONG,
    num_hidden_head=NUM_HIDDEN_HEAD,
    dim_hidden_head=DIM_HIDDEN_HEAD,
    dim_output=DIM_OUTPUT,
    dropout=DROPOUT,
    bias=BIAS,
)

####################################################################################################
#                                  TRAINING SPECIFIC CONFIGURATION                                 #
####################################################################################################
BATCH_SIZE = 2
TRAIN_EPOCHS = 10
PEAK_LR = 0.004
INIT_LR = 1e-8
TOTAL_STEPS_PER_EPOCH = 39291958 // BATCH_SIZE
TOTAL_ITERS = TOTAL_STEPS_PER_EPOCH

lr_schedule = LinearWarmup(
    initial_learning_rate=INIT_LR, peak_learning_rate=PEAK_LR, warmup_steps=TOTAL_ITERS
)
embedding_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
other_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
criterion = tf.keras.losses.BinaryCrossentropy(from_logits=False)

####################################################################################################
#                                    BUILD MODEL & SEPARATE VARS                                   #
####################################################################################################
# TF Lazy Execution
dummy_sparse = tf.zeros((1, NUM_CAT_FEATURES), dtype=tf.int32)
dummy_dense = tf.zeros((1, NUM_DENSE_FEATURES), dtype=tf.float32)
_ = model((dummy_sparse, dummy_dense))
model.load_weights("debug_wukong_weights.h5")
set_dump_enabled()  # 启用 Dump

embedding_parameters = []
other_parameters = []

for var in model.trainable_variables:
    if hasattr(var, "path"):
        # path is available in TF 2.13+
        if "sparse_embedding" in var.path and "embeddings" in var.name:
            embedding_parameters.append(var)
        else:
            other_parameters.append(var)
    else:
        if "sparse_embedding" in var.name:
            embedding_parameters.append(var)
        else:
            other_parameters.append(var)


####################################################################################################
#                                         TRAINING STEP                                            #
####################################################################################################
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, tf.squeeze(outputs))

    grads = tape.gradient(loss, model.trainable_variables)
    emb_grads = []
    other_grads = []

    for grad, var in zip(grads, model.trainable_variables):
        if grad is not None:
            if hasattr(var, "path"):
                # path is available in TF 2.13+
                if "sparse_embedding" in var.path and "embeddings" in var.name:
                    emb_grads.append((grad, var))
                else:
                    other_grads.append((grad, var))
            else:
                if "sparse_embedding" in var.name:
                    emb_grads.append((grad, var))
                else:
                    other_grads.append((grad, var))
    embedding_optimizer.apply_gradients(emb_grads)
    other_optimizer.apply_gradients(other_grads)

    return loss


####################################################################################################
#                                           TRAINING LOOP                                          #
####################################################################################################

inputs = (
    tf.convert_to_tensor(
        np.array(
            [
                [
                    0,
                    101,
                    110,
                    239,
                    3,
                    5,
                    106,
                    5,
                    0,
                    284,
                    101,
                    104,
                    99,
                    0,
                    406,
                    260,
                    1,
                    291,
                    1,
                    2,
                    992,
                    0,
                    1,
                    187,
                    1,
                    2,
                ],
                [
                    22,
                    67,
                    130,
                    111,
                    0,
                    1,
                    220,
                    5,
                    0,
                    296,
                    64,
                    123,
                    63,
                    1,
                    124,
                    120,
                    0,
                    101,
                    1,
                    2,
                    123,
                    0,
                    0,
                    4,
                    1,
                    2,
                ],
            ],
            dtype=np.int32,
        )
    ),
    tf.convert_to_tensor(
        np.array(
            [
                [
                    1.0986123,
                    5.1474943,
                    1.0986123,
                    2.0794415,
                    3.0445225,
                    2.0794415,
                    1.0986123,
                    2.0794415,
                    2.0794415,
                    0.6931472,
                    0.6931472,
                    0.0,
                    2.0794415,
                ],
                [
                    0.0,
                    5.4595857,
                    0.6931472,
                    0.6931472,
                    8.016977,
                    5.1474943,
                    4.158883,
                    2.3978953,
                    5.170484,
                    0.0,
                    2.0794415,
                    0.0,
                    0.6931472,
                ],
            ],
            dtype=np.float32,
        )
    ),
)

labels = tf.convert_to_tensor(np.array([1, 0], dtype=np.float32))

loss = train_step(inputs, labels)
print(f"Loss: {loss.numpy()}")
