import torch
import logging
import sys
from torch.utils.tensorboard import SummaryWriter

from model.pytorch.wukong import Wukong
from data.pytorch.criteo_kaggle_dataset import get_dataloader


####################################################################################################
#                                  DATASET SPECIFIC CONFIGURATION                                  #
####################################################################################################
NPZ_FILE_PATH = "/data/Datasets/criteo-kaggle/kaggleAdDisplayChallenge_processed.npz"  # path to the Criteo Kaggle dataset in .npz format
NUM_CAT_FEATURES = 26  # number of categorical features in Criteo Kaggle dataset
NUM_DENSE_FEATURES = 13  # number of dense features in Criteo Kaggle dataset
# dimensions of sparse embeddings for each categorical feature in Criteo Kaggle dataset
NUM_SPARSE_EMBS = [
    1460,
    583,
    10131227,
    2202608,
    305,
    24,
    12517,
    633,
    3,
    93145,
    5683,
    8351593,
    3194,
    27,
    14992,
    5461306,
    10,
    5652,
    2173,
    4,
    7046547,
    18,
    15,
    286181,
    105,
    142572,
]
DIM_OUTPUT = 1  # dimension of the model output for binary classification

####################################################################################################
#                                   MODEL SPECIFIC CONFIGURATION                                   #
####################################################################################################
NUM_LAYERS = 4  # number of Wukong layers
DIM_EMB = 128  # dimension of embeddings
NUM_EMB_LCB = 32  # number of low-rank components for embedding compression in LCB
NUM_EMB_FMB = 32  # number of factors for multi-branch factorization in FMB
RANK_FMB = 24  # rank for multi-branch factorization in FMB
NUM_HIDDEN_WUKONG = 3  # number of hidden layers in Wukong MLPs
DIM_HIDDEN_WUKONG = 2048  # dimension of hidden layers in Wukong MLPs
NUM_HIDDEN_HEAD = 2  # number of hidden layers in the final prediction head MLP
DIM_HIDDEN_HEAD = 512  # dimension of hidden layers in the final prediction head
DROPOUT = 0.1  # dropout rate

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
)

####################################################################################################
#                                  TRAINING SPECIFIC CONFIGURATION                                 #
####################################################################################################
DEVICE = torch.device("musa")
BATCH_SIZE = 4096  # training batch size
TRAIN_EPOCHS = 3  # number of training epochs
CRITRION = (
    torch.nn.BCEWithLogitsLoss()
)  # binary cross-entropy loss for binary classification
OPTIMIZER = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=1e-5
)  # Adam optimizer

####################################################################################################
#                                       CREATE DATALOADER                                          #
####################################################################################################
dataloader = get_dataloader(
    npz_file_path=NPZ_FILE_PATH,
    split="train",
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)

####################################################################################################
#                                         CREATE LOGGER                                            #
####################################################################################################
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger = logging.getLogger("wukong_training")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

####################################################################################################
#                                           TRAINING LOOP                                          #
####################################################################################################
model = model.to(DEVICE).train()
for epoch in range(TRAIN_EPOCHS):
    for batch_idx, (sparse_inputs, dense_inputs, labels) in enumerate(dataloader):
        outputs = model(sparse_inputs.to(DEVICE), dense_inputs.to(DEVICE))
        loss = CRITRION(outputs, labels.to(DEVICE))
        model.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        if (batch_idx + 1) % 100 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{TRAIN_EPOCHS}], "
                f"Batch [{batch_idx+1}/{len(dataloader)}], "
                f"Loss: {loss.item():.4f}"
            )
