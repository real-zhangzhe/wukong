import torch
import numpy as np
import random
import logging
import sys
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter

from model.pytorch.wukong import Wukong
from data.pytorch.criteo_kaggle_dataset import get_dataloader


####################################################################################################
#                                           SET RANDOM SEEDS                                       #
####################################################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

####################################################################################################
#                                         CREATE LOGGER                                            #
####################################################################################################
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H.%M.%S")
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
os.makedirs(f"logs/pytorch/{formatted_time}", exist_ok=True)
file_handler = logging.FileHandler(
    f"logs/pytorch/{formatted_time}/training.log", mode="a", encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger = logging.getLogger("wukong_training")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
LOGGER_PRINT_INTERVAL = 10
writer = SummaryWriter(log_dir=f"logs/pytorch/{formatted_time}/tensorboard")
os.system("cp " + __file__ + f" logs/pytorch/{formatted_time}/")
checkpoint_dir = f"logs/pytorch/{formatted_time}/checkpoints"
SAVE_CHECKPOINTS = False
if SAVE_CHECKPOINTS:
    os.makedirs(checkpoint_dir, exist_ok=True)

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
NUM_LAYERS = 8  # number of Wukong layers
DIM_EMB = 128  # dimension of embeddings
NUM_EMB_LCB = 32  # number of low-rank components for embedding compression in LCB
NUM_EMB_FMB = 32  # number of factors for multi-branch factorization in FMB
RANK_FMB = 24  # rank for multi-branch factorization in FMB
NUM_HIDDEN_WUKONG = 3  # number of hidden layers in Wukong MLPs
DIM_HIDDEN_WUKONG = 2048  # dimension of hidden layers in Wukong MLPs
NUM_HIDDEN_HEAD = 2  # number of hidden layers in the final prediction head MLPs
DIM_HIDDEN_HEAD = 256  # dimension of hidden layers in the final prediction head
DROPOUT = 0.5  # dropout rate
BIAS = True  # whether to use bias terms in the model
DTYPE = torch.float32  # data type for model parameters and computations

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
DEVICE_STR = "musa"
DEVICE = torch.device(DEVICE_STR)
BATCH_SIZE = 4096  # training batch size
TRAIN_EPOCHS = 10  # number of training epochs
PEAK_LR = 0.004  # peak learning rate
INIT_LR = 1e-8  # initial learning rate
critrion = torch.nn.BCELoss()  # binary cross-entropy loss for binary classification
embedding_parameters = [
    param
    for name, param in model.named_parameters()
    if "embedding.sparse_embedding" in name
]
other_parameters = [
    param
    for name, param in model.named_parameters()
    if "embedding.sparse_embedding" not in name
]
embedding_optimizer = torch.optim.SGD(
    embedding_parameters, lr=PEAK_LR
)  # SGD optimizer for embeddings
other_optimizer = torch.optim.Adam(other_parameters, lr=PEAK_LR)  # Adam optimizer
embedding_optimizer_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    embedding_optimizer,
    start_factor=INIT_LR / PEAK_LR,
    total_iters=(39291958 // BATCH_SIZE),
)
other_optimizer_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    other_optimizer,
    start_factor=INIT_LR / PEAK_LR,
    total_iters=(39291958 // BATCH_SIZE),
)

####################################################################################################
#                                       CREATE DATALOADER                                          #
####################################################################################################
train_dataloader = get_dataloader(
    npz_file_path=NPZ_FILE_PATH,
    split="train",
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)
valid_dataloader = get_dataloader(
    npz_file_path=NPZ_FILE_PATH,
    split="valid",
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)


####################################################################################################
#                                          VALID FUNCTION                                          #
####################################################################################################
def validate(model, dataloader):
    model.eval()
    num_samples = 0
    num_correct = 0
    pos_samples = 0
    pos_correct = 0
    with torch.no_grad():
        for sparse_inputs, dense_inputs, labels in dataloader:
            outputs = model(sparse_inputs.to(DEVICE), dense_inputs.to(DEVICE))
            labels = labels.to(DEVICE)
            predictions = outputs.squeeze() >= 0.5
            num_samples += labels.size(0)
            pos_samples += (labels == 1).sum().item()
            pos_correct += ((predictions == 1) & (labels == 1)).sum().item()
            num_correct += (predictions == labels).sum().item()
    model.train()
    accuracy = num_correct / num_samples if num_samples > 0 else 0
    recall_pos = pos_correct / pos_samples if pos_samples > 0 else 0
    return accuracy, num_samples, recall_pos, pos_samples


####################################################################################################
#                                           TRAINING LOOP                                          #
####################################################################################################
scaler = torch.amp.GradScaler(DEVICE_STR, enabled=(torch.float16 == DTYPE))
model.to(DEVICE).train()
logger.info("Use device: " + DEVICE_STR)
logger.info("Use dtype: " + str(DTYPE))
step = 0
for epoch in range(TRAIN_EPOCHS):
    for batch_idx, (sparse_inputs, dense_inputs, labels) in enumerate(train_dataloader):
        if torch.float16 == DTYPE:
            with torch.amp.autocast(DEVICE_STR):
                outputs = model(
                    sparse_inputs.to(DEVICE),
                    dense_inputs.to(DEVICE).to(torch.float16),
                )
                loss = critrion(
                    outputs.squeeze(),
                    labels.to(DEVICE).to(torch.float16),
                )
            model.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(embedding_optimizer)
            scaler.step(other_optimizer)
            scaler.update()
            embedding_optimizer_lr_scheduler.step()
            other_optimizer_lr_scheduler.step()
        elif torch.float32 == DTYPE:
            outputs = model(sparse_inputs.to(DEVICE), dense_inputs.to(DEVICE))
            loss = critrion(outputs.squeeze(), labels.to(DEVICE).to(torch.float32))
            model.zero_grad()
            loss.backward()
            embedding_optimizer.step()
            other_optimizer.step()
            embedding_optimizer_lr_scheduler.step()
            other_optimizer_lr_scheduler.step()
        else:
            raise ValueError("Unsupported DTYPE. Use torch.float16 or torch.float32.")
        if (batch_idx + 1) % LOGGER_PRINT_INTERVAL == 0:
            logger.info(
                f"Epoch [{epoch+1}/{TRAIN_EPOCHS}], "
                f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                f"Loss: {loss.item():.4f}, "
                f"Embedding LR: {embedding_optimizer_lr_scheduler.get_last_lr()[0]:.6f}, "
                f"Other LR: {other_optimizer_lr_scheduler.get_last_lr()[0]:.6f}"
            )
        writer.add_scalar("training_loss", loss, step)
        writer.add_scalar(
            "embedding_optimizer_lr",
            embedding_optimizer_lr_scheduler.get_last_lr()[0],
            step,
        )
        writer.add_scalar(
            "other_optimizer_lr", other_optimizer_lr_scheduler.get_last_lr()[0], step
        )
        step += 1
    # Validate the model at the end of each epoch
    accuracy, num_samples, recall_pos, pos_samples = validate(model, valid_dataloader)
    logger.info(
        f"Validation after Epoch {epoch+1}: "
        f"Accuracy: {accuracy*100:.2f}%, "
        f"Total Samples: {num_samples}, "
        f"Positive Recall: {recall_pos*100:.2f}%, "
        f"Positive Samples: {pos_samples}"
    )
    writer.add_scalar("validation_accuracy", accuracy, epoch + 1)
    writer.add_scalar("validation_recall_pos", recall_pos, epoch + 1)
    if SAVE_CHECKPOINTS:
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir, f"wukong_epoch_{epoch+1}.pth"),
        )
        logger.info(
            f"Model checkpoint saved for epoch {epoch+1} at {checkpoint_dir}/wukong_epoch_{epoch+1}.pth"
        )
