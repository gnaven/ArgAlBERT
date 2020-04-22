import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "driver.py"

if torch.cuda.is_available():
    
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
    
MOSI_ACOUSTIC_DIM = 74
MOSI_VISUAL_DIM = 47

HUMOR_ACOUSTIC_DIM = 81
#HUMOR_VISUAL_DIM = 371
HUMOR_VISUAL_DIM = 91

ACOUSTIC_DIM = 81
VISUAL_DIM = 91

H_MERGE_SENT = 768
DATASET_LOCATION = "/home/gnaven/Debate/ArgAlBert/data"
