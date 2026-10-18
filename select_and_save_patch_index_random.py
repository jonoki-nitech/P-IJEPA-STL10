from simple_ijepa.transformer import  VisionTransformer
from simple_ijepa.ijepa import IJEPA
import torch
from data_utils import load_stl10
from torchvision import transforms
from einops.layers.torch import Rearrange
from torchvision.utils import save_image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
import json
import sys
from tqdm import tqdm
import os


def block_to_index(left, top, w, h):

    unpatchfy = Rearrange("(h w)->h w",
                        h=12, w=12)

    blocks = torch.arange(0, 144)
    blocks = unpatchfy(blocks)

    blocks = blocks[top:top+h, left:left+w]

    index = blocks.flatten().tolist()

    return index


if __name__ == "__main__":

    # params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    # NOTE: unlabeled はこのスクリプトでは使用しないため読み込まない
    stl10_ds_test = load_stl10(split="test",
                               transform=transforms.ToTensor())


    N = len(stl10_ds_test)

    # 先頭N枚のテスト画像のみ処理 (引数未指定 = 全枚)
    n_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if n_limit > 0:
        N = min(n_limit, N)
    T = 20
    block_size = 2

    selected_index_list = []

    os.makedirs("selected_patch_index", exist_ok=True)

    for n in tqdm(range(N)):

        all_index = range(144)

        left = random.choice(range(11))
        top = random.choice(range(11))

        context_indices_list = [block_to_index(left, top, block_size, block_size)]
        target_indices_list = [[all_index]]

        for t in range(T):
            n  = 0
            end_flag = False
            while True:
                left = random.choice(range(11))
                top = random.choice(range(11))
                next_index = block_to_index(left, top, block_size, block_size)
                next_index_not_multiple = [x for x in next_index if x not in context_indices_list[0]]
                if len(next_index_not_multiple) == block_size*block_size:
                    context_indices_list[0] = context_indices_list[0] + next_index
                    break
                n += 1
                if n > 100:
                    end_flag = True
                    break

            if end_flag:
                # print(f"end @ {t+1}/{T}")
                break

        selected_index_list.append(context_indices_list[0])

    with open(f"selected_patch_index/block_{block_size}_tmax_20_test_random_.json", "w") as f:
        json.dump(selected_index_list, f)
