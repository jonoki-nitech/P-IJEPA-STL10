from data_utils import get_stl10_root, ensure_stl10_splits, STL10Partial
from torchvision import transforms
import json
from simple_ijepa.transformer import  VisionTransformer
import torch
from torch import nn
from simple_ijepa.utils import trunc_normal_
from einops.layers.torch import Rearrange
from lars import LARS
import torch.nn.functional as F
import csv
from torchvision.utils import save_image
from tqdm import tqdm
from mae import MAE
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os

class STL10WithMasks(STL10Partial):
    def __init__(self, root, split='train', input_patch_num=10, mask_list=None, transform=None, target_transform=None, download=False, num_images=None):
        super().__init__(root=root, split=split, transform=transform,
                         target_transform=target_transform)

        # 先頭num_images枚のみを対象にする (None = 全枚)
        if num_images is not None:
            self.data = self.data[:num_images]
            self.labels = self.labels[:num_images]
            mask_list = mask_list[:num_images]

        if mask_list is None:
            raise ValueError("mask_list must be provided.")
        if len(mask_list) != len(self.data):
            raise ValueError("Length of mask_list must match STL10 dataset length.")

        self.mask_list = mask_list   # 例: 各サンプルに対応するマスクのリスト（list of lists）

        self.input_patch_num = input_patch_num
        self.block_size = 2

    def __getitem__(self, idx):
        # STL10 の標準サンプル取得
        img, target = super().__getitem__(idx)

        # ここでマスクを取り出す
        masks = self.mask_list[idx][:self.input_patch_num*self.block_size*self.block_size]   # 例: [mask1, mask2, ...] のリスト
        target_masks = [x for x in range(144) if x not in masks ]

        return img, target, torch.tensor(masks), torch.tensor(target_masks)

class Rec_Model(torch.nn.Module):

    def __init__(self,
                 hidden_emb_dim=512,
                 img_size=96,
                 patch_size=8,
                 depth=12,
                 heads=6):
        super(Rec_Model, self).__init__()

        self.patch_size = patch_size
        self.img_size = img_size

        self.encoder = VisionTransformer(image_size=96,
                                           patch_size=patch_size,
                                           dim=hidden_emb_dim,
                                           depth=depth,
                                           heads=heads,
                                           mlp_dim=hidden_emb_dim * 2)

        self.decoder = VisionTransformer(image_size=96,
                                           patch_size=patch_size,
                                           dim=hidden_emb_dim,
                                           depth=depth,
                                           heads=heads,
                                           mlp_dim=hidden_emb_dim * 2)

        self.decoder_head = nn.Sequential(nn.Linear(hidden_emb_dim, 3*patch_size**2), nn.Tanh())

        # self.decoder_head = nn.Sequential(nn.Linear(hidden_emb_dim, hidden_emb_dim), nn.BatchNorm1d(hidden_emb_dim),
        #                        nn.ReLU(inplace=True), nn.Linear(hidden_emb_dim, hidden_emb_dim), nn.BatchNorm1d(hidden_emb_dim),
        #                        nn.ReLU(inplace=True), nn.Linear(hidden_emb_dim, 3*patch_size**2), nn.Tanh())

        self.mask_token = nn.Parameter(torch.zeros(1, hidden_emb_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)


    def forward(self, img, mask_list, target_mask_list):
        B = img.shape[0]
        target_patch_num = target_mask_list.shape[1]
        pos_emb = self.encoder.pos_embedding.to(img.device)

        # encode
        features = self.encoder(img, patchify=True, pos_embed=True, mask=mask_list)
        pos_embed = torch.stack([pos_emb[mask_list[i], :] for i in range(B)])
        features += pos_embed

        # decode
        mask_tokens = self.mask_token.expand(B, target_patch_num, -1).clone()
        mask_pos_embed = torch.stack([pos_emb[target_mask_list[i], :] for i in range(B)])
        mask_tokens += mask_pos_embed

        decoder_input = torch.cat([features, mask_tokens], dim=1)
        rec_tokens = self.decoder(decoder_input, patchify=False, pos_embed=False)
        rec_tokens = rec_tokens[:, -target_patch_num:]

        b, n, d = rec_tokens.shape
        rec_patches = self.decoder_head(rec_tokens.reshape(b*n, d))
        rec_patches = rec_patches.reshape(b, n, -1)

        # make target patches
        patchify = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size)

        img_patches = patchify(img)
        target_patches = torch.gather(img_patches, 1,
                             target_mask_list.unsqueeze(-1).expand(-1, -1, img_patches.size(-1)))

        return rec_patches, target_patches

def visualize_rec_image(img, rec_patches, target_mask):
    patchify = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=8,
                p2=8)
    unpatchify = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=8,
            p2=8,
            h=12, w=12)
    img_patches = patchify(img)
    img_patches[:, target_mask[0], :] = rec_patches
    img_rec = unpatchify(img_patches)

    return img_rec



if __name__ == "__main__":

    # params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "test"

    block_size = 2

    input_patch_num = 5

    with open(f"selected_patch_index/block_{block_size}_tmax_20_test_.json", "r") as f:
        mask_list_test = json.load(f)
    with open(f"selected_patch_index/block_{block_size}_tmax_20_test_random_.json", "r") as f:
        mask_list_test_random = json.load(f)

    ensure_stl10_splits(("test",))

    # 先頭N枚のテスト画像のみ処理 (引数未指定 = 全枚)
    n_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # dataset
    stl10_ds_test = STL10WithMasks(
    get_stl10_root(),
    split="test",
    download=False,
    transform=transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))]),
    mask_list=mask_list_test,
    input_patch_num=input_patch_num,
    num_images=(n_limit if n_limit > 0 else None)
    )

    stl10_ds_test_random = STL10WithMasks(
    get_stl10_root(),
    split="test",
    download=False,
    transform=transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))]),
    mask_list=mask_list_test_random,
    input_patch_num=input_patch_num,
    num_images=(n_limit if n_limit > 0 else None)
    )

    # train_data_loader = torch.utils.data.DataLoader(
    #     stl10_ds_train,
    #     batch_size=1024,
    #     drop_last=False,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=16,
    #     persistent_workers=False)

    # model
    model = MAE().to(device)

    # test (visualize reconstructed image)
    N = len(stl10_ds_test)
    state_dict = torch.load("models/mae/latest.ckpt", map_location=device)
    model.load_state_dict(state_dict["state_dict"])
    normalize = transforms.Normalize(
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713)
    )
    inv = transforms.Normalize(
                    mean = [-0.4467 / 0.2603, -0.4398 / 0.2566, -0.4066 / 0.2713],
                    std = [1 / 0.2603, 1 / 0.2566, 1 / 0.2713]
                )

    # selected patches
    os.makedirs(f"images/rec_images/n={input_patch_num}/selected", exist_ok=True)
    mse_select_patches = torch.zeros(N)
    for n in tqdm(range(N)):
        x, y, input_mask, target_mask = stl10_ds_test[n]
        x, input_mask, target_mask = x.unsqueeze(0).to(device), input_mask.unsqueeze(0).to(device), target_mask.unsqueeze(0).to(device)
        pred, rec_patches, target_patches, context_patches = model.mae(x, input_mask, target_mask)

        x_input = visualize_rec_image(x, torch.zeros_like(rec_patches), target_mask)
        mask = visualize_rec_image(torch.ones_like(x, dtype=torch.bool), torch.zeros_like(rec_patches, dtype=bool), target_mask)
        pred_image = model.mae.tokens_as_image(pred.detach())
        pred_image[mask] = x_input[mask]


        x_input = inv(x_input)
        x = inv(x)
        pred_image = inv(pred_image)
        img_result = torch.cat([x_input, pred_image, x], dim=0)
        mse = F.mse_loss(x[~mask], pred_image[~mask])
        mse_select_patches[n] = mse.item()
        if (n+1) % 100 == 0 or n < 10:
            vutils.save_image(img_result, "tmp.png")
            image = mpimg.imread("tmp.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"MSE={mse:.6f}", fontsize=10)
            plt.savefig(f"images/rec_images/n={input_patch_num}/selected/{n+1}.png", bbox_inches="tight", pad_inches=0.2)

    torch.save(mse_select_patches, f"images/rec_images/n={input_patch_num}/selected/mse_select_patches.pt")

    # random patches
    os.makedirs(f"images/rec_images/n={input_patch_num}/random", exist_ok=True)
    mse_random_patches = torch.zeros(N)
    for n in tqdm(range(N)):
        x, y, input_mask, target_mask = stl10_ds_test_random[n]
        x, input_mask, target_mask = x.unsqueeze(0).to(device), input_mask.unsqueeze(0).to(device), target_mask.unsqueeze(0).to(device)
        pred, rec_patches, target_patches, context_patches = model.mae(x, input_mask, target_mask)

        x_input = visualize_rec_image(x, torch.zeros_like(rec_patches), target_mask)
        mask = visualize_rec_image(torch.ones_like(x, dtype=torch.bool), torch.zeros_like(rec_patches, dtype=bool), target_mask)
        pred_image = model.mae.tokens_as_image(pred.detach())
        pred_image[mask] = x_input[mask]


        x_input = inv(x_input)
        x = inv(x)
        pred_image = inv(pred_image)
        img_result = torch.cat([x_input, pred_image, x], dim=0)
        mse = F.mse_loss(x[~mask], pred_image[~mask])
        mse_random_patches[n] = mse.item()
        if (n+1) % 100 == 0 or n < 10:
            vutils.save_image(img_result, "tmp.png")
            image = mpimg.imread("tmp.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"MSE={mse:.6f}", fontsize=10)
            plt.savefig(f"images/rec_images/n={input_patch_num}/random/{n+1}.png", bbox_inches="tight", pad_inches=0.2)

    torch.save(mse_random_patches, f"images/rec_images/n={input_patch_num}/random/mse_random_patches.pt")

    print(f"selected: mean={mse_select_patches.mean()} std={mse_select_patches.std()}")
    print(f"random: mean={mse_random_patches.mean()} std={mse_random_patches.std()}")
