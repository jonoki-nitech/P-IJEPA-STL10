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

def reconstruct_img_from_patch(ori_img, patch_index, patch_size=8):

    h = 96//patch_size
    w = 96//patch_size

    patchfy = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                        p1=patch_size,
                        p2=patch_size)

    unpatchfy = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                        p1=patch_size,
                        p2=patch_size,
                        h=h, w=w)

    patches = patchfy(ori_img)

    for i in range(patches.shape[1]):
        if not i in patch_index:
            patches[:, i] = torch.zeros(patch_size**2*3)

    img = unpatchfy(patches)

    return img

def sigma_heatmap(sigma, patch_index=None, patch_size=8):

    h = 96//patch_size
    w = 96//patch_size

    unpatchfy = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                        p1=1,
                        p2=1,
                        h=h, w=w)

    sigma = unpatchfy(sigma)

    sigma = F.interpolate(sigma, size=(96, 96))

    sigma = to_color_tensor(sigma.squeeze())

    return sigma.unsqueeze(0)

def to_color_tensor(x, cmap='viridis'):
    # x: (H, W)
    # x_min, x_max = x.min(), x.max()
    x_min, x_max=0.25, 0.55
    x_norm = (x - x_min) / (x_max - x_min)
    colormap = cm.get_cmap(cmap)
    x_color = torch.from_numpy(colormap(x_norm.numpy())[:, :, :3])  # (H, W, 3)
    x_color = x_color.permute(2, 0, 1)  # (3, H, W)
    return x_color

def block_to_index(left, top, w, h):

    unpatchfy = Rearrange("(h w)->h w",
                        h=12, w=12)

    blocks = torch.arange(0, 144)
    blocks = unpatchfy(blocks)

    blocks = blocks[top:top+h, left:left+w]

    index = blocks.flatten().tolist()

    return index

def select_next_block(sigma, patch_size=8, k=4, stride=1):

    h = 96//patch_size
    w = 96//patch_size

    index = torch.arange(sigma.shape[1])

    sigma = sigma.reshape(1, 1, h, w)
    index = index.reshape(1, 1, h, w)

    sigma_avg = F.avg_pool2d(sigma, kernel_size=k, stride=1)
    max_idx_flat = torch.argmax(sigma_avg)
    max_idx_pooled = torch.unravel_index(max_idx_flat, sigma_avg.shape[-2:])

    top = max_idx_pooled[0] * stride
    left = max_idx_pooled[1] * stride
    # end_h = start_h + k
    # end_w = start_w + k

    return block_to_index(left, top, k, k)

if __name__ == "__main__":

    # params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    # NOTE: unlabeled/train はこのスクリプトでは使用しないため読み込まない
    stl10_ds_test = load_stl10(split="test",
                               transform=transforms.ToTensor())

    # 先頭N枚のテスト画像のみ処理 (引数未指定 = 全枚)
    n_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if n_limit > 0:
        stl10_ds_test = torch.utils.data.Subset(
            stl10_ds_test, range(min(n_limit, len(stl10_ds_test))))

    dataloader = torch.utils.data.DataLoader(
        stl10_ds_test,
        batch_size=256,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        # CPU (特にWindows) ではworker起動のspawnが重いため0にする
        num_workers=(16 if device.type == "cuda" else 0),
        persistent_workers=False)

    # model
    encoder = VisionTransformer(
    image_size=96,
    patch_size=8,
    dim=512,
    depth=6,
    heads=6,
    mlp_dim=1024,
    ).to(device)

    ijepa = IJEPA(encoder,
                hidden_emb_dim=512,
                patch_size=8,
                num_targets=1)

    state_dict = torch.load("models/embdim=512_epoch=300_patchsize=8/training_model.pth", map_location=device)
    ijepa.load_state_dict(state_dict)
    ijepa = ijepa.to(device)


    # inference
    ijepa.eval()

    T = 20
    block_size = 2

    selected_index_list = []

    os.makedirs("selected_patch_index", exist_ok=True)

    for img, label in tqdm(dataloader):

        batch_size = img.shape[0]

        all_index = range(144)

        context_indices_list = []
        target_indices_list = []

        for i in range(batch_size):
            left = random.choice(range(11))
            top = random.choice(range(11))

            context_indices_list.append(block_to_index(left, top, block_size, block_size))
            target_indices_list.append([all_index])

        # context_images_list = []
        # sigma_images_list = []

        flag = False

        for t in range(T):

            with torch.no_grad():
                img = img.to(device)

                target_features = ijepa.target_encoder(img).detach()
                target_features = F.layer_norm(target_features, (target_features.size(-1),))

                # Extract patches from the image
                patches = ijepa.context_encoder.to_patch_embedding(img)
                B, num_patches, _ = patches.size()

                # Extract context features
                context_indices_list = torch.tensor(context_indices_list).to(patches.device)
                context_features = ijepa.context_encoder(patches, patchify=False, mask=context_indices_list)

                # Apply positional embeddings to context features
                pos_emb = ijepa.context_encoder.pos_embedding.to(context_features.device)
                context_pos_embed = torch.stack([pos_emb[context_indices_list[i], :] for i in range(B)])
                context_features += context_pos_embed

                min_target_size = min(min(len(indices) for indices in target_list) for target_list in target_indices_list)
                losses = []
                for j in range(ijepa.num_targets):
                    # Init mask tokens and apply pos encodings
                    target_indices_batch = torch.stack([torch.tensor(target_indices_list[i][j], device=img.device) for i in range(B)])
                    mask_tokens = ijepa.mask_token.expand(B, min_target_size, -1).clone()
                    mask_pos_embed = torch.stack([pos_emb[target_indices_batch[i], :] for i in range(B)])
                    mask_tokens += mask_pos_embed

                    # Concat context_features and mask_tokens, obtain predictions for masked target blocks
                    predictor_input = torch.cat([context_features, mask_tokens], dim=1)
                    predictor_output = ijepa.mu_predictor(predictor_input, patchify=False, pos_embed=False)

                    predictor_output = F.layer_norm(predictor_output, (predictor_output.size(-1),))
                    preds = predictor_output[:, -min_target_size:]

                    sigma_predictor_output = ijepa.sigma_predictor(predictor_input, patchify=False, pos_embed=False)
                    sigma_predictor_output = ijepa.sigma_predictor_head(sigma_predictor_output)
                    sigmas = sigma_predictor_output[:, -min_target_size:]

                    targets = torch.stack([target_features[i, target_indices_list[i][j], :] for i in range(B)])

                    pred_error = ((preds - targets)**2).mean(dim=-1)

                # # visualize
                # img = img.to("cpu")
                # sigmas = sigmas.to("cpu")
                # context_img = reconstruct_img_from_patch(img, context_indices_list)
                # target_img = reconstruct_img_from_patch(img, target_indices_list[0][0])
                # sigma_img = sigma_heatmap(sigmas)

                # context_images_list.append(context_img.squeeze().permute(1,2,0).cpu().numpy())
                # sigma_images_list.append(sigma_img.squeeze().permute(1,2,0).cpu().numpy())


                # select patches which has largest sigma
                index_expanded = context_indices_list.unsqueeze(-1).expand(-1, -1, 1)
                values = torch.full(index_expanded.shape, fill_value=-torch.inf).to(device)
                sigmas.scatter_(dim=1, index=index_expanded, src=values)
                context_indices_list = context_indices_list.tolist()
                for i in range(batch_size):
                    next_index = select_next_block(sigmas[i, :].unsqueeze(0), k=block_size)
                    next_index_not_multiple = [x for x in next_index if x not in context_indices_list[i]]
                    if len(next_index_not_multiple) != block_size*block_size:
                        print(f"t = {t+1} / {T}: there is no block to see without multiple!")
                        flag = True
                        break
                    context_indices_list[i] = context_indices_list[i] + next_index
                if flag:
                    break

        # # # visualize saccade simulation
        # if n % 100 == 0:

        #     row_labels = ['context', 'sigma']      # 左に表示
        #     col_labels = [f't = {i}' for i in range(0, T)]  # 上に表示

        #     fig, axes = plt.subplots(2, T, figsize=(T*2, 4))
        #     for i in range(T):
        #         axes[0, i].imshow(context_images_list[i])
        #         axes[0, i].axis('off')
        #         axes[1, i].imshow(sigma_images_list[i])
        #         axes[1, i].axis('off')
        #         axes[0, i].set_title(col_labels[i], fontsize=10)

        #     for ax, row_label in zip(axes[:,0], row_labels):
        #         ax.set_ylabel(row_label, fontsize=12, rotation=0, labelpad=40, va='center')

        #     plt.tight_layout()
        #     plt.savefig(f"images/saccade/image_{n}.png")
        #     plt.close()

        if torch.is_tensor(context_indices_list):
            context_indices_list = context_indices_list.tolist()
        for i in range(batch_size):
            selected_index_list.append(context_indices_list[i])

    with open(f"selected_patch_index/block_{block_size}_tmax_20_test_.json", "w") as f:
        json.dump(selected_index_list, f)
