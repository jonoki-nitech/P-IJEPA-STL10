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
import os
from tqdm import tqdm

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

def pred_error_heatmap(pred_error, patch_index=None, patch_size=8):

    h = 96//patch_size
    w = 96//patch_size

    unpatchfy = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                        p1=1,
                        p2=1,
                        h=h, w=w)
    
    pred_error = unpatchfy(pred_error)

    pred_error = F.interpolate(pred_error, size=(96, 96))

    pred_error = to_color_tensor(pred_error.squeeze(), x_min=0.0, x_max=1.0)

    return pred_error.unsqueeze(0)

def surprise_map_heatmap(surprise_map, patch_index=None, patch_size=8):

    h = 96//patch_size
    w = 96//patch_size

    unpatchfy = Rearrange("(b c h w) -> b c h w",
                        b=1, c=1, 
                        h=h, w=w)
    
    surprise_map = unpatchfy(surprise_map)

    surprise_map = F.interpolate(surprise_map, size=(96, 96))

    surprise_map = to_color_tensor(surprise_map.squeeze(), x_min=torch.min(surprise_map), x_max=torch.max(surprise_map))

    return surprise_map.unsqueeze(0)

def to_color_tensor(x, cmap='viridis', x_min=0.25, x_max=0.55):
    # x: (H, W)
    # x_min, x_max = x.min(), x.max()
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
    stl10_ds = load_stl10(split="test",
                          transform=transforms.ToTensor())

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

    N = 30
    T = 40
    block_size = 3
    Top = Left = 96 // 8 - block_size + 1

    for n in tqdm(range(N)):
        img, label = stl10_ds[n]
        img = img.unsqueeze(0)

        all_index = range(144)

        context_images_list = []
        sigma_images_list = []
        surprise_map_images_list = []
        surprise_map = torch.zeros(144)
        surprise_map_count = torch.zeros(144)

        # init sigma map
        for top in range(Top):
            for left in range(Left):
                context_indices_list = [[]]# [block_to_index(4, 4, 4, 4)]# +block_to_index(8, 4, 2, 2)]
                # t = [x for x in all_index if x not in context_indices_list[0]]
                target_indices_list = [[all_index]]

                for t in range(T):

                    with torch.no_grad():
                        img = img.to(device)

                        target_features = ijepa.target_encoder(img).detach()
                        target_features = F.layer_norm(target_features, (target_features.size(-1),))

                        # Extract patches from the image
                        patches = ijepa.context_encoder.to_patch_embedding(img)
                        B, num_patches, _ = patches.size()

                        pos_emb = ijepa.context_encoder.pos_embedding.to(device)
                        context_indices_list = torch.tensor(context_indices_list).to(patches.device)

                        if t != 0:
                            # Extract context features
                            context_features = ijepa.context_encoder(patches, patchify=False, mask=context_indices_list)

                            # Apply positional embeddings to context features
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
                            if t == 0:
                                predictor_input = mask_tokens
                            else:
                                predictor_input = torch.cat([context_features, mask_tokens], dim=1)
                            predictor_output = ijepa.mu_predictor(predictor_input, patchify=False, pos_embed=False)

                            predictor_output = F.layer_norm(predictor_output, (predictor_output.size(-1),))
                            preds = predictor_output[:, -min_target_size:]

                            sigma_predictor_output = ijepa.sigma_predictor(predictor_input, patchify=False, pos_embed=False)
                            sigma_predictor_output = ijepa.sigma_predictor_head(sigma_predictor_output)
                            sigmas = sigma_predictor_output[:, -min_target_size:]

                            targets = torch.stack([target_features[i, target_indices_list[i][j], :] for i in range(B)])

                            pred_error = ((preds - targets)**2).mean(dim=-1, keepdim=True) / sigmas

                        # visualize
                        img = img.to("cpu")
                        sigmas = sigmas.to("cpu")
                        pred_error = pred_error.to("cpu")
                        # context_img = reconstruct_img_from_patch(img, context_indices_list)
                        # target_img = reconstruct_img_from_patch(img, target_indices_list[0][0])
                        # sigma_img = sigma_heatmap(sigmas)
                        # surprise_map_img = surprise_map_heatmap(surprise_map)
            
                        # context_images_list.append(context_img.squeeze().permute(1,2,0).cpu().numpy())
                        # sigma_images_list.append(sigma_img.squeeze().permute(1,2,0).cpu().numpy())
                        # surprise_map_images_list.append(surprise_map_img.squeeze().permute(1,2,0).cpu().numpy())

                        # select patches which has largest sigma(4*4block)
                        next_index = select_next_block(sigmas, k=block_size)
                        if t == 0:
                            next_index = block_to_index(left, top, block_size, block_size)
                        next_index_not_multiple = [x for x in next_index if x not in context_indices_list[0]] 
                        context_indices_list = context_indices_list.tolist()
                        context_indices_list[0] = context_indices_list[0] + next_index

                        # add pred_error / sigma of next block to surprise map
                        if len(next_index_not_multiple) != 0 and t < T-1 and t > 0:
                            surprise_map[torch.tensor(next_index_not_multiple)] += pred_error[0, torch.tensor(next_index_not_multiple), 0]
                            surprise_map_count[torch.tensor(next_index_not_multiple)] += 1

        # visualize saccade simulation and surprise map

        if surprise_map_count.min() == 0:
            print(f"! @ n={n}")
            surprise_map_count[surprise_map_count==0] = 10000

        surprise_map_mean = surprise_map / surprise_map_count 

        os.makedirs(f"images/surprise_map_mean/{n}", exist_ok=True)

        surprise_map_img = surprise_map_heatmap(surprise_map_mean)

        fig, axes = plt.subplots(1, 2, figsize=(4, 2))

        def show_no_border(ax, img, label):
            ax.imshow(img)
            for spine in ax.spines.values():
                spine.set_visible(False)   # ← 黒枠を消す
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(label)

        show_no_border(axes[0], img.squeeze().permute(1,2,0).cpu().numpy(), "original")
        show_no_border(axes[1], surprise_map_img.squeeze().permute(1,2,0).cpu().numpy(), "surprise map mean")

        plt.tight_layout()
        plt.savefig(f"images/surprise_map_mean/{n}/surprise_map_mean.png")
        plt.close()