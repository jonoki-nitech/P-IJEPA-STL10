from simple_ijepa.transformer import  VisionTransformer
from simple_ijepa.ijepa import IJEPA
import torch
from data_utils import load_stl10
from torchvision import transforms
from einops.layers.torch import Rearrange
from torchvision.utils import save_image
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
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

    if patch_index is not None:
        sigma[0, patch_index] = 0

    unpatchfy = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                        p1=1,
                        p2=1,
                        h=h, w=w)

    sigma = unpatchfy(sigma)

    sigma = torch.nn.functional.interpolate(sigma, size=(96, 96))

    sigma = to_color_tensor(sigma.squeeze())

    return sigma.unsqueeze(0)

def to_color_tensor(x, cmap='viridis'):
    # x: (H, W)
    # x_min, x_max = x.min(), x.max()
    x_min, x_max=0.25, 0.55
    x_norm = (x - x_min) / (x_max - x_min)
    colormap = (matplotlib.colormaps[cmap]
                if hasattr(matplotlib, "colormaps")
                else cm.get_cmap(cmap))
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

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    sigmas_list = torch.zeros([N, 144])
    pred_error_list = torch.zeros_like(sigmas_list)

    os.makedirs("images/uncertainty", exist_ok=True)

    for n in range(N):
        n = n# + 100
        img, label = stl10_ds[n]
        img = img.unsqueeze(0).to(device)

        all_index = range(144)

        context_indices_list = [block_to_index(4, 4, 4, 4)]# +block_to_index(8, 4, 2, 2)]
        observed_patch_index = context_indices_list[0]
        # t = [x for x in all_index if x not in context_indices_list[0]]
        target_indices_list = [[all_index]]

        with torch.no_grad():
            target_features = ijepa.target_encoder(img).detach()
            target_features = torch.nn.functional.layer_norm(target_features, (target_features.size(-1),))

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

                predictor_output = torch.nn.functional.layer_norm(predictor_output, (predictor_output.size(-1),))
                preds = predictor_output[:, -min_target_size:]

                sigma_predictor_output = ijepa.sigma_predictor(predictor_input, patchify=False, pos_embed=False)
                sigma_predictor_output = ijepa.sigma_predictor_head(sigma_predictor_output)
                sigmas = sigma_predictor_output[:, -min_target_size:]

                targets = torch.stack([target_features[i, target_indices_list[i][j], :] for i in range(B)])

                pred_error = ((preds - targets)**2).mean(dim=-1)

            sigmas_list[n] = sigmas.squeeze()
            pred_error_list[n] = pred_error.squeeze()

            # visualize
            img = img.to("cpu")
            sigmas = sigmas.to("cpu")
            context_img = reconstruct_img_from_patch(img, context_indices_list)
            target_index = [x for x in target_indices_list[0][0] if x not in context_indices_list]
            target_img = reconstruct_img_from_patch(img, target_index)
            # sigma_img = sigma_heatmap(sigmas)
            sigma_img = sigma_heatmap(sigmas, patch_index=context_indices_list)

            # alpha = 0.3
            # overlay_tensor = (1 - alpha) * img + alpha * sigma_img

            imgs = torch.cat([img, context_img, target_img, sigma_img], dim=0)
            save_image(imgs, f'images/uncertainty/images_{n}.png', nrow=4)

            # plot sigmas and pred error


    # 散布図は未観測パッチのみを対象にする (観測済みパッチは除外)
    unobserved_mask = [x for x in range(144) if x not in observed_patch_index]
    sigmas_unobs = sigmas_list[:, unobserved_mask].flatten().cpu().numpy()
    pred_error_unobs = pred_error_list[:, unobserved_mask].flatten().cpu().numpy()

    corr = np.corrcoef(sigmas_unobs, pred_error_unobs)[0, 1]
    plt.scatter(sigmas_unobs, pred_error_unobs, alpha=0.1)
    plt.xlabel("uncertainty")
    plt.ylabel("prediction error")
    plt.title(f"corr = {corr}")
    plt.savefig("images/uncertainty/sigma_pred_error.png")
    plt.close()
