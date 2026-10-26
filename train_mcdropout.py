"""Training of a MC-Dropout baseline I-JEPA.

Identical to train.train_ijepa (see train.py / run_training.py), except:

1. The sigma-predictor is NOT trained: the loss is the vanilla I-JEPA MSE
   between the mu-predictor output and the (EMA) target representation
   (weighted_loss / Gaussian NLL in train.py), so no uncertainty part.
2. The mu-predictor carries nn.Dropout(mu_dropout) (timm-style placement:
   attention probabilities + attention output projection + two in the
   feed-forward), so MC-Dropout uncertainty estimates become available at
   inference time (run the predictor with dropout active several times and
   take the prediction variance).
3. The optimizer excludes the sigma-predictor parameters: they stay at their
   random initialization. The sigma-predictor is still INSTANTIATED so the
   resulting training_model.pth keeps the exact state_dict keys of the
   probabilistic model (the evaluation pipeline loads I-JEPA checkpoints
   strictly). Do NOT use the sigma-based strategies with this checkpoint --
   its sigma head is untrained; use MC-Dropout uncertainties instead.

Dropout in every other module (context/target encoder, sigma-predictor, MAE)
is 0.0, i.e. absent in behaviour: the Dropout layers render as identity.
"""
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import multiprocessing
import random
import numpy as np
from tqdm.auto import tqdm
from torchinfo import summary
from data_utils import load_stl10

from simple_ijepa.ijepa import IJEPA
from simple_ijepa.transformer import VisionTransformer

from simple_ijepa.stl10_eval import STL10Eval
from simple_ijepa.utils import training_transforms
from simple_ijepa.dataset import MaskedImageDataset, collate_fn

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# cosine EMA schedule (increase from tau_base to one) as defined in https://arxiv.org/abs/2010.07922
# k -> current training step, K -> maximum number of training steps
def update_gamma(k, K, tau_base):
    k = torch.tensor(k, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)

    tau = 1 - (1 - tau_base) * (torch.cos(torch.pi * k / K) + 1) / 2
    return tau.item()


def train_ijepa_mcdropout(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim = 512
    image_size = 96
    patch_size = 8
    depth = 6
    heads = 6
    mlp_dim = dim * 2
    num_targets = 4
    mu_dropout = getattr(args, "mu_dropout", 0.1)

    encoder = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    ijepa = IJEPA(encoder,
                  hidden_emb_dim=dim,
                  patch_size=patch_size,
                  num_targets=num_targets,
                  mu_dropout=mu_dropout)

    if args.ckpt_path:
        model_state = torch.load(args.ckpt_path, map_location=device)
        ijepa.load_state_dict(model_state)
    ijepa = ijepa.to(device)

    # summary(ijepa, input_size=(2, 3, 96, 96))

    params = (list(ijepa.context_encoder.parameters()) + [ijepa.mask_token] +
              list(ijepa.mu_predictor.parameters()))
    optimizer = torch.optim.Adam(params,
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    stl10_ds = load_stl10(split="unlabeled",
                          transform=training_transforms((image_size, image_size)))
    num_patches = int((image_size // patch_size))**2
    dataset = MaskedImageDataset(stl10_ds,
                                 num_patches=num_patches,
                                 num_targets=num_targets)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=True,
    )

    scaler = GradScaler(enabled=args.fp16_precision and torch.cuda.is_available())

    stl10_eval = STL10Eval()
    total_num_steps = (len(train_loader) *
                       (args.num_epochs + 2)) - args.update_gamma_after_step
    gamma = args.gamma
    global_step = 0
    total_loss = 0.0
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader,
                            desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, (images, context_indices,
                   target_indices_list) in enumerate(progress_bar):
            images = images.to(device)
            with autocast(enabled=args.fp16_precision and torch.cuda.is_available()):
                loss = vanilla_ijepa_loss(ijepa, images, context_indices,
                                          target_indices_list)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (global_step > args.update_gamma_after_step
                    and global_step % args.update_gamma_every_n_steps == 0):
                ijepa.update_params(gamma)
                gamma = update_gamma(global_step, total_num_steps, args.gamma)

            if global_step <= args.update_gamma_after_step:
                ijepa.copy_params()

            total_loss += loss.item()
            epoch_loss += loss.item()
            avg_loss = total_loss / (global_step + 1)
            ep_loss = epoch_loss / (step + 1)

            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.num_epochs} | "
                f"Step {global_step+1} | "
                f"Epoch Loss: {ep_loss:.7f} |"
                f"Total Loss: {avg_loss:.7f} |"
                f"EMA gamma: {gamma:.6f} |"
                f"Lr: {current_lr:.6f}")

            global_step += 1
            if global_step % args.log_every_n_steps == 0:
                torch.save(ijepa.state_dict(),
                           f"{args.save_model_dir}/training_model.pth")
                ijepa.save_encoder(f"{args.save_model_dir}/encoder.pth")

            if global_step % (args.log_every_n_steps * 5) == 0:
                stl10_eval.evaluate(ijepa)
                print("!" * 100)


def vanilla_ijepa_loss(ijepa, img, context_indices_list, target_indices_list):
    """Vanilla I-JEPA loss (train.py's forward without the sigma part).

    Identical forward for the target/context/mask-token path, but the loss
    is the plain MSE between the mu-predicted features and the target-
    encoder representations of each target block. The sigma-predictor is
    neither run nor used (its weights remain at their initialization).
    """
    # Target model processes all patches
    with torch.no_grad():
        target_features = ijepa.target_encoder(img).detach()
        target_features = F.layer_norm(target_features,
                                       (target_features.size(-1),))

    # Extract patches from the image
    patches = ijepa.context_encoder.to_patch_embedding(img)
    B, num_patches, _ = patches.size()

    # Extract context features
    context_indices_list = torch.tensor(context_indices_list).to(patches.device)
    context_features = ijepa.context_encoder(patches, patchify=False,
                                             mask=context_indices_list)

    # Apply positional embeddings to context features
    pos_emb = ijepa.context_encoder.pos_embedding.to(context_features.device)
    context_pos_embed = torch.stack([pos_emb[context_indices_list[i], :]
                                     for i in range(B)])
    context_features += context_pos_embed

    min_target_size = min(min(len(indices) for indices in target_list)
                          for target_list in target_indices_list)
    losses = []
    for j in range(ijepa.num_targets):
        # Init mask tokens and apply pos encodings
        target_indices_batch = torch.stack(
            [torch.tensor(target_indices_list[i][j], device=img.device)
             for i in range(B)])
        mask_tokens = ijepa.mask_token.expand(B, min_target_size, -1).clone()
        mask_pos_embed = torch.stack([pos_emb[target_indices_batch[i], :]
                                      for i in range(B)])
        mask_tokens += mask_pos_embed

        # Concat context_features and mask_tokens, obtain predictions for
        # masked target blocks
        predictor_input = torch.cat([context_features, mask_tokens], dim=1)
        predictor_output = ijepa.mu_predictor(predictor_input, patchify=False,
                                              pos_embed=False)

        predictor_output = F.layer_norm(predictor_output,
                                        (predictor_output.size(-1),))
        preds = predictor_output[:, -min_target_size:]

        targets = torch.stack(
            [target_features[i, target_indices_list[i][j], :]
             for i in range(B)])
        losses.append(F.mse_loss(preds, targets))

    total_loss = torch.mean(torch.stack(losses))
    return total_loss
