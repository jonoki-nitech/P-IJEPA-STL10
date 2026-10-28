"""MC-Dropout uncertainty helpers (shared by the evaluation scripts).

The mu predictor of the MC-Dropout I-JEPA baseline carries dropout as its
only stochastic component; the encoders are deterministic. At inference,
K dropout-enabled forwards of the predictor give per-dim K-sample
statistics of the prediction:

    z_k = LN(mu_predictor(input; dropout mask k)),  k = 1..K   (B, T, D)

    mean_d   : E[z_d]            -> used as the prediction (resid/errors)
    var[d]   : E[z_d^2] - E[z_d]^2   (tiny negative values clipped to 0)
    var_pooled(patch) = mean_d var[d]        (B, T, 1)

The pooled quantity is a per-patch SCALAR kept in VARIANCE units (no sqrt),
shaped exactly like the sigma head's output so the sigma strategies can run
unchanged on top of it.
"""

import torch
import torch.nn.functional as F


def mu_has_dropout(ijepa):
    """True when the mu predictor contains at least one Dropout(p > 0)."""
    return any(isinstance(m, torch.nn.Dropout) and m.p > 0
               for m in ijepa.mu_predictor.modules())


def mc_predict(ijepa, predictor_input, tail_size, mc_samples=1):
    """Prediction + per-patch pooled MC variance from the mu predictor.

    predictor_input: (B, T_in, D) token sequence (as fed to mu_predictor)
    tail_size: predictions are taken from the LAST tail_size output tokens
               (the mask-token positions), as the standard inference does.
    mc_samples: 1 -> deterministic eval forward, variance = None (the
                standard path, result identical to the plain inference);
                K > 1 -> K dropout-enabled forwards (mu predictor only;
                all other modules keep their current mode).

    Returns (preds, var_pooled):
        preds       (B, tail, D)  -- K-sample mean (K = 1: plain forward)
        var_pooled  (B, tail, 1)  -- per-patch mean over dims of the per-dim
                                     K-sample variance (variance units); None
                                     when mc_samples == 1
    """
    if mc_samples <= 1:
        mu_out = ijepa.mu_predictor(
            predictor_input, patchify=False, pos_embed=False)
        mu_out = F.layer_norm(mu_out, (mu_out.size(-1),))
        return mu_out[:, -tail_size:], None

    if not mu_has_dropout(ijepa):
        raise ValueError(
            "mc_samples > 1 requires dropout inside ijepa.mu_predictor "
            "(an MC-Dropout checkpoint, e.g. the mu-dropout=0.1 baseline); "
            "the loaded model has none")

    was_training = ijepa.mu_predictor.training
    ijepa.mu_predictor.train()
    try:
        zs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                z = ijepa.mu_predictor(
                    predictor_input, patchify=False, pos_embed=False)
                zs.append(F.layer_norm(z, (z.size(-1),))[:, -tail_size:])
    finally:
        if not was_training:
            ijepa.mu_predictor.eval()
    z = torch.stack(zs)                       # (K, B, tail, D)
    mean = z.mean(dim=0)                      # E[z]
    var = (z ** 2).mean(dim=0) - mean ** 2    # E[z^2] - E[z]^2
    var_pooled = var.clamp_min(0).mean(dim=-1, keepdim=True)
    return mean, var_pooled
