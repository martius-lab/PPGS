import torch


def hinge_loss(start, target, pred_target, margin, hinge_params, device):
    """
    Loss pushing pairs of embeddings to be at an L2 distance of at least margin.
    hinge_params.forward engine additionally applied the loss on the output of the forward model. This is widely
        regarded as a bad idea. If combined with hinge_params.hinge_cutoff, pairs for which the distance between
        the two embeddings produced by the encoder is too small are skipped.
    hinge_params.loss controls the functional shape of the loss function
    """
    enc_dist = torch.nn.PairwiseDistance()(start, target)
    dist = enc_dist

    if hinge_params.forward_hinge:
        forward_dist = torch.nn.PairwiseDistance()(start, pred_target)
        forward_dist = torch.where(enc_dist < hinge_params.hinge_cutoff, torch.tensor(margin, device=device), forward_dist)
        dist = torch.cat([enc_dist, forward_dist], dim=0)

    if hinge_params.loss == "peaked":
        arg_peak = hinge_params.arg_peak
        p_loss = torch.where(dist <= arg_peak, dist/arg_peak, 1-((dist-arg_peak)/(margin-arg_peak)))
        return torch.mean(torch.max(p_loss, other=torch.zeros_like(dist, device=device)))
    if hinge_params.loss == "hinge":
        return torch.mean(torch.max(margin - dist, other=torch.zeros_like(dist, device=device)))
    if hinge_params.loss == "quadratic":
        return torch.mean(torch.max(1 - ((dist ** 2) / (margin ** 2)), other=torch.zeros_like(dist, device=device)))

    raise Exception('Unrecognized hinge loss.')
