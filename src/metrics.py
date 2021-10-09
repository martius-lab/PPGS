import torch


def embedding_accuracy(y_pred, y_true, margin):
    """
    Calculates the fraction of pairs of embeddings from (y_true, y_pred) that are within a certain distance of each
    other. Note that this distance must be half of the margin: if it was more, a point landing closer to the target than
    to the start could still be identified as the start.
    """
    d = torch.nn.PairwiseDistance(2)
    return d(y_pred, y_true).le(margin*0.5).float().mean()  # margin is halved for recognizing states!


def multiclass_accuracy(y_pred, y_true):
    """
    Simple multiclass accuracy implementation.
    """
    return torch.argmax(y_pred, dim=-1).reshape(-1).eq(y_true.reshape(-1)).float().mean()


def fraction_of_large_transitions(start, true_target, pred_target, margin):
    """
    Computes the fraction of predicted latent transitions that are larger than half of the margin:
    (1) among all transitions
    (2) among transitions that correspond to identical start/target states (loops)
    (3) among transitions that correspond to identical start/target states
    """
    diff = start - true_target
    loops = 1.0 - (diff.view(diff.shape[0], -1).abs() > 1e-6).any(dim=1).float()
    long = (torch.norm(pred_target - start, dim=1) > (margin/2)).float()
    return long.mean(), (long * loops).sum() / (loops.sum() + 1e-6), \
        (long * (1.0 - loops)).sum() / ((1.0 - loops).sum() + 1e-6)


def adjusted_forward_loss(emb_start, emb_target, pred_target):
    """
    Computes the ration between the mean L2 errors of the forward model and the mean encoder distance between two states
    """
    d = torch.nn.PairwiseDistance(2)
    return d(emb_target, pred_target).mean() / d(emb_start, emb_target).mean()
