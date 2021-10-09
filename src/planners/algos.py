import torch
import numpy as np
import networkx
from networkx.algorithms.approximation import vertex_cover
import itertools


def mc_bfs(start, goal, anchor, forward_model, action_set, max_depth, margin, early_stop, batch_size, device):
    """
    Memory-constrained breadth first search for normalized embeddings.
    """
    # assume all embeddings are normalized and use cosine similarity instead of L2 distance
    min_similarity = 1 - (margin**2)/8  # min_similarity for being reideintified
    start = start.cpu().detach().numpy()
    goal = goal.cpu().detach().numpy()

    best_action_sequence = []
    best_similarity = (start @ goal.T).item()  # similarity of the most promising node to the goal
    visited = start  # tensor of visited nodes, unsorted
    fringe = start  # tensor concatenating embeddings of all nodes on the fringe of the tree growth
    fringe_actions = [[]]  # list of action sequences required to reach each node in the fringe

    for _ in range(max_depth):
        if early_stop and best_similarity > min_similarity:
            # early stop returns the first action sequence that gets close enough to the goal
            return best_action_sequence

        # resampling step to keep fringe size bounded
        if len(fringe) > batch_size:
            indices = np.random.choice(fringe.shape[0], batch_size)
            fringe, fringe_actions = fringe[indices], [fringe_actions[i] for i in indices]

        # grow fringe with forward model (fringe size becomes 5x larger) and action sequences list
        children = np.concatenate([forward_model.forward(torch.tensor(fringe, device=device),
                                                         torch.cat([anchor] * fringe.shape[0], dim=0),
                                                         torch.zeros(size=(fringe.shape[0], ),dtype=torch.long, device=device)
                                                         .fill_(a)).cpu().detach().numpy() for a in action_set])
        children_actions = [[action_seq + [a] for action_seq in fringe_actions] for a in action_set]
        children_actions = list(itertools.chain.from_iterable(children_actions))
        children_sim = (children @ goal.T).reshape(-1)

        # we now sort all children by their distance to the goal. This allows us to do some sort of hashing.
        sort_indices = np.argsort(children_sim)
        children = children[sort_indices]
        children_sim = children_sim[sort_indices]
        children_actions = [children_actions[idx] for idx in sort_indices]
        if children_sim[-1] > best_similarity: # update most promising node
            best_similarity = children_sim[-1]
            best_action_sequence = children_actions[-1]

        # aggregate fringe and visited set, find minimum weight vertex cover with vey high weights for visited nodes
        # all nodes in the solution are removed from the fringe; the fringe is added to visited
        all_nodes = np.concatenate((visited, children))
        similarities = (all_nodes @ all_nodes.T) > min_similarity
        np.fill_diagonal(similarities, False)
        G = networkx.convert_matrix.from_numpy_matrix(similarities)
        networkx.set_node_attributes(G, {i: 1e6 if i < visited.shape[0] else 1e-6 for i in range(all_nodes.shape[0])}, 'weight')
        res = vertex_cover.min_weighted_vertex_cover(G, 'weight')
        to_keep = [i for i in range(len(children)) if i+len(visited) not in res]
        fringe = children[to_keep]
        fringe_actions = [children_actions[i] for i in to_keep]
        visited = np.concatenate((visited, fringe))

        if len(fringe) <= 0:  # if the fringe is empty, the tree was fully explored and the goal was not reached
            return best_action_sequence
    return best_action_sequence


def flat_mc_bfs(start, goal, anchor, forward_model, action_set, max_depth, margin, early_stop, batch_size, device):
    """
    Memory-constrained breadth first search for unnormalized embeddings.
    """

    def dist(a, b):
        return np.sum((np.expand_dims(a, 0) - np.expand_dims(b, 1))**2, axis=-1)

    # assume all embeddings are normalized and use cosine similarity instead of L2 distance
    max_dist = margin/2  # min_similarity for being reideintified
    start = start.cpu().detach().numpy()
    goal = goal.cpu().detach().numpy()

    best_action_sequence = []
    best_dist = dist(start, goal).item()  # similarity of the most promising node to the goal
    visited = start  # tensor of visited nodes, unsorted
    fringe = start  # tensor concatenating embeddings of all nodes on the fringe of the tree growth
    fringe_actions = [[]]  # list of action sequences required to reach each node in the fringe

    for _ in range(max_depth):

        if early_stop and best_dist < max_dist:
            # early stop returns the first action sequence that gets close enough to the goal
            return best_action_sequence

        # resampling step to keep fringe size bounded
        if len(fringe) > batch_size:
            indices = np.random.choice(fringe.shape[0], batch_size)
            fringe, fringe_actions = fringe[indices], [fringe_actions[i] for i in indices]

        # grow fringe with forward model (fringe size becomes 5x larger) and action sequences list
        children = np.concatenate([forward_model.forward(torch.tensor(fringe, device=device),
                                                         torch.cat([anchor] * fringe.shape[0], dim=0),
                                                         torch.zeros(size=(fringe.shape[0], ),dtype=torch.long, device=device)
                                                         .fill_(a)).cpu().detach().numpy() for a in action_set])
        children_actions = [[action_seq + [a] for action_seq in fringe_actions] for a in action_set]
        children_actions = list(itertools.chain.from_iterable(children_actions))
        children_dist = dist(children, goal).reshape(-1)

        # we now sort all children by their distance to the goal. This allows us to do some sort of hashing.
        sort_indices = np.flip(np.argsort(children_dist))
        children = children[sort_indices]
        children_dist = children_dist[sort_indices]
        children_actions = [children_actions[idx] for idx in sort_indices]
        if children_dist[-1] > max_dist: # update most promising node
            best_dist = children_dist[-1]
            best_action_sequence = children_actions[-1]

        # aggregate fringe and visited set, find minimum weight vertex cover with vey high weights for visited nodes
        # all nodes in the solution are removed from the fringe; the fringe is added to visited
        all_nodes = np.concatenate((visited, children))
        similarities = dist(all_nodes, all_nodes) < max_dist
        np.fill_diagonal(similarities, False)
        G = networkx.convert_matrix.from_numpy_matrix(similarities)
        networkx.set_node_attributes(G, {i: 1e6 if i < visited.shape[0] else 1e-6 for i in range(all_nodes.shape[0])}, 'weight')
        res = vertex_cover.min_weighted_vertex_cover(G, 'weight')
        to_keep = [i for i in range(len(children)) if i+len(visited) not in res]
        fringe = children[to_keep]
        fringe_actions = [children_actions[i] for i in to_keep]
        visited = np.concatenate((visited, fringe))

        if len(fringe) <= 0:  # if the fringe is empty, the tree was fully explored and the goal was not reached
            return best_action_sequence
    return best_action_sequence


def selective_mc_bfs(start, goal, forbidden, anchor, forward_model, action_set, max_depth, margin, early_stop, batch_size, snap, device):
    """
    Given a start and a target in latent space, performs a DFS by growing the action/state tree with the forward
    model.
    """
    min_similarity = 1 - (margin**2)/8  # min_similarity for being reideintified
    start = start.cpu().detach().numpy()
    goal = goal.cpu().detach().numpy()

    best_action_sequence = []
    best_similarity = (start @ goal.T).item()
    if forbidden is not None:
        best_similarity = best_similarity - np.sum(start @ forbidden.T > min_similarity)  # similarity of the most promising node to the goal
    # also p
    visited = start  # tensor of visited nodes, unsorted
    fringe = start  # tensor concatenating embeddings of all nodes on the fringe of the tree growth
    fringe_actions = [[]]  # list of action sequences required to reach each node in the fringe

    for _ in range(max_depth):
        if early_stop and best_similarity > min_similarity:
            return best_action_sequence

        if len(fringe) > batch_size:
            indices = np.random.choice(fringe.shape[0], batch_size)
            fringe, fringe_actions = fringe[indices], [fringe_actions[i] for i in indices]

        children = np.concatenate([forward_model.forward(torch.tensor(fringe, device=device),
                                                         torch.cat([anchor] * fringe.shape[0], dim=0),
                                                         torch.zeros(size=(fringe.shape[0], ),dtype=torch.long, device=device)
                                                         .fill_(a)).cpu().detach().numpy() for a in action_set])
        children_actions = [[action_seq + [a] for action_seq in fringe_actions] for a in action_set]
        children_actions = list(itertools.chain.from_iterable(children_actions))

        if snap:
            close_enough = np.any(children @ forbidden.T > min_similarity, axis=-1, keepdims=True)
            closest_forbidden = (children @ forbidden.T > min_similarity) * \
                                (children @ forbidden.T == np.amax(children @ forbidden.T, axis=-1, keepdims=True))
            closest_forbidden = np.stack([closest_forbidden]*children.shape[1], axis=-1) * forbidden[np.newaxis, ...]
            closest_forbidden = np.sum(closest_forbidden, axis=1)
            children = np.where(close_enough, closest_forbidden, children)

        children_sim = (children @ goal.T).reshape(-1)
        if forbidden is not None:
            children_sim -= np.sum(children @ forbidden.T > min_similarity, axis=-1)  # negative similarity if already visited

        sort_indices = np.argsort(children_sim)
        children = children[sort_indices]
        children_sim = children_sim[sort_indices]
        children_actions = [children_actions[idx] for idx in sort_indices]
        if children_sim[-1] > best_similarity: # update most promising node
            best_similarity = children_sim[-1]
            best_action_sequence = children_actions[-1]

        all_nodes = np.concatenate((visited, children))
        similarities = (all_nodes @ all_nodes.T) > min_similarity
        np.fill_diagonal(similarities, False)
        G = networkx.convert_matrix.from_numpy_matrix(similarities)
        networkx.set_node_attributes(G, {i: 1e6 if i < visited.shape[0] else 1e-6 for i in range(all_nodes.shape[0])}, 'weight')
        res = vertex_cover.min_weighted_vertex_cover(G, 'weight')
        to_keep = [i for i in range(len(children)) if i+len(visited) not in res]
        fringe = children[to_keep]
        fringe_actions = [children_actions[i] for i in to_keep]
        visited = np.concatenate((visited, fringe))

        if len(fringe) <= 0:  # if the fringe is empty, the tree was fully explored and the goal was not reached
            return best_action_sequence
    return best_action_sequence if best_similarity > 0 else []  # if the best target has already been visited, give up





def get_latent_trajectory(start, anchor, action_seq, forward_model, device):
    s = [start]
    for a in action_seq:
        s.append(forward_model.forward(s[-1], anchor, torch.tensor([a], dtype=torch.long, device=device)))
    return s



