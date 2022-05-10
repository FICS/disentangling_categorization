# https://github.com/facebookresearch/EGG/blob/master/egg/core/language_analysis.py
from scipy.spatial import distance
from scipy.stats import spearmanr
import editdistance

from .info_theory import *

distances = {
    "edit": lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2),
    "cosine": distance.cosine,
    "hamming": distance.hamming,
    "jaccard": distance.jaccard,
    "euclidean": distance.euclidean,
}
vis_distance_fn = distances.get("cosine")
msg_distance_fn = distances.get("edit")


def topographic_similarity(messages, in_vectors):
    """
    output: rho (topsim), p-value
    """
    message_dist = distance.pdist(messages, msg_distance_fn)
    vis_dist = distance.pdist(in_vectors, vis_distance_fn)
    
    spr = spearmanr(vis_dist, message_dist, nan_policy="raise")
    topsim = spr.correlation
    p = spr.pvalue
    return topsim, p
