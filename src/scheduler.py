import logging
import torch
import time
import torch.fx as fx

# questions
# recomputation_policy is recomps a dictionary of statistical stats for nodes?

class Scheduler:
    """
    The scheduler class of the mu-Two algorithim
    """
    def __init__(self, candidates: list,
                 statistical_stats: dict[fx.Node, dict]):
        """
        Initializing the scheduler with the following parameter attributes
        :param candidates: list
            The list of intermediate nodes in the graphprofiler for either recompuation or swapping
        :param statistical_stats: dict
            A dictionary of the run_time and memory statistics for the nodes in the graphprofiler.
        """
        self.candidates = candidates
        self.statistical_stats = statistical_stats

    def max_recomp_candidate(self, candidate_set: tuple):
        """
        :paran candidate_set: tuple
        """
        pass

    def recomputation_policy(self, candidate_set: tuple,
                             mem_limit: int,
                             max_peak_memory: int,
                             initialization: tuple,
                             recomps: list):
        """
        :param candidate_set: tuple
        :param mem_limit:
        :param max_peak_memory:
        :param initialization:
        :param recomps: list
            The list of intermediate nodes for recomputation
        """
        while candidate_set != 0:
            r_cand = self.max_recomp_candidate(candidate_set=candidate_set)
            recomps.append(r_cand)
            candidate = r_cand
            self.candidates.remove(candidate)