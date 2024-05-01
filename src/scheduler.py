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
    def __init__(self,
                 statistical_stats: dict[str, dict]):
        """
        Initializing the scheduler with the following parameter attributes
        :param candidates: list
            The list of intermediate nodes in the graphprofiler for either recompuation or swapping. This will also include
            the activation nodes as well
        :param statistical_stats: dict
            A dictionary of the run_time and memory statistics for the nodes in the graphprofiler.
        """
        self.statistical_stats = statistical_stats
        self.candidates, self.ranks = self._get_candidates(stats=statistical_stats)
        self.recomps = []
        self.candidates_dict[fx.Node, dict] = {}
        self.rank_mapping[int, str] = {statistical_stats[k]: k for k in statistical_stats.keys()}

    def _get_candidates(self, stats: dict[str, dict]) -> tuple[list, list]:
        """
        Get the candidates for swapping or recompuation of the graph. This should be the intermediate nodes of the graphprofiler
        :param stats: dict
            dictionary with the key being the node name, and a dictionary with the following values:
                'forward_pass'
                'category'
                'rank'
                'run_time'
                'active_memory'
                'peak_memory'
                'forward_swap_time'
                'inactive_start_time'
                'backward_start_time'
                'inactive_end_time'
                'inactive_end_time'
        return:
        candidates: tuple[list, list]
            The candidate rank number, the candidate name
        """
        candidates = []
        ranks = []
        # iterate over the node keys
        for node in stats.keys():
            if stats[node]['category'] == 'intermediate':
                candidates.append(node)
                ranks.append(stats[node]['rank'])
        return candidates, ranks

    def _calc_recomp_ratio(self, candidate: str) -> float:
        """
        Calculate the recompuation ratio for a node
        :param candidate: str
            the candidate to calculate the recompuation ratio for
        """
        # get the rank number
        rank = self.statistical_stats[candidate]['rank']
        # get the total recomputetime from all subsequent nodes
        count = rank
        total_runtime = 0
        while count > 0:
            x = self.rank_mapping[count]
            total_runtime += self.statistical_stats[x]['run_time']
        return self.statistical_stats[candidate]['peak_memory'] / total_runtime

    def max_recomp_candidate(self) -> str:
        """
        function gets the candidate with the largest recompute_ratio
        :paran candidate_set: tuple
        """
        candidate = None
        recompute_ratio = 0
        # iterate over the candidates to find the candidate with the highest recomputation ratio
        for cand in self.candidates:
            if candidate is None:
                candidate = cand
            # calculate the recomp_ratio
            compare_ratio = self._calc_recomp_ratio(cand)
            if recompute_ratio < compare_ratio:
                candidate = cand
                recompute_ratio = compare_ratio
        return candidate

    def _update_recomps(self, recomps: list, candidate: tuple):
        """
        :param recomps: list
            The list of intermediate nodes for recomputation
        :param candidate: tuple
        """
        pass

    def _update_candidates(self, t, recomp_cnt, candidates):
        """
        :param t:
        :param recomp_cnt:
        :param candidates:
        """
        for candidiate in candidates:
            if t in self.candidates_dict[candidiate]['recomp_src']:
                self.candidates_dict[candidiate]['recomp_src'].remove(t)
                self.candidates_dict[candidiate]['recomp_src'].extend(self.candidates_dict[t]['recomp_src'])
                self.candidates_dict[candidiate]['recomp_time'] += self._get_recomp_time(recomp_src=self.candidates_dict[t]['recomp_src'])
                for rp in recomps:
                    if candidiate in self.candidates_dict[rp]['recomp_src']:
                        self.candidates_dict[candidiate]['recomp_total_time'] += self.candidates_dict[candidiate]['recomp_time']
            if candidiate in self.candidates_dict[t]['recomp_src']:
                self.candidates_dict[candidiate]['recomp_total_time'] = recomp_cnt * self.candidates_dict[candidiate]['recomp_time']
            self.update_recompute_ratio(candidates=candidates)

    def compute_memory(self, node_tensor: torch.Tensor):
        """
        :param node_tensor: torch.Tensor
        """
        # get the current usage in gpu memory
        try:
            torch_bytes = torch.numel(node_tensor) * torch.element_size(node_tensor)
        except Exception as e:
            try:
                torch_bytes = torch.Tensor.nelement(node_tensor) * torch.Tensor.element_size(node_tensor)
            except Exception as e:
                logging.error(f"Error in computing memory for the tensor: {e}")
                torch_bytes = 0
        return torch_bytes

    def recomputation_policy(self,
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
        mem_consumption = max_peak_memory
        while len(self.candidates) != 0:
            r_cand = self.max_recomp_candidate(candidate_set=self.candidates)
            self.recomps.append(r_cand)
            candidate = r_cand
            self.candidates.remove(candidate)
            recomp_cnt = self._update_recomps(recomps=self.recomps, candidate=candidate)
            self._update_candidates(candidate=candidate, recomp_cnt=recomp_cnt, candidates=self.candidates)
            mem_consumption -= self.compute_memory(node_tensor=candidate)
            if (mem_consumption - mem_limit) <= 0 :
                break


    def _get_recomp_src(self, candidate: tuple):
        """
        :param candidate: tuple
        """
        pass

    def _get_recomp_time(self, recomp_src: list):
        """
        :param recomp_src: list
        """
        recomp_time = 0
        for n in recomp_src:
            recomp_time += self.statistical_stats[n]['run_time']

        return recomp_time


    def initialization(self, candidates: list) -> None:
        """
        :param candidates: list
            The list of intermediate nodes in the graphprofiler for either recompuation or swapping
        """
        for candidate in candidates:
            recomp_src = self._get_recomp_src(candidate=candidate)
            recomp_time = self._get_recomp_time(recomp_src=self._get_recomp_src(candidate=candidate))
            mem_consumption = self.compute_memory(node_tensor=candidate)
            self.candidates_dict[candidate] = {'recomp_src': recomp_src,
                                               'recomp_time': recomp_time,
                                               'recomp_total_time': recomp_time,
                                               'recompute_ratio': mem_consumption/recomp_time
                                               }

    def max_candidate(self, candidates: list):
        """
        :param candidates: list
        """
        max_candidate = None
        for candidate in candidates:
            if max_candidate is None:
                max_candidate = candidate
            elif self.candidates_dict[candidate]['recompute_ratio'] < self.candidates_dict[max_candidate]['recompute_ratio']:
                max_candidate = candidate

        return max_candidate


    def update_recompute_ratio(self, candidates: list) -> None:
        """
        :param candidates: tuple
        """
        for candidate in candidates:
            candidate_memory = self.compute_memory(node_tensor=candidate)
            self.candidates_dict[candidate]['recompute_ratio'] = candidate_memory/self.candidates_dict[candidate]['recomp_time']

    def updating_existing_recomputations(self, recomps: list, candidate: tuple):
        """
        :param recomps: list
            The list of intermediate nodes for recomputation
        :param candidate: tuple
        """
        recomp_cnt = 1
        for rp in recomps:
            if candidate in self.candidates_dict[rp]['recomp_src']:
                recomp_cnt += 1
                self.candidates_dict[rp]['recomp_src'].remove(candidate)
                self.candidates_dict[rp]['recomp_src'].extend(self.candidates_dict[candidate]['recomp_src'])
                self.candidates_dict[rp]['recomp_time'] += self._get_recomp_time(recomp_src=self.candidates_dict[candidate]['recomp_src'])

        return recomp_cnt