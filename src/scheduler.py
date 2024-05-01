import logging
import torch
import time
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
import torch.fx as fx

# questions
# recomputation_policy is recomps a dictionary of statistical stats for nodes?


class Scheduler:
    """
    The scheduler class of the mu-Two algorithim
    """
    def __init__(self,
                 gm: torch.fx.GraphModule,
                 statistical_stats: dict[str, dict],
                 node_pass_usage: dict[fx.Node, dict]):
        """
        Initializing the scheduler with the following parameter attributes
        :param candidates: list
            The list of intermediate nodes in the graphprofiler for either recompuation or swapping. This will also include
            the activation nodes as well
        :param statistical_stats: dict
            A dictionary of the run_time and memory statistics for the nodes in the graphprofiler.
        """
        self.graph_module = gm
        self.graph_module = self.remove_detach_nodes(gm=self.graph_module)
        self.statistical_stats = statistical_stats
        self.candidates, self.ranks = self._get_candidates(stats=statistical_stats)
        self.node_pass_usage = node_pass_usage
        self.recomps = []
        self.candidates_dict[str, dict] = {}
        self.node_name_mapping = self.get_name_to_node_map()
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

    def _update_recomps(self, candidate: any):
        """
        :param candidate: tuple
        """
        recomp_cnt = 1
        for rp in self.recomps:
            if candidate in self.candidates_dict[rp]['recomp_src']:
                recomp_cnt += 1
                self.candidates_dict[rp]['recomp_src'].remove(candidate)
                self.candidates_dict[rp]['recomp_src'].extend(self.candidates_dict[candidate]['recomp_src'])
                self.candidates_dict[rp]['recomp_time'] += self._get_recomp_time(recomp_src=self.candidates_dict[candidate]['recomp_src'])

        return recomp_cnt

    def _update_candidates(self, candidate: str, recomp_cnt: int):
        """
        :param t:
        :param recomp_cnt:
        :param candidates:
        """
        for cand in self.candidates:
            if candidate in self.candidates_dict[cand]['recomp_src']:
                self.candidates_dict[cand]['recomp_src'].remove(candidate)
                self.candidates_dict[cand]['recomp_src'].extend(self.candidates_dict[candidate]['recomp_src'])
                self.candidates_dict[cand]['recomp_time'] += self._get_recomp_time(recomp_src=self.candidates_dict[candidate]['recomp_src'])
                for rp in self.recomps:
                    if cand in self.candidates_dict[rp]['recomp_src']:
                        self.candidates_dict[cand]['recomp_total_time'] += self.candidates_dict[cand]['recomp_time']
            if cand in self.candidates_dict[candidate]['recomp_src']:
                self.candidates_dict[cand]['recomp_total_time'] = recomp_cnt * self.candidates_dict[cand]['recomp_time']
            self.update_recompute_ratio(candidates=cand)

    def get_memory(self, candidate: str):
        """
        :param candidate: str
        """
        return self.statistical_stats[candidate]['peak_memory']

    def recomputation_policy(self,
                             mem_limit: int,
                             max_peak_memory: int) -> None:
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
            mem_consumption -= self.get_memory(candidate=candidate)
            if (mem_consumption - mem_limit) <= 0 :
                break

    # def _get_recomp_src(self, candidate: any) -> None:
    #     """
    #     :param candidate: tuple
    #     """
    #     logging.info(f"Error in getting recomp_src for the candidate: {e}\n creating the recomp_src list")
    #     rank = self.statistical_stats[candidate]['rank']
    #     # get the total recomputetime from all subsequent nodes
    #     count = rank
    #     recomp_src = []
    #     total_runtime = 0
    #     while count >= 0:
    #         x = self.rank_mapping[count]
    #         recomp_src.append(x)
    #         total_runtime += self.statistical_stats[x]['run_time']
    #         count -= 1
    #     self.candidates_dict[candidate]['recomp_time'] = total_runtime

    def _get_recomp_src(self, candidate: any) -> None:
        """
        :param candidate: tuple
        """
        recomp_src = []
        cache_recent_subset = []
        count = 0
        while True:
            num_args = 0
            num_placeholders = 0
            if count == 0:
                # initial args are not counted in the recompute
                cand_args = [*self.node_name_mapping[candidate].args]
                # num_args += len(cand_args)
                # index_check = num_args * -1
                # append the nodes for recomp
                for argument in cand_args:
                    if argument.opcode == 'placeholder':
                        continue
                    else:
                        # add the args of the arg to the list
                        recomp_args = [*argument.args]
                        num_args += len(recomp_args)
                        for secondary_argument in recomp_args:
                            # add to the list of recomps
                            recomp_src.append(secondary_argument)
                            # add to the cache subset
                            cache_recent_subset.append(secondary_argument)
                # add to the count
                count += 1
                # check if the number of args equals the number of placeholders
                if num_args == num_placeholders:
                    break
            else:
                # iterate over the cache subset to get its args
                num_args += len(cache_recent_subset)
                for cached_node in cache_recent_subset:
                    # check if it is a placeholder node
                    if cached_node.opcode == 'placeholder':
                        num_placeholders += 1
                        cache_recent_subset.remove(cached_node)
                        continue
                    else:
                        # iterate over the args and add them
                        cached_args = [*cached_node.args]
                        for secondary_argument in cached_args:
                            # add to the recomp_src
                            recomp_src.append(secondary_argument)
                            # add to the cache
                            cache_recent_subset.append(secondary_argument)
                        # remove the cached node from the cache
                        cache_recent_subset.remove(cached_node)
                # check if the args and num placeholders equal
                if num_args == num_placeholders:
                    break
        return recomp_src

    def _get_recomp_time(self, candidate: any) -> float:
        """
        :param recomp_src: list
        """

        try:
            return self.candidate_dict[candidate]['recomp_time']
        except Exception:
            try:
                recomp_time = 0
                for recomp_node in self.candidates_dict[candidate]['recomp_src']:
                    recomp_time += self.statistical_stats[recomp_node.name]['run_time']
                self.candidates_dict[candidate]['recomp_time'] = recomp_time
                return recomp_time
            except Exception as e:
                logging.error(f"Error in getting recomp_time for the candidate: {e}")
                raise ValueError(f"Error in getting recomp_time for the candidate: {e}")

    def initialization(self) -> None:
        """
        :param candidates: list
            The list of intermediate nodes in the graphprofiler for either recompuation or swapping
        """
        for candidate in self.candidates:
            self._get_recomp_src(candidate=candidate)
            recomp_time = self._get_recomp_time(candidate=candidate)
            mem_consumption = self.get_memory(candidate=candidate)
            self.candidates_dict[candidate]['recompute_ratio'] = mem_consumption/recomp_time

    def max_candidate(self):
        """
        :param candidates: list
        """
        max_candidate = None
        for candidate in self.candidates:
            if max_candidate is None:
                max_candidate = candidate
            elif self.candidates_dict[max_candidate]['recompute_ratio'] < self.candidates_dict[candidate]['recompute_ratio']:
                max_candidate = candidate

        return max_candidate

    def update_recompute_ratio(self) -> None:
        """
        :param candidates: tuple
        """
        for candidate in self.candidates:
            candidate_memory = self.get_memory(candidate=candidate)
            self.candidates_dict[candidate]['recompute_ratio'] = candidate_memory/self.candidates_dict[candidate]['recomp_time']

    def updating_existing_recomputations(self, candidate: str):
        """
        :param recomps: list
            The list of intermediate nodes for recomputation
        :param candidate: tuple
        """
        recomp_cnt = 1
        for rp in self.recomps:
            if candidate in self.candidates_dict[rp]['recomp_src']:
                recomp_cnt += 1
                self.candidates_dict[rp]['recomp_src'].remove(candidate)
                self.candidates_dict[rp]['recomp_src'].extend(self.candidates_dict[candidate]['recomp_src'])
                self.candidates_dict[rp]['recomp_time'] += self._get_recomp_time(candidate=candidate)
        return recomp_cnt

    def replace_subsequent_uses_of(self,
            graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
    ) -> None:
        old_node_users = old_node.users
        for node in reversed(graph.nodes):
            if node == new_node:
                break
            if node in old_node_users:
                node.replace_input_with(old_node, new_node)

    def remove_detach_nodes(self, gm: fx.GraphModule) -> fx.GraphModule:
        for node in gm.graph.nodes:
            if node.target == torch.ops.aten.detach.default:
                input_node = node.all_input_nodes[0]
                node.replace_all_uses_with(input_node)
                if len(node.users) == 0:
                    gm.graph.erase_node(node)
        gm.graph.lint()
        gm.recompile()
        return gm

    def get_name_to_node_map(self) -> None:
        name_to_node = {}
        for node in self.graph_module.graph.nodes:
            name_to_node[node.name] = node
        return name_to_node

    def activation_checkpointing(self) -> fx.GraphModule:
        # iterate over the nodes that need recomputation
        for node_name in self.recomps:
            node = self.node_name_mapping[node_name]
            first_back_access = self.node_pass_usage[node]['first_backward_use_node']
            node_to_recompute = [node]
            node_to_recompute_names = [node_name]
            nodes_required_to_recompute = self.candidates_dict[node_name]['recomp_src']

            recompute_subgraph = _extract_graph_with_inputs_outputs(
                joint_graph=self.graph_module.graph,
                inputs=nodes_required_to_recompute,
                outputs=node_to_recompute,
            )
            print("Extracted recomputation sub-graph: ")
            recompute_subgraph.print_tabular()

            # Insert the nodes of the new sub-graph in the old graph before the first
            # backward access of the node to be recomputed.
            with self.graph_module.graph.inserting_before(first_back_access):
                for n in recompute_subgraph.nodes:
                    if n.op == "placeholder" or n.op == "output":
                        continue
                    # Copy the nodes of the new sub-graph to old graph and transform its
                    # inputs to match the old-graph inputs. The arg_transform function
                    # will pass the input arguments of the new node and will expect a
                    # mapping to the nodes of the old graph.
                    new_node = self.graph_module.graph.node_copy(
                        n, arg_transform=lambda arg: self.node_name_mapping[arg.name]
                    )

                    if n.name in node_to_recompute_names:
                        old_node = self.node_name_mapping[n.name]
                        # Replace all the uses of the old node with new recomputation node
                        self.replace_subsequent_uses_of(
                            self.graph_module.graph, old_node=old_node, new_node=new_node
                        )
                    # Add the new node to our name to node mapping
                    self.node_name_mapping[n.name] = new_node

        self.graph_module.graph.lint()
        self.graph_module.recompile()
        return self.graph_module