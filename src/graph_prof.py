from enum import Enum
from typing import Dict
import logging
import torch
import time
import torch.fx as fx
import json
from typing import Dict, Any
from pathlib import Path

import os
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._functional_collectives import all_reduce


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"
    # starter code


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # You should perform the static analysis of the graph here. In
        # particular, you might want to find the intermediate
        # nodes/activations/feature_maps in the graph that will be defined as
        # those nodes which are not parameters (not placeholder node types) but
        # are created during the forward pass and are also used in the backward
        # pass for computation. 

        # The boundary between the forward pass and backward pass can be
        # identified by locating the node
        # '%sep : [num_users=1] =
        # call_function[target=torch.ops.separator.sep.default]' which will 
        # define the end of the forward pass. You will see the loss function
        # after thsi operation and then you will encounter a node named,
        # '%sep_backward : [num_users=1] =
        # call_function[target=torch.ops.separator.sep_backward.default]'. This
        # node marks the beginning of the backward pass. 

        # For these intermediate nodes in the graph, you will record their last
        # use in the forward pass and their first use in the backward pass.

        # The parameters of the models are the placeholder (input) nodes of the
        # graph. Note that not all the placeholder nodes of the graph are
        # parameters. The number of parameters of the graphs and the gradients
        # should be equal.

        # You will also see several operators of the type 
        #' %tag_grad :[num_users=1] =
        # call_function[target=torch.ops.dummy.tag_grad.default]'. These are
        # also dummy operations added after every gradient produced in the
        # backward pass.
        # static data analysis
        self.total_runtime_sec: list[float] = []
        self.runtimes_sec: Dict[torch.fx.Node, list[float]] = {}
        self.gpu_total_memory: list[int] = []
        self.activation_memory: list[int] = []
        self.parameter_memory: list[int] = []
        self.intermediate_memory: list[int] = []
        self.gradient_memory: list[int] = []
        # self.node_memory_stat: Dict[torch.fx.Node, int] = {}
        self.swap: bool = True

        # self.activation_memory_total: int = 0
        # self.parameter_memory_total: int = 0
        # self.intermediate_memory_total: int = 0
        # self.gradient_memory_total: int = 0
        # graph logic for last use and first use
        self.graph_meta_data: Dict[str, Any] = {}
        self.env_cpu = {}
        #create an indicator for the backward pass
        self.forward_pass : bool = True
        self.forward_pass_build: bool = True
        # set the gpu environment
        self.gpu_env = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
        logging.info(f"GPU enviornment: {self.gpu_env}")
        # Printing the input nodes, node users and node names.
        logging.info(f"Graph initialized listing out the nodes:")
        # create the mapping for last use in the forward pass
        self.node_pass_usage : dict[fx.Node, dict] = {}
        # create a counter for the id of the nodes in both forward and backwards pass
        self.last_forward_pass_id : list[int] = []
        self.first_backward_pass_id : list[int] = []
        # counter_id = 0
        for node in self.module.graph.nodes:
            # logging.info(f"Node name: {node.name}")
            # logging.info(f"Node type: {node.op}")
            # logging.info(f"Node target: {node.target}")
            # logging.info(f"Input to this node: {[x.name for x in node.all_input_nodes]}")
            # logging.info(f"Users of this node: {node.users}")
            # logging.info("\n")
            if node.name == 'sep':
                logging.info(f"Forward pass Ended at node {node.name} while building usage mapping")
                self.forward_pass_build = False

            # create the forward pass last use dict
            if self.forward_pass_build:

                # check if the node inupts are empty
                if len(node.all_input_nodes) == 0:
                    # add the node to the forward pass dict with None last use node strind
                    self.node_pass_usage[node] = {
                                    # 'node': node,
                                    'last_forward_use_node': None,
                                    'first_backward_use_node': None,
                                    'last_backward_use_node': None
                    }
                else:
                    for input_node in node.all_input_nodes:

                        # add the node key to the forward pass if it does not exist else update the last use string
                        if input_node not in self.node_pass_usage.keys():
                            self.node_pass_usage[input_node] = {
                                # 'node': input_node,
                                'last_forward_use_node': node,
                                'first_backward_use_node': None,
                                'last_backward_use_node': None
                            }
                        else:
                            self.node_pass_usage[input_node]['last_forward_use_node'] = node
            else:
                # check if the node is in the pass dictionary
                for input_node in node.all_input_nodes:
                    if input_node not in self.node_pass_usage.keys():
                        self.node_pass_usage[input_node] = {
                            # 'node': input_node,
                            'last_forward_use_node': None,
                            'first_backward_use_node': node,
                            'last_backward_use_node': node
                        }
                    else:
                        if self.node_pass_usage[input_node]['first_backward_use_node'] is None:
                            # update
                            self.node_pass_usage[input_node]['first_backward_use_node'] = node
                            self.node_pass_usage[input_node]['last_backward_use_node'] = node
                        else:
                            # update only the last backward use node
                            self.node_pass_usage[input_node]['last_backward_use_node'] = node
            # categorize the node
            if node.target == torch.ops.aten._fused_adam.default:
                param_adam_args = node.args[0]
                grad_adam_args = node.args[1]
                # iterate over the param nodes and categorize them in the graph metadata
                for param in param_adam_args:
                    self.graph_meta_data[param.name] = 'parameter'
                for grad in grad_adam_args:
                    self.graph_meta_data[grad.name] = 'gradient'
            else:
                self.graph_meta_data[node.name] = self._categorize_node(node=node)
        logging.info(f"Graph metadata dict: {self.graph_meta_data}")
        # create a list of the nodes that are last used
        self.last_forward_pass_id = [self.node_pass_usage[x]['last_forward_use_node'] for x in self.node_pass_usage.keys()]
        self.first_backward_pass_id = [self.node_pass_usage[x]['first_backward_use_node'] for x in self.node_pass_usage.keys()]
        self.last_backward_pass_id = [self.node_pass_usage[x]['last_backward_use_node'] for x in self.node_pass_usage.keys()]

    def _categorize_node(self, node: fx.Node) -> str:
        """
        Categorize the node as an intermediate node, parameter, activation/feature map, gradient
        """
        # categorize the node based on the input
        if node.op == 'placeholder':
            return 'parameter'
        if 'relu' in node.name:
                return 'activation_feature'
        elif 'convolution' in node.name.lower():
                return 'activation_feature'
        elif 'pooling' in node.name.lower():
                return 'activation_feature'
        elif 'tag_grad' in node.name.lower():
                return 'gradient'
        return 'intermediate'

    def _get_memory_usage(self,
                          node_tensor: torch.Tensor) -> int:
        """
        Function gets the memory usage of a feature map, activation, paramter, or gradient node
        """
        # get the current usage in gpu memory
        try:
            torch_bytes = torch.numel(node_tensor) * torch.element_size(node_tensor)
            # self.node_memory_stat[node] = torch_bytes
        except Exception as e:
            try:
                torch_bytes = torch.Tensor.nelement(node_tensor) * torch.Tensor.element_size(node_tensor)
                # self.node_memory_stat[node] = torch_bytes
            except Exception as e:
                # logging.warning(f"node: {node} may not be a tensor type of object {type(self.env[node])}: {e} ")
                torch_bytes = 0
            return torch_bytes

    # def update_total_memory(self, node: fx, add_flag: bool) -> None:
    #     """
    #     function updates the total memory of the category memory total
    #     """
    #     # get the category
    #     node_category = self.graph_meta_data[node.name]
    #     # get the memory usage
    #     try:
    #         mem_usage = self.node_memory_stat[node]
    #     except KeyError:
    #         logging.warning(f"node: {node.name} may not be a tensor type of object")
    #         return None
    #     # check if it is currently the forward pass
    #     if add_flag:
    #         match node_category:
    #             case 'activation_feature':
    #                 self.activation_memory_total += mem_usage
    #                 logging.info(f"activation_feature adding memusage for node {node.name}: {mem_usage} total {self.activation_memory_total}")
    #             case 'gradient':
    #                 self.gradient_memory_total += mem_usage
    #             case 'intermediate':
    #                 self.intermediate_memory_total += mem_usage
    #             case 'parameter':
    #                 self.parameter_memory_total += mem_usage
    #     else:
    #         match node_category:
    #             case 'activation_feature':
    #                 self.activation_memory_total -= mem_usage
    #                 logging.info(
    #                     f"activation_feature subtracting memusage for node {node.name}: {mem_usage} total {self.activation_memory_total}")
    #             case 'gradient':
    #                 self.gradient_memory_total -= mem_usage
    #             case 'intermediate':
    #                 self.intermediate_memory_total -= mem_usage
    #             case 'parameter':
    #                 self.parameter_memory_total -= mem_usage

    def update_memory_usage(self) -> None:
        """
        Function iterates over the env parameter to collect the memory usage of each feature, paramter, or gradient node
        """
        # set the totals
        parameter_memory_total = 0
        activation_memory_total = 0
        gradient_memory_total = 0
        intermediate_memory_total = 0
        # iterate over the keys
        for node_env in self.env.keys():
            # get the category of the node
            category = self.graph_meta_data[node_env.name]
            # check if the value is a torch tensor
            if torch.is_tensor(self.env[node_env]):
                # get the memory usage
                mem_usage = self._get_memory_usage(node_tensor=self.env[node_env])

                # add the mem_usage to the total
                match category:
                    case 'activation_feature':
                        activation_memory_total += mem_usage
                    case 'gradient':
                        gradient_memory_total += mem_usage
                    case 'intermediate':
                        intermediate_memory_total += mem_usage
                    case 'parameter':
                        parameter_memory_total += mem_usage


        self.parameter_memory.append(parameter_memory_total)
        self.activation_memory.append(activation_memory_total)
        self.gradient_memory.append(gradient_memory_total)
        self.intermediate_memory.append(intermediate_memory_total)

    def analysis_dump(self) -> None:
        """
        Dump the statistics captured in the run for all of the totals
        """
        # create the analysis dict
        analysis_dict = {
            # 'activation_memory_total': self.activation_memory_total,
            # 'gradient_memory_total': self.gradient_memory_total,
            # 'intermediate_memory_total': self.intermediate_memory_total,
            # 'parameter_memory_total': self.parameter_memory_total,
            'total_runtime_sec': self.total_runtime_sec,
            'gpu_total_memory': self.gpu_total_memory,
            'activation_memory': self.activation_memory,
            'parameter_memory': self.parameter_memory,
            'intermediate_memory': self.intermediate_memory,
            'gradient_memory': self.gradient_memory
        }

        # create the path if it does not exist
        Path("analysis").mkdir(parents=True, exist_ok=True)
        # dump the analysis to the file path
        json_analysis = json.dumps(analysis_dict, indent=4)
        with open("analysis/analysis.json", "w") as outfile:
            outfile.write(json_analysis)

        logging.info(f"data written to json dumps")
        outfile.close()


    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> torch.Any:
        t_start = time.time()
        return_val =  super().run(
                        *args, initial_env=initial_env, enable_io_processing=enable_io_processing
                    )
        t_end = time.time()
        self.total_runtime_sec.append(t_end - t_start)
        # dump the analysis
        self.analysis_dump()
        return return_val

    def GPU_checker(self, node: fx.Node):
        """
        Check if the tensors inputs to the node are on gpu. if not raise value error
        """
        # iterate over the node inputs
        for input_tensor in node.all_input_nodes:
            if torch.is_tensor(self.env[input_tensor]):
                # check if the tesnor is in gpu
                if self.env[input_tensor].get_device() == -1:
                    if not torch.backends.mps.is_available():
                        raise ValueError(f"Tensor is not on gpu for {input_tensor}, it is in CPU")
                else:
                    logging.info(f"Tensor is on gpu for {input_tensor}")
                    # if node.name == 'div_1':
                        # logging.info(f"ENV: {self.env[input_tensor]}")
                        # logging.info(f"ENV_cpu: {self.env_cpu[input_tensor]}")
            else:
                logging.info(f"Tensor is not on gpu or is not a tensor for {input_tensor} value is {type(self.env[input_tensor])}")

    def swap_memory(self, node: fx.Node, to_cpu: bool=True) -> None:
        """
        Function swaps the memory of a feature map to either CPU or GPU. CPU swap happens on the last use
        of the feature map in the forward pass. GPU swap happens on the first use of the feature map in the backward
        pass

        node: fx.Node with inputs to swap the memory to cpu or gpu
        to_cpu: bool if true swap to cpu if false swap to gpu
        """
        # swap memory to cpu when true
        if to_cpu:
            # check if the node is a last use in the forward pass
            if node in self.last_forward_pass_id:
                # logging.info(f"Node: {node.name} has inputs to swap to cpu...")
                # iterate through the input nodes
                for input_node in node.all_input_nodes:
                    if self._categorize_node(input_node) == 'activation_feature':
                        # check if the forward pass last node equals the current node
                        if node == self.node_pass_usage[input_node]['last_forward_use_node'] and torch.is_tensor(self.env[input_node]):
                            logging.info(f"Node: {node.name}: swapping {input_node.name} to CPU")
                            # add the key and the cpu tensor in env_cpu
                            self.env_cpu[input_node] = self.env[input_node].to('cpu')
                            # remove the env tensor from memory
                            self.env[input_node] = None

        else:
            # check if the node is the first use in the backwards pass
            if node in self.first_backward_pass_id:
                # logging.info(f"Node: {node.name} has inputs to swap back to GPU...")
                # iterate through the input nodes
                for input_node in node.all_input_nodes:
                    # check if the forward pass last node equals the current node
                    if self.graph_meta_data[input_node.name] == 'activation_feature':
                        try:
                            if node == self.node_pass_usage[input_node]['first_backward_use_node'] and torch.is_tensor(self.env_cpu[input_node]):
                                logging.info(f"Node: {node.name}: swapping {input_node.name} back to GPU")
                                # bring the tensor back to GPU in the env environment
                                self.env[input_node] = self.env_cpu[input_node].to(self.gpu_env)
                                # remove the env tensor from memory
                                self.env_cpu[input_node] = None
                        except Exception as e:
                            logging.error(f"wasn't moved to cpu {input_node}: {e}")

    def run_node(self, n: fx.Node) -> Any:

        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.
        if n.name == 'sep':
            self.forward_pass = False
        if not self.forward_pass:
            logging.info(f"In the backwards pass, checking if swapping back to memory is needed...")
            # self.GPU_checker(node=n)
            # swap any tensors needed as inputs back to gpu
            if self.swap:
                self.swap_memory(node=n, to_cpu=False)

        # you can start measuring the run-time of a node here
        logging.info(f"Running node: {n} Nodetype: {n.op}")
        t_start = time.time()
        result = super().run_node(n)
        t_end = time.time()

        # set the
        self.env[n] = result
        if n not in self.runtimes_sec.keys():
            # create a key with the empty list of times
            self.runtimes_sec[n] = []
        self.runtimes_sec[n].append(t_end - t_start)
        # get the environment keys after each run
        # logging.info(f"Listing the environment variable keys in environment: {self.env.keys()}")
        # you can end measuring the run-time of a node here
        # HINT: Use torch.cuda.Events for doing time measurements of operations.

        # check if the node result is a tensor
        # print(f"Result type: {type(result)}")
            # logging.info(f"node stats: {self.node_memory_stat}")


        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.
        if self.forward_pass:
            # logging.info(f"In the forwards pass, checking if swapping to CPU memory is needed...")
            # self.GPU_checker(node=n)
            # swap any tensors needed as inputs back to gpu
            if self.swap:
                self.swap_memory(node=n, to_cpu=True)
        else:
            if n in self.last_backward_pass_id:
                # check if the input nodes are the last use in the backwards pass
                for input_node in n.all_input_nodes:
                    if self.node_pass_usage[input_node]['last_backward_use_node'] == n:
                        # remove from the env
                        self.env[input_node] = None

        # get the total memory of the categories
        self.update_memory_usage()
        # get the memory of result of gpu
        gpu_memory_usage = torch.mps.current_allocated_memory()
        self.gpu_total_memory.append(gpu_memory_usage)

        return result


if __name__ == "__main__":
    layers = 10
    # dim = 100
    # num_iters = 5
    # mps_device = torch.device("mps")
    # model = TTModel(layers, dim).to(mps_device)
    # TT_nn_graph = fx.symbolic_trace(model)
    # print(type(TT_nn_graph))
    # profiler_example = GraphProfiler(module=TT_nn_graph)
    # batch = torch.randn(10, dim).to(mps_device)
    # profiler_example.run(batch)
    # print(profiler_example.gpu_total_memory)
    # print(profiler_example.total_runtime_sec)