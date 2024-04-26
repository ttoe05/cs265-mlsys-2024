from functools import wraps
import os
import logging
from typing import Any
import torch.multiprocessing as mp
import torch
import torch.nn as nn
from graph_tracer import compile, SEPFunction
from graph_prof import GraphProfiler
from util import init_logger
from typing import Callable
from benchmarks import Experiment, model_names, model_batch_sizes
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.fx as fx

# This is the dummy model that is for use in starter code. But we will
# experiment with Resnet and Bert models from Torch Benchmark suite.


class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)
    # starter code


# Anymodel that is used will be wrapped with this model. We do this to call a
# dummy function 'SEPFunction', which is the separator function, that will call
# an identity operator at the end of the forward pass. This identity operator
# will get recorded in the computational graph and will inform you where the
# backward pass ends.


class WrappedDummyModel(nn.Module):
    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return SEPFunction.apply(self.mod(x))


# This is the train_step function that takes in a model, optimizer and an input
# mini batch and calls the forward pass, loss function and the optimizer step. A
# computational graph corresponding to a train_step will be captured by the
# compiler. 


def train_step(
    model: torch.nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor
):
    out: torch.Tensor = model(batch)
    out.sum().backward()
    optim.step()
    optim.zero_grad()


# Below is a user defined function that accepts a graph module and arguments of
# used to run the graph. You can essentially do any operation, graph
# modification, profiling etc. inside this function. Subsequent to modifications
# or graph analysis, the function expects you to return the modified graph back.
# In the given example, we just print the graph, and then initilize the graph
# profiler. The graph profiler extends the class fx.Interpreter, that allows you
# to run the graph node by node, more explanation in graph_prof.py.


def graph_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
    # logging.info(gm.graph)
    # print(f"Graph Tabular: {gm.graph.print_tabular()}")

    graph_profiler = GraphProfiler(gm)
    with torch.no_grad():
        graph_profiler.run(*args)

    return gm


# We first initialize the model, pass it to the wrapper model, then create a
# random input mini-batch and initilize the optimizer. We then call the compile
# function that takes in two arguments, a train_step function and a
# graph_transformation function. The train_step function is the one that will be
# traced by the compiler and a computational graph for the same will be created.
# This computational graph is then passed to the graph_transformation function
# to do any graph profiling, modifications and optimizations. This modified
# graph is stored and will be returned as the compiled function. In essence we
# do the following inside the compile function:

# def compile (train_step, graph_transformation):
#     @wraps(train_step)
#     def inner(*args, **kwargs):
#         if not_compiled:
#             original_graph, input_args = graph_tracer(train_step)
#             modified_graph = graph_transformation(original_graph, input_args)
#         output = modified_graph(*args, **kwargs)
#         return output
#     return inner


def experiment(model_name: str=None):
    # logging.getLogger().setLevel(logging.DEBUG)
    mps_device = torch.device("mps")
    if model_name is None:
        # run the dummy model code
        torch.manual_seed(20)
        batch_size = 100
        layers = 10
        dim = 100
        num_iters = 5
        dummy_model = DummyModel(dim=dim, layers=layers)
        model = WrappedDummyModel(dummy_model).to(mps_device)
        batch = torch.randn(batch_size, dim).to(mps_device)
        # model = WrappedDummyModel(dummy_model).cuda()
        # batch = torch.randn(batch_size, dim).cuda()
        optim = torch.optim.Adam(
            model.parameters(), lr=0.01, foreach=False, fused=True, capturable=True
        )

        compiled_fn = compile(train_step, graph_transformation)
        compiled_fn(model, optim, batch)
        return compiled_fn

    else:
        # check if the model name is in the list for training
        if model_name not in model_names:
            raise ValueError(f"Need to select a model from {model_names} or pass None to use the dummy model")
        # get the batch size based on the name
        batch_size = model_batch_sizes[model_name]
        logging.info(f"Running profiler on the model {model_name} with batch size: {batch_size}")
        # initialize the experiment from benchmarks
        # exp = Experiment(model_name=model_name, batch_size=batch_size)
        # model_wrapped = WrappedDummyModel(exp.model).to(mps_device)

        # create the fake_tensor
        exp = Experiment(model_names[1], model_batch_sizes[model_names[1]])
        # exp.run()
        compiled_fn = compile(exp.train_step, graph_transformation)
        compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
        print(f"compile function type: {type(compiled_fn)}")

# def plot_gpu_memory



if __name__ == "__main__":
    init_logger(file_name='graph_profiler.log')
    model_name = model_names[1]
    experiment(model_name=model_name)



