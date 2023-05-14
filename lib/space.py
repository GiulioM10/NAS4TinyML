import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from individual import Individual, ISpace
import types
from typing import Union, Text
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Union, Tuple, List, Dict
import time

def compute_naswot_score(net: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device):
    with torch.no_grad():
        codes = []

        def hook(self: nn.Module, m_input: torch.Tensor, m_output: torch.Tensor):
            code = (m_output > 0).flatten(start_dim=1)
            codes.append(code)

        hooks = []
        for m in net.modules():
            if isinstance(m, nn.ReLU):
                hooks.append(m.register_forward_hook(hook))

        _ = net(inputs)

        for h in hooks:
            h.remove()

        full_code = torch.cat(codes, dim=1)

        # Fast Hamming distance matrix computation
        del codes, _
        full_code_float = full_code.float()
        k = full_code_float @ full_code_float.t()
        del full_code_float
        not_full_code_float = torch.logical_not(full_code).float()
        k += not_full_code_float @ not_full_code_float.t()
        del not_full_code_float

        return torch.slogdet(k).logabsdet.item()
    


def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array

def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()

def _no_op(self, x):
    return x


# LogSynflow
def compute_synflow_per_weight(net, inputs, targets, device, mode='param', remap: Union[Text, None] = 'log'):
    net = net.train()

    # Disable batch norm
    for layer in net.modules():
        if isinstance(layer, (_BatchNorm, nn.BatchNorm2d, torch.nn.GroupNorm)):
            # TODO: this could be done with forward hooks
            layer._old_forward = layer.forward
            layer.forward = types.MethodType(_no_op, layer)

    # Convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # Convert to original values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # Keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = net(inputs)
    if isinstance(output, tuple):
        output = output[1]
    torch.sum(output).backward()

    # Select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            if remap is not None:
                g = torch.log(layer.weight.grad + 1)
                # remap_fun = {
                #     'log': lambda x: torch.log(x + 1),
                #     # Other reparametrizations can be added here
                #     # 'atan': torch.arctan,
                #     # 'sqrt': torch.sqrt
                # }
                # # LogSynflow
                # g = remap_fun[remap](layer.weight.grad)
            else:
                # Traditional synflow
                g = layer.weight.grad
            return torch.abs(layer.weight * g)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # Apply signs of all params
    nonlinearize(net, signs)

    # Enable batch norm again
    for layer in net.modules():
        if isinstance(layer, (_BatchNorm, nn.BatchNorm2d, torch.nn.GroupNorm)):
            layer.forward = layer._old_forward
            del layer._old_forward

    net.float()
    return sum_arr(grads_abs)


METRIC_NAME_MAP = {
    # log(x)
    'logsynflow': compute_synflow_per_weight,
    # x
    'synflow': lambda n, inputs, targets, dev: compute_synflow_per_weight(n, inputs, targets, dev, remap=None),
    'naswot': compute_naswot_score,
}


def kaiming_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def init_model(model):
    model.apply(kaiming_normal)
    model.eval()
    return model

def compute_tfm(exemplar: Individual,
                batch_or_loader: Union[Tuple[torch.Tensor, torch.Tensor], DataLoader],
                device: torch.device,
                metrics: Tuple = tuple(METRIC_NAME_MAP.keys()),
                ignore_errors: bool = False):
    network = exemplar.get_network()
    metric_trials = {}
    metric_times = {}
    init_time = 0
    for i in range(3):
        if isinstance(batch_or_loader, tuple):
            inputs, targets = batch_or_loader
        elif isinstance(batch_or_loader, DataLoader):
            inputs, targets = next(iter(batch_or_loader))
            inputs, targets = inputs.to(device), targets.to(device)
        else:
            raise ValueError("Invalid argument")
        start_time = time.time()
        network = init_model(network)
        end_time = time.time()
        init_time += (end_time - start_time)
        for metric_name in metrics:
            start_time = time.time()
            try:
                val = METRIC_NAME_MAP[metric_name](network, inputs, targets, device)
            except RuntimeError as ex:
                if not(ignore_errors):
                    # In case of errors set the value to None but keep running!
                    val = None
                else:
                    raise ex
            end_time = time.time()
            if metric_name not in metric_trials:
                metric_trials[metric_name] = []
                metric_times[metric_name] = []
            metric_trials[metric_name].append(val)
            metric_times[metric_name].append(end_time - start_time)
    for metric_name in metrics:
        if None in metric_trials[metric_name]:
            metric_trials[metric_name] = None
            metric_times[metric_name] = None
        else:
            metric_trials[metric_name] = np.mean(metric_trials[metric_name])
            metric_times[metric_name] = np.sum(metric_times[metric_name])
        
    return metric_trials, metric_times, init_time




class Space(ISpace):
    def __init__(self, net_length: int, block_list:List[str], ks_list: List[int],
               channel_list: List[int], exp_list: List[int], downsample_blocks: List[int],
               dataset: DataLoader, device: torch.device, 
               metrics: List[str]  = ['naswot', 'logsynflow'] #, 'synflow'
               ) -> None:
        """This object stores and manages all the info and properties of a given search space

        Args:
            net_length (int): Number of blocks to be stacked to form a network
            block_list (List[str]): Block types
            ks_list (List[int]): Kernel sizes
            channel_list (List[int]): Output channels
            exp_list (List[int]): Expansion terms
            downsample_blocks (List[int]): Position of downsampling blocks
            dataset (DataLoader): Dataset
            device (torch.device): Device to be used
            metrics (List[str], optional): Metrics to guide search. Defaults to ['naswot', 'logsynflow']#.
        """
        self.net_length = net_length
        self.block_list = block_list
        self.ks_list = ks_list
        self.channel_list = channel_list
        self.exp_list = exp_list
        self.downsample_blocks = downsample_blocks
        self.dataset = dataset
        self.metrics = metrics
        self.device = device

    def get_random_population(self, N: int) -> List[Individual]:
        """Get a random population of networks      
        Args:
            N (int): Size of population     
        Returns:
            List[Individual]: The individuals composing the population
        """
        individuals = []
        for _ in range(N):
          genome = self.sample_random_genome()
          individuals.append(Individual(genome, self, self.device))
        return individuals
  
    def sample_random_genome(self) -> List[List]:
        """Sample a random genome from the search space

        Returns:
            List[List]: The random genome
        """
        genome = []
        for i in range(self.net_length):
          block = np.random.choice(self.block_list, 1)[0]
          ker_size = np.random.choice(self.ks_list, 1)[0]
          channels = np.random.choice(self.channel_list, 1)[0]
          expansion = np.random.choice(self.exp_list, 1)[0]
          downsample = (i + 1) in self.downsample_blocks
          genome.append([block, ker_size, channels, expansion, downsample])
        return genome
  
    def compute_tfm(self, individual: Individual) -> Dict:
        """Compute training free metrics of an individual       
        Args:
            individual (Individual): An individual

        Returns:
              Dict: The various scores
        """
        metric_trials, _, _ = compute_tfm(individual, self.dataset, self.device, self.metrics)
        return metric_trials
    
    def mutate_nucleotide(self, nucleotide_type: int) -> Union[str, int, bool]:
        if nucleotide_type == 0:
            res = np.random.choice(self.block_list)
        elif nucleotide_type == 1:
            res = np.random.choice(self.ks_list)
        elif nucleotide_type == 2:
            res = np.random.choice(self.channel_list)
        elif nucleotide_type == 3:
            res = np.random.choice(self.exp_list)
        elif nucleotide_type == 4:
            res = np.random.choice([True, False])
        else:
            raise ValueError("Unknown nucleotide_type")
        return res
    
    def mutation(self, individual: Individual, R: int = 1, skip_downsampling: bool = True):
        new_genotype = individual.genotype.copy()
        gene_length = len(new_genotype[0]) if not skip_downsampling else (len(new_genotype[0]) - 1)
        start_gene = np.random.choice(self.net_length)
        start_nucleotide = np.random.choice(gene_length)
        for i in range(R):
            new_genotype[(start_gene + ((start_nucleotide + i) // gene_length))%self.net_length][(start_nucleotide + i)%gene_length] \
                = self.mutate_nucleotide((start_nucleotide + i)%gene_length)
        return Individual(new_genotype, self, self.device)
        