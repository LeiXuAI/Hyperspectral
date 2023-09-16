# referred from https://github.com/iancovert/dl-selection.git 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPSILON = np.finfo(float).eps
def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return torch.clamp(probs, min=eps, max=1-eps)

def concrete_sample(logits, temperature, shape=torch.Size([])):
    '''
    Sampling for Concrete distribution.

    See Eq. 10 of Maddison et al., 2017.
    '''
    uniform_shape = torch.Size(shape) + logits.shape
    u = clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                               device=logits.device))
    gumbels = - torch.log(- torch.log(u))
    scores = (logits + gumbels) / temperature
    return scores.softmax(dim=-1)

def bernoulli_concrete_sample(logits, temperature, shape=torch.Size([])):
    '''
    Sampling for BinConcrete distribution.

    See PyTorch source code, differs from Eq. 16 of Maddison et al., 2017.
    '''
    uniform_shape = torch.Size(shape) + logits.shape
    u = clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                               device=logits.device))
    return torch.sigmoid((F.logsigmoid(logits + EPSILON) - F.logsigmoid(-logits + EPSILON)
                          + torch.log(u + EPSILON) - torch.log(1 - u + EPSILON)) / temperature)

#########################################################################################################
class ConcreteGates(nn.Module):
    '''
    Input layer that selects features by learning binary gates for each feature,
    based on [1].

    [1] Dropout Feature Ranking for Deep Learning Models (Chang et al., 2017)

    Args:
      input_size: number of inputs.
      k: number of features to be selected.
      temperature: temperature for Concrete samples.
      init: initial value for each gate's probability of being 1.
      append: whether to append the mask to the input on forward pass.
    '''
    def __init__(self, input_size, output_size, k, temperature=1.0, init=0.99, implicit_temp=0.2, return_mask=True):
        super().__init__()
        init_logit = - torch.log(1 / torch.tensor(init) - 1) * implicit_temp
        self.logits = nn.Parameter(torch.full(
            (input_size,), init_logit, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.output_size = output_size
        self.temperature = temperature
        self.implicit_temp = implicit_temp
        self.k = k

    @property
    def probs(self):
        return torch.sigmoid(self.logits / self.implicit_temp)

    def forward(self, x, n_samples=None, return_mask=True):
        # Sample mask.
        n = n_samples if n_samples else 1
        m = self.sample(sample_shape=(n, len(x)))
        #print(m)
        #import pdb; pdb.set_trace()
        # Apply mask.
        x = x * m

        if not n_samples:
            x = x.squeeze(0)
            m = m.squeeze(0)

        if return_mask:
            return x, m
        else:
            return x

    def sample(self, n_samples=None, sample_shape=None):
        '''Sample approximate binary masks.'''
        if n_samples:
            sample_shape = torch.Size([n_samples])
        return bernoulli_concrete_sample(self.logits / self.implicit_temp,
                                               self.temperature, sample_shape)

    def get_inds(self, num_features=None, threshold=None, **kwargs):
        if num_features:
            inds = torch.argsort(self.probs, descending=True)[-self.k:]
        elif threshold:
            inds = (self.probs > threshold).nonzero()[:, 0]
        else:
            raise ValueError('num_features or threshold must be specified')
        return torch.sort(inds)[0].cpu().data.numpy()

    def get_left_inds(self, num_features=None):
        selected_inds = self.get_inds(num_features)
        whole_inds = [i for i in range(self.input_size)]
        unselected_inds = list(set(whole_inds).symmetric_difference(set(selected_inds)))
        return unselected_inds

    def extra_repr(self):
        return 'input_size={}, temperature={}, append={}'.format(
            self.input_size, self.temperature, self.append)


##################################################################################################################
class ConcreteSelector(nn.Module):
    '''
    Input layer that selects features by learning a binary matrix, based on [2].

    [2] Concrete Autoencoders for Differentiable Feature Selection and
    Reconstruction (Balin et al., 2019)

    Args:
      input_size: number of inputs.
      k: number of features to be selected.
      temperature: temperature for Concrete samples.
    '''
    def __init__(self, input_size, k, temperature=10.0):
        super().__init__()
        #self.logits = nn.Parameter(
        #    torch.zeros(k, input_size, dtype=torch.float32, requires_grad=True))
        self.w = torch.empty(k, input_size)
        self.logits = nn.Parameter(
            torch.nn.init.xavier_normal_(self.w), requires_grad=True
        )
        self.input_size = input_size
        self.k = k
        self.output_size = k
        self.temperature = temperature

    @property
    def probs(self):
        return self.logits.softmax(dim=1)

    def forward(self, x, n_samples=None, **kwargs):
        # Sample selection matrix.
        import pdb; pdb.set_trace()
        n = n_samples if n_samples else 1
        M = self.sample(sample_shape=(n, len(x))) # [1, 256, 20, 784]
        # Apply selection matrix.
        x = torch.matmul(x.unsqueeze(1), M.permute(0, 1, 3, 2)).squeeze(2)
        # [256, 1, 784] [1, 256, 784, 20] -> [1, 256, 1, 20]
        # if [1, 256, 784, 20] [256, 20, 784] -> [1, 256, 784, 784]
        # if [1, 256, 784, 20] [1, 20, 784] -> [1, 256, 784, 784]

        # Post processing.
        if not n_samples:
            x = x.squeeze(0)
        return x

    def sample(self, n_samples=None, sample_shape=None):
        '''Sample approximate binary matrices.'''
        if n_samples:
            sample_shape = torch.Size([n_samples])
        return concrete_sample(self.logits, self.temperature,
                                     sample_shape)

    def get_inds(self, **kwargs):
        inds = torch.argmax(self.logits, dim=1)
        return torch.sort(inds)[0].cpu().data.numpy()

    def get_left_inds(self, **kwargs):
        selected_inds = self.get_inds()
        whole_inds = [i for i in range(self.input_size)]
        unselected_inds = list(set(whole_inds).symmetric_difference(set(selected_inds)))
        return unselected_inds
    
    def extra_repr(self):
        return 'input_size={}, temperature={}, k={}'.format(
            self.input_size, self.temperature, self.k)

#######################################################################################################################
class ConcreteMask(nn.Module):
    '''
    Input layer that selects features by learning a k-hot mask.

    Args:
      input_size: number of inputs.
      k: number of features to be selected.
      temperature: temperature for Concrete samples.
      append: whether to append the mask to the input on forward pass.
    '''
    def __init__(self, input_size, output_size, k, temperature=10.0, implicit_temp=0.2, return_mask=False):
        super().__init__()
        self.w = torch.empty(k, input_size)
        self.logits = nn.Parameter(
            torch.nn.init.xavier_normal_(self.w), requires_grad=True
        )
        self.input_size = input_size
        self.k = k
        self.output_size = output_size
        self.temperature = temperature
        self.implicit_temp = implicit_temp

    @property
    def probs(self):
        return (self.logits / self.implicit_temp).softmax(dim=1)

    def forward(self, x, n_samples=None, return_mask=False):
        # Sample mask.
        n = n_samples if n_samples else 1
        m = self.sample(sample_shape=(n, len(x)))

        # Apply mask.
        x = x * m

        if not n_samples:
            x = x.squeeze(0)
            m = m.squeeze(0)

        if return_mask:
            return x, m
        else:
            return x

    def sample(self, n_samples=None, sample_shape=None):
        '''Sample approximate k-hot vectors.'''
        if n_samples:
            sample_shape = torch.Size([n_samples])
        elif not sample_shape:
            raise ValueError('n_samples or sample_shape must be specified')
        samples = concrete_sample(self.logits / self.implicit_temp,
                                        self.temperature, sample_shape)
        return torch.max(samples, dim=-2).values

    def get_inds(self, **kwargs):
        inds = torch.argmax(self.logits, dim=1)
        return torch.sort(inds)[0].cpu().data.numpy()
    
    def get_left_inds(self, **kwargs):
        selected_inds = self.get_inds()
        whole_inds = [i for i in range(self.input_size)]
        unselected_inds = list(set(whole_inds).symmetric_difference(set(selected_inds)))
        return unselected_inds

    def extra_repr(self):
        return 'input_size={}, temperature={}, k={}, append={}'.format(
            self.input_size, self.temperature, self.k, self.append)

#############################################################################################


# Implicit temperature for the link function to accelerate optimization.
class ConcreteMax(nn.Module):
    '''
    Input layer that selects features by learning probabilities for independent
    sampling from a Concrete variable, based on [3].

    [3] Learning to Explain: An Information Theoretic Perspective on Model
    Interpretation (Chen et al., 2018)

    Args:
      input_size: number of inputs.
      k: number of features to be selected.
      temperature: temperature for Concrete samples.
      append: whether to append the mask to the input on forward pass.
    '''
    def __init__(self, input_size, k, temperature=10.0, implicit_temp=0.2):
        super().__init__()
        self.w = torch.empty(k, input_size)
        self.logits = nn.Parameter(
            torch.nn.init.xavier_normal_(self.w), requires_grad=True
        )
        self.input_size = input_size
        self.k = k
        self.output_size = input_size
        self.temperature = temperature
        self.implicit_temp = implicit_temp

    @property
    def probs(self):
        return (self.logits / self.implicit_temp).softmax(dim=1)[0]

    def forward(self, x, n_samples=None, return_mask=True):
        # Sample mask.
        n = n_samples if n_samples else 1
        m = self.sample(sample_shape=(n, len(x)))

        # Apply mask.
        x = x * m

        if not n_samples:
            x = x.squeeze(0)
            m = m.squeeze(0)

        if return_mask:
            return x, m
        else:
            return x

    def sample(self, n_samples=None, sample_shape=None):
        '''Sample approximate k-hot vectors.'''
        if n_samples:
            sample_shape = torch.Size([n_samples])
        elif not sample_shape:
            raise ValueError('n_samples or sample_shape must be specified')
        samples = concrete_sample(self.logits.repeat(self.k, 1) / self.implicit_temp,
                                        self.temperature, sample_shape)
        return torch.max(samples, dim=-2).values

    def get_inds(self, **kwargs):
        inds = torch.argsort(self.logits[0])[-self.k:]
        return torch.sort(inds)[0].cpu().data.numpy()
    
    def get_left_inds(self, **kwargs):
        selected_inds = self.get_inds()
        whole_inds = [i for i in range(self.input_size)]
        unselected_inds = list(set(whole_inds).symmetric_difference(set(selected_inds)))
        return unselected_inds

    def extra_repr(self):
        return 'input_size={}, temperature={}, k={}'.format(
            self.input_size, self.temperature, self.k)


class ConcreteNew(nn.Module):
    '''
    Input layer that selects features by learning a k-hot vector.

    Args:
      input_size: number of inputs.
      k: number of features to be selected.
      temperature: temperature for Concrete samples.
    '''
    def __init__(self, input_size, k, temperature=10.0):
        super().__init__()
        self.logits = nn.Parameter(
            torch.randn(k, input_size, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.k = k
        self.output_size = k
        self.temperature = temperature

    @property
    def probs(self):
        probs = torch.softmax(self.logits / self.temperature, dim=1)
        return torch.clamp(torch.sum(probs, dim=0), max=1.0)

    def forward(self, x, n_samples=None, return_mask=True):
        # Sample mask.
        n = n_samples if n_samples else 1
        m = self.sample(sample_shape=(n, len(x)))

        # Apply mask.
        x = x * m

        if not n_samples:
            x = x.squeeze(0)
            m = m.squeeze(0)

        if return_mask:
            return x, m
        else:
            return x

    def sample(self, n_samples=None, sample_shape=None):
        '''Sample approximate binary masks.'''
        if n_samples:
            sample_shape = torch.Size([n_samples])
        return bernoulli_concrete_sample(self.probs, self.temperature,
                                               sample_shape)

    def get_inds(self, **kwargs):
        return torch.argsort(self.probs)[-self.k:].cpu().data.numpy()
    
    def get_left_inds(self, **kwargs):
        selected_inds = self.get_inds()
        whole_inds = [i for i in range(self.input_size)]
        unselected_inds = list(set(whole_inds).symmetric_difference(set(selected_inds)))
        return unselected_inds

    def extra_repr(self):
        return 'input_size={}, temperature={}, k={}'.format(
            self.input_size, self.temperature, self.k)