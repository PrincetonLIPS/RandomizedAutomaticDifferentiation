import numpy as np
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandLinear(torch.nn.Linear):
    """
    Linear layer with randomized automatic differentiation. Supports both 
    random projections (sparse=False) and sampling (sparse=True).

    Arguments:
        *args, **kwargs: The regular arguments to torch.nn.Linear.
        keep_frac: The fraction of hidden units to keep after reduction with randomized autodiff.
        full_random: If true, different hidden units are sampled for each batch element.
        Only compatible with sparse=True, as it leads to extreme memory usage with random projections.
        sparse: Sampling if true, random projections if false.
    """

    def __init__(self, *args, keep_frac=0.5, full_random=False, sparse=False, **kwargs):
        super(RandLinear, self).__init__(*args, **kwargs)
        self.keep_frac = keep_frac
        self.full_random = full_random
        self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))
        self.sparse = sparse

    def forward(self, input, retain=False, skip_rand=False):
        """
        If retain is True, uses the same random projection or sample as the last time this was called.
        This is achieved through reusing random seeds.

        If skip_rand is True, behaves like a regular torch.nn.Linear layer (sets keep_frac=1.0).
        """
        if not retain:
            self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))

        if skip_rand:
            keep_frac = 1.0
        else:
            keep_frac = self.keep_frac

        return RandMatMul.apply(input, self.weight, self.bias, keep_frac, self.full_random, self.random_seed, self.sparse)


class RandConv2dLayer(torch.nn.Conv2d):
    """
    Conv2d layer with randomized automatic differentiation. Supports both 
    random projections (sparse=False) and sampling (sparse=True).

    Arguments:
        *args, **kwargs: The regular arguments to torch.nn.Conv2d.
        keep_frac: The fraction of hidden units to keep after reduction with randomized autodiff.
        full_random: If true, different hidden units are sampled for each batch element.
        Only compatible with sparse=True, as it leads to extreme memory usage with random projections.
        sparse: Sampling if true, random projections if false.
    """

    def __init__(self, *args, keep_frac=0.5, full_random=False, sparse=False, **kwargs):
        super(RandConv2dLayer, self).__init__(*args,**kwargs)
        self.conv_params = {
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
        }
        self.keep_frac = keep_frac
        self.full_random = full_random
        self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))
        self.sparse = sparse

    def forward(self, input, retain=False, skip_rand=False):
        """
        If retain is True, uses the same random projection or sample as the last time this was called.
        This is achieved through reusing random seeds.

        If skip_rand is True, behaves like a regular torch.nn.Conv2d layer (sets keep_frac=1.0).
        """
        if not retain:
            self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))

        if skip_rand:
            keep_frac = 1.0
        else:
            keep_frac = self.keep_frac

        return RandConv2d.apply(input, self.weight, self.bias, \
            self.conv_params, keep_frac, self.full_random, self.random_seed, self.sparse)


class RandReLULayer(torch.nn.ReLU):
    """
    ReLU layer with randomized automatic differentiation. Supports both 
    random projections (sparse=False) and sampling (sparse=True).

    Not used in experiments as it leads to gradients with high variance.

    Arguments:
        *args, **kwargs: The regular arguments to torch.nn.ReLU.
        keep_frac: The fraction of hidden units to keep after reduction with randomized autodiff.
        full_random: If true, different hidden units are sampled for each batch element.
        Only compatible with sparse=True, as it leads to extreme memory usage with random projections.
        sparse: Sampling if true, random projections if false.
    """

    def __init__(self, *args, keep_frac=0.5, full_random=False, sparse=False, **kwargs):
        super(RandReLULayer, self).__init__(*args, **kwargs)
        self.keep_frac = keep_frac
        self.full_random = full_random
        self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))
        self.sparse = sparse

    def forward(self, input, retain=False, skip_rand=False):
        """
        If retain is True, uses the same random projection or sample as the last time this was called.
        This is achieved through reusing random seeds.

        If skip_rand is True, behaves like a regular torch.nn.ReLU layer (sets keep_frac=1.0).
        """
        if not retain:
            self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))

        if skip_rand:
            keep_frac = 1.0
        else:
            keep_frac = self.keep_frac

        return RandReLU.apply(input, keep_frac, self.full_random, self.random_seed, self.sparse)



###################################################################################################
############################ BEGIN: Common Methods for all layers #################################
###################################################################################################


def input2rp(input, kept_feature_size, full_random=False, random_seed=None):
    """
    Converts either a Linear layer or Conv2d layer input into a dimension reduced form
    using random projections.

    If the input is 2D, it is interpreted as (batch size) x (hidden size). The (hidden size) dimension
    will be reduced and the output will be (batch size) x (kept_feature_size).

    If the input is 4D, it is interpreted as (batch size) x (feature size) x (height) x (width). The
    (batch size) x (feature size) dims will be interpreted as a "effective batch size", and the array will
    be reduced along (height) x (width). The reduced tensor will be (batch size * feature size) x (kept_feature_size).

    Returns the reduced input and the random matrix used for the random projection. If using a random seed,
    the random matrix can be discarded, and the random seed can be used again in the rp2input method to regenerate
    the same random matrix.

    Arguments:
        input: Tensor of size (batch size) x (hidden size) or (batch size) x (feature size) x (height) x (width)
        kept_feature_size: The number to reduce the dimension to.
        full_random: If true, different hidden units are sampled for each effective batch element.
        WARNING: Will lead to extensive memory use if set to True in this method.
        random_seed: Use this random seed if not None.
    """

    def shp(t):
        return tuple(t.size())

    if len(shp(input)) == 4:
        batch_size = (shp(input)[0], shp(input)[1])
        feature_len = shp(input)[2] * shp(input)[3]
    elif len(shp(input)) == 2:
        batch_size = (shp(input)[0], )
        feature_len = shp(input)[1]

    if full_random:
        rand_matrix_size = (*batch_size, feature_len, kept_feature_size)
        matmul_view = input.view(*batch_size, 1, feature_len)
    else:
        rand_matrix_size = (feature_len, kept_feature_size)
        matmul_view = input.view(*batch_size, feature_len)

    # Create random matrix
    def gen_rad_mat(rm_size, feat_size, device):
        bern = torch.randint(2, size=rm_size, device=device, requires_grad=False)
        return (2.0 * bern - 1) / feat_size**0.5

    if random_seed:
        with torch.random.fork_rng():
            torch.random.manual_seed(random_seed)
            rand_matrix = gen_rad_mat(rand_matrix_size, kept_feature_size, input.device)
    else:
        rand_matrix = gen_rad_mat(rand_matrix_size, kept_feature_size, input.device)

    with torch.autograd.grad_mode.no_grad():
        dim_reduced_input = \
                torch.matmul(matmul_view, rand_matrix)
    return dim_reduced_input, rand_matrix


def rp2input(dim_reduced_input, input_shape, rand_matrix=None, random_seed=None, full_random=False):
    """
    Inverse of input2rp. Accepts the outputted reduced tensor from input2rp along with
    the expected size of the input.

    One and only one of rand_matrix or random_seed must be provided.
    This method must take either the rand_matrix outputted by input2rp, or the random seed
    used by input2rp. If the random seed is provided, this method will reconstruct rand_matrix, which
    contains the random matrix used to project the input.

    Arguments:
        dim_reduced_input: The outputted reduced tensor from input2rp.
        input_shape: The shape of the input tensor fed into input2rp.
        rand_matrix: The random matrix generated by input2rp.
        random_seed: Set this random seed to the same one as input2rp to reconstruct the random indices.
        full_random: Must be set to the same value as when input2rp was called.
    """

    def shp(t):
        return tuple(t.size())

    if rand_matrix is None and random_seed is None:
        print("ERROR in rp2input: One of rand_matrix or random_seed must be provided.")
        return
    if rand_matrix is not None and random_seed is not None:
        print("ERROR in rp2input: Only one of rand_matrix or random_seed must be provided.")
        return

    if len(input_shape) == 4:
        batch_size = (input_shape[0], input_shape[1])
        feature_len = input_shape[2] * input_shape[3]
    elif len(input_shape) == 2:
        batch_size = (input_shape[0], )
        feature_len = input_shape[1]

    kept_feature_size = shp(dim_reduced_input)[-1]
    if full_random:
        rand_matrix_shape = (*batch_size, feature_len, kept_feature_size)
    else:
        rand_matrix_shape = (feature_len, kept_feature_size)

    # Create random matrix
    def gen_rad_mat(rm_size, feat_size, device):
        bern = torch.randint(2, size=rm_size, device=device, requires_grad=False)
        return (2.0 * bern - 1) / feat_size**0.5

    if random_seed is not None:
        with torch.random.fork_rng():
            torch.random.manual_seed(random_seed)
            rand_matrix = gen_rad_mat(rand_matrix_shape, kept_feature_size, dim_reduced_input.device)

    with torch.autograd.grad_mode.no_grad():
        input = torch.matmul(dim_reduced_input, torch.transpose(rand_matrix, -2, -1))
        input = input.view(input_shape)

    return input


def input2sparse(input, kept_feature_size, full_random=False, random_seed=None):
    """
    Converts either a Linear layer or Conv2d layer input into a dimension reduced form
    using sampling.

    If the input is 2D, it is interpreted as (batch size) x (hidden size). The (hidden size) dimension
    will be reduced and the output will be (batch size) x (kept_feature_size).

    If the input is 4D, it is interpreted as (batch size) x (feature size) x (height) x (width). The
    (batch size) x (feature size) dims will be interpreted as a "effective batch size", and the input will
    be reduced along (height) x (width). The reduced tensor will be (batch size * feature size) x (kept_feature_size).

    Returns the reduced input and the random indices used for the sampling. If using a random seed,
    the random indices can be discarded, and the random seed can be used again in the sparse2input method to regenerate
    the same random indices.

    Arguments:
        input: Tensor of size (batch size) x (hidden size) or (batch size) x (feature size) x (height) x (width)
        kept_feature_size: The number to reduce the dimension to.
        full_random: If true, different hidden units are sampled for each effective batch element.
        random_seed: Use this random seed if not None.
    """

    def shp(t):
        return tuple(t.size())

    if len(shp(input)) == 4:
        batch_size = shp(input)[0] * shp(input)[1]
        feature_len = shp(input)[2] * shp(input)[3]
    elif len(shp(input)) == 2:
        batch_size = shp(input)[0]
        feature_len = shp(input)[1]

    if full_random:
        gather_index_shape = (batch_size, kept_feature_size)
    else:
        gather_index_shape = (1, kept_feature_size)

    # Create random matrix
    if random_seed is not None:
        with torch.random.fork_rng():
            torch.random.manual_seed(random_seed)
            gather_index = torch.randint(feature_len, gather_index_shape, device=input.device, dtype=torch.long)
    else:
        gather_index = torch.randint(feature_len, gather_index_shape, device=input.device, dtype=torch.long)

    with torch.autograd.grad_mode.no_grad():
        gathered_input = \
                torch.gather(input.view(batch_size, feature_len),
                             index=gather_index.expand(batch_size, -1), dim=-1).clone()
        # Normalization to ensure unbiased.
        gathered_input *= feature_len / kept_feature_size

    return gathered_input, gather_index


def sparse2input(gathered_input, input_shape, gather_index=None, random_seed=None, full_random=False):
    """
    Inverse of input2sparse. Accepts the outputted reduced tensor from input2sparse along with
    the expected size of the input.

    One and only one of gather_index or random_seed must be provided.
    This method must take either the gather_index outputted by input2sparse, or the random seed
    used by input2sparse. If the random seed is provided, this method will reconstruct gather_index, which
    contains the random indices used to sample the input.

    Arguments:
        gathered_input: The outputted reduced tensor from input2sparse.
        input_shape: The shape of the input tensor fed into input2sparse.
        gather_index: The random indices generated by input2sparse.
        random_seed: Set this random seed to the same one as input2sparse to reconstruct the random indices.
        full_random: Must be set to the same value as when input2sparse was called.
    """

    def shp(t):
        return tuple(t.size())

    if gather_index is None and random_seed is None:
        print("ERROR in sparse2input: One of gather_index or random_seed must be provided.")
        return
    if gather_index is not None and random_seed is not None:
        print("ERROR in sparse2input: Only one of gather_index or random_seed must be provided.")
        return

    if len(input_shape) == 4:
        batch_size = input_shape[0] * input_shape[1]
        feature_len = input_shape[2] * input_shape[3]
    elif len(input_shape) == 2:
        batch_size = input_shape[0]
        feature_len = input_shape[1]

    kept_feature_size = shp(gathered_input)[-1]
    if full_random:
        gather_index_shape = (batch_size, kept_feature_size)
    else:
        gather_index_shape = (1, kept_feature_size)

    if random_seed is not None:
        with torch.random.fork_rng():
            torch.random.manual_seed(random_seed)
            gather_index = torch.randint(feature_len, gather_index_shape, device=gathered_input.device, dtype=torch.long)

    with torch.autograd.grad_mode.no_grad():
        input = torch.zeros(batch_size, feature_len, device=gathered_input.device)

        batch_index = torch.arange(batch_size).view(batch_size, 1)
        input.index_put_((batch_index, gather_index), gathered_input, accumulate=True)
        input = input.view(input_shape)

    return input


#################################################################################################
############################ END: Common Methods for all layers #################################
#################################################################################################


class RandReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, keep_frac, full_random, random_seed, sparse):
        batch_size = input.size()[:-1]
        num_activations = input.size()[-1]

        ctx.input_shape = tuple(input.size())
        ctx.num_activations = num_activations
        ctx.keep_frac = keep_frac
        ctx.full_random = full_random
        ctx.random_seed = random_seed
        ctx.sparse = sparse
        kept_activations = int(num_activations * keep_frac + 0.999)

        # If we don't need to project, just fast-track.
        if ctx.keep_frac == 1.0:
            ctx.save_for_backward(input)
            return F.relu(input)

        if sparse:
            dim_reduced_input, _ = input2sparse(input, kept_activations, random_seed=random_seed, full_random=full_random)
        else:
            dim_reduced_input, _ = input2rp(input, kept_activations, random_seed=random_seed, full_random=full_random)

        # Saved Tensors should be low rank
        ctx.save_for_backward(dim_reduced_input)

        with torch.autograd.grad_mode.no_grad():
            return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.keep_frac < 1.0:
            (dim_reduced_input,) = ctx.saved_tensors
            if ctx.sparse:
                input = sparse2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)
            else:
                input = rp2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)
        else:
            (input,) = ctx.saved_tensors

        def cln(t):
            if t is None:
                return None
            ct = t.clone().detach()
            ct.requires_grad_(True)
            return ct

        cinput = cln(input)

        with torch.autograd.grad_mode.enable_grad():
            output = F.relu(cinput)
        input_grad_input = output.grad_fn(grad_output)

        return input_grad_input, None, None, None, None


class RandMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, keep_frac, full_random, random_seed, sparse):
        # Calculate dimensions according to input and keep_frac
        batch_size = input.size()[:-1]
        num_activations = input.size()[-1]

        ctx.input_shape = tuple(input.size())
        ctx.num_activations = num_activations
        ctx.keep_frac = keep_frac
        ctx.full_random = full_random
        ctx.random_seed = random_seed
        ctx.sparse = sparse
        kept_activations = int(num_activations * keep_frac + 0.999)

        # If we don't need to project, just fast-track.
        if ctx.keep_frac == 1.0:
            ctx.save_for_backward(input, weight, bias)
            linear_out = F.linear(input, weight, bias=bias)
            return linear_out

        if sparse:
            dim_reduced_input, _ = input2sparse(input, kept_activations, random_seed=random_seed, full_random=full_random)
        else:
            dim_reduced_input, _ = input2rp(input, kept_activations, random_seed=random_seed, full_random=full_random)

        # Saved Tensors should be low rank
        ctx.save_for_backward(dim_reduced_input, weight, bias)

        with torch.autograd.grad_mode.no_grad():
            return F.linear(input, weight, bias=bias)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.keep_frac < 1.0:
            dim_reduced_input, weight, bias = ctx.saved_tensors
            if ctx.sparse:
                input = sparse2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)
            else:
                input = rp2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)
        else:
            input, weight, bias = ctx.saved_tensors

        def cln(t):
            if t is None:
                return None
            ct = t.clone().detach()
            ct.requires_grad_(True)
            return ct

        cinput = cln(input)
        cweight = cln(weight)
        cbias = cln(bias)

        with torch.autograd.grad_mode.enable_grad():
            output = F.linear(cinput, cweight, bias=cbias)
        bias_grad_input, input_grad_input, weight_grad_input = output.grad_fn(grad_output)

        # Why are the gradients for F.linear like this???
        return input_grad_input, weight_grad_input.T, bias_grad_input.sum(axis=0), None, None, None, None


class RandConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, conv_params, keep_frac, full_random, random_seed, sparse):
        ctx.input_shape = tuple(input.size())
        ctx.keep_frac = keep_frac
        ctx.conv_params = conv_params
        ctx.full_random = full_random
        ctx.random_seed = random_seed
        ctx.sparse = sparse

        # If we don't need to project, just fast-track.
        if keep_frac == 1.0:
            ctx.save_for_backward(input, weight, bias)
            conv_out = F.conv2d(input, weight, bias=bias, **ctx.conv_params)
            return conv_out

        kept_image_size = int(keep_frac * ctx.input_shape[2] * ctx.input_shape[3] + 0.999)
        if ctx.sparse:
            dim_reduced_input, _ = input2sparse(input, kept_image_size, full_random=full_random, random_seed=random_seed)
        else:
            dim_reduced_input, _ = input2rp(input, kept_image_size, full_random=full_random, random_seed=random_seed)

        with torch.autograd.grad_mode.no_grad():
            conv_out = F.conv2d(input, weight, bias=bias, **ctx.conv_params)

        # Save appropriate for backward pass.
        ctx.save_for_backward(dim_reduced_input, weight, bias)

        with torch.autograd.grad_mode.no_grad():
            return conv_out

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.keep_frac < 1.0:
            dim_reduced_input, weight, bias = ctx.saved_tensors
            if ctx.sparse:
                input = sparse2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)
            else:
                input = rp2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)
        else:
            input, weight, bias = ctx.saved_tensors

        def cln(t):
            if t is None:
                return None
            ct = t.clone().detach()
            ct.requires_grad_(True)
            return ct

        cinput = cln(input)
        cweight = cln(weight)
        cbias = cln(bias)

        with torch.autograd.grad_mode.enable_grad():
            output = F.conv2d(cinput, cweight, bias=cbias, **ctx.conv_params)

        input_grad_output = grad_output
        input_grad_input, weight_grad_input, bias_grad_input = output.grad_fn(input_grad_output)

        return input_grad_input, weight_grad_input, bias_grad_input, None, None, None, None, None
