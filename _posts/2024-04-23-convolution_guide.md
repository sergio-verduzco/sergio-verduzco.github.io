---
title: "A recipe for direct and transpose convolutions in JAX and PyTorch"
date: 2024-04-23
layout: post
---

## Introduction

When it comes down to understanding convolutions (i.e. cross-correlations) as used in deep learning, the best resource is probably the [guide by Dumoulin and Visin](https://arxiv.org/abs/1603.07285). After reading such a guide you can understand the computation performed by convolution, together with the relation between various sizes: input size, output size, kernel size, padding, stride, and dilation.

Understanding all this, however, is just a pre-requisite to coding convolutions. Next you need to understand the API of the framework you will use. In the case of [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) and [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D), the API to implement 2D convolutional layers is quite straightforward: you just provide the size quantities such as number of output channels, padding, strides, and kernel size. Using the `groups` argument to implement depthwise-separable convolutions such as in [MobileNet](https://arxiv.org/abs/1704.04861) is a bit more subtle, but not too bad.

JAX has a number of [convenience functions for convolution](https://jax.readthedocs.io/en/latest/notebooks/convolutions.html), but the most flexible function is [jax.lax.conv_general_dilated](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated), which is different from the ones in PyTorch and Tensorflow.

The difference between JAX and PyTorch/TensorFlow comes from two main reasons. The first one is JAX's functional approach: all the relevant size values are passed when invoking `jax.conv_general_dilated`, whereas PyTorch/TensorFlow create a layer that stores many of the size values. The second reason for the difference is a bit more subtle: at the time of this writing, JAX uses a fractionally-strided convolutions, whereas PyTorch uses  a gradient-based approach for transpose convolutions. This is not our concern here, though.

This document is a guide to implement convolutions in PyTorch and JAX, especially in cases when you must control the output shape of direct and transpose convolutions.

## Handling direct convolutions in PyTorch

It can be confusing to think about the proper arguments to use for the convolution functions, especially when there are strides and dilations to consider. This can be cleared up by remembering that in any dimension the size values satisfy a simple arithmetic relation (Relationship 15 in Dumolin and Visin):

$$ o = \lfloor \frac{i + 2p -k - (k-1)(d-1)}{s} \rfloor + 1 $$

where $o$ = output size, $i$ = input size, $p$ = padding, $k$ = kernel size, $d$ = dilation, $s$ = stride.

Since you usually know the sizes of the input, output, stride, kernel, and dilation that you want, to provide consistent arguments for the convolution APIs all you need to calculate is the padding with this equation. The following Python function can be used for this:

```python
import math

def padding_fun(input_dim, output_dim, stride, kernel_size, dilation):
    """Calculate the padding for a convolution.

    The padding is calculated to attain the desired input and output
    dimensions.

    :param input_dim: dimension of the (square) input
    :type input_dim: int
    :param output_dim: height or width of the square convolution output
    :type output_dim: int
    :param stride: stride
    :type stride: int
    :param kernel_size: kernel size
    :type kernel_size: int
    :param dilation: dilation
    :type dilation: int
    :returns: padding for the convolution, residual for the transpose convolution
    :rtype: int, int
    """
    pad = math.ceil(0.5 * (
        stride * (output_dim - 1) - input_dim + dilation * (kernel_size - 1) + 1))
    err_msg = "kernel, stride, dilation and input/output sizes do not match"
    if pad >= 0:
        r = (input_dim + 2 * pad - dilation * (kernel_size - 1) - 1) % stride
        # verify that the padding is correct
        assert ( output_dim ==
            math.floor((input_dim + 2 * pad - dilation * (kernel_size - 1) - 1) / stride) + 1
            ), err_msg
        return pad, r
    else:
        raise ValueError(err_msg)
```

## Transpose convolutions

Transpose convolution has some extra confusing aspects. The first one is understanding that it is not the inverse of the convolution operation. It is just a convolution that, as far as the shape of its input and output, will go in the opposite way as an "original" convolution. Let's say the original convolution has a 4x4 input, no padding, and a 3x3 kernel size, leading to an output of size 2x2. A transpose convolution would have input of size 2x2 and output of size 4x4. If the kernel of the transpose convolution is also of size 3x3, then you need a padding of 2 on each side for all sizes to be consistent.

Notice that in the previous example, your transpose convolution could have a kernel of size 5x5, and a padding of 3 on each side would still lead to a 4x4 output. In general, the same kernel size is used for the original and the transpose convolution, so this should not produce much trouble.

So, if the transpose convolution is just another convolution, why do we have [tf.keras.layers.Conv2DTranspose](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose) and [torch.nn.ConvTranspose2D](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)? My guess is that these functions implement the transpose convolution as the backwards pass of the direct convolution, which avoids the meaningless products with the zero paddings in the input. But understanding this is not necessary to use these functions. What's necessary is to understand the arguments they expect.

To understand PyTorch's [ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) layer, let's first consider once again what the transpose convolution does.  Let's say we have a normal convolutional layer $CL$ with kernel, stride, and padding given by $k, s, p$ respectively.
We know that if we apply the $CL$ convolution to a 2D square input with size $H_i \times H_i$ on each channel, then the output will have the size $H_o \times H_o$ where $H_o = \lfloor H_i -k + 2p \rfloor /s + 1$. The transpose convolution of $CL$ is a convolution $CLT$ with the same kernel size, such that if applied to inputs of size $H_o \times H_o$, would produce outputs of size $H_i \times H_i$. In this way $CLT$ can recover the shape of the input after $CL$ has been applied, which is useful in things like autoencoders.

To obtain $CLT$ we could simply calculate the padding and stride that let us recover the original shape, and perform that convolution, but  `ConvTranspose2d` does this in a manner that is more computationally efficient. The first three arguments (input channels, output channels, and kernel size) are what the transpose convolution (e.g. $CLT$) will actually use. The `stride` and `padding` arguments are what the direct convolution (e.g. $CL$) would use. The `output_padding` argument handles the case when $H_i - k + 2p$ is not a multiple of $s$, which implies that there is not a one-to-one relation between $H_i$ and $H_o$, so to recover $H_i$ you may need to add extra rows and columns.

We can illustrate this with a simple example. Suppose you have a direct convolution acting on a 6x6 input, with kernel of size 3x3, stride 1, and no padding. This will result in an output of size 4x4. Let's see if `ConvTranspose2d` inverts this shape.

```python
import torch
from torch import nn

in_chan = 256
out_chan = 64
k = 3
s = 1
p = 0
op = 0

conv_transp = nn.ConvTranspose2d(in_chan, out_chan, k,
                                 stride=s,
                                 padding=p,
                                 output_padding=op)

input = torch.randn(32, 256, 4, 4)
out = conv_transp(input)

out.size()

# Output: torch.Size([32, 64, 6, 6])
```
As expected the output has a batch size of 32, 64 channels, and images of size 6x6.

But what if you don't know the padding? Then you can once again use the `padding` function defined above, but in the arguments you switch the `input_dim` and `output_dim` arguments, to account for the fact that we are doing the transpose convolution!

## Transpose convolutions in JAX

In the case of JAX, we can use [jax.lax.conv_general_dilated](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated) for both the direct and the transpose convolutions, which may in fact be a simpler way to think about this, but the users need to calculate the parameters of the transpose convolution by themselves. For some time there has been a [pull request](https://github.com/google/jax/pull/5772) for a function that provides an interface closer to those in PyTorch and TensorFlow, but at the time of this writing this has not been merged, perhaps because the `flax.linen.ConvTranspose` function provides something similar (Flax is a neural network library for JAX).

`flax.linen.ConvTranspose` is a wrapper around the `jax.lax.conv_transpose` function, which in turn seems to be a version of `jax.lax.conv_general_dilated` with fewer arguments (those relevant for transpose convolutions). Thus, the most general way to address convolutions in JAX is to understand the arguments to `jax.lax.conv_general_dilated`. After this the other methods will seem simple.

The `lhs` argument is the input (e.g. an image), and the `rhs` the kernel. **These and all other arguments work the same both for direct and transpose convolutions**. The `window_strides` and `padding` arguments are thus straightforward.

Just like using a window stride of 2 in direct convolution would make the output roughly half as small as the input, using an `lhs_dilation` argument equal to 2 will make the output roughly twice as large as the input. This is thus an important argument in transpose convolutions. `lhs_dilation` is in fact a dilation for the input; it does to the input array the same thing that the usual dilation does to the kernel.

The `rhs_dilation` is the kernel dilation. This is what the `dilation` argument would be in PyTorch's `ConvTranspose2d`.

The `dimension_numbers` argument is an object that for the input, the kernel, and the output, specify how many dimensions are, and what each one represents.

At this point we have enough elements to run some examples of transpose convolutions in JAX. First, let's repeat the simple example of a direct convolution with 6x6 input, 3x3 kernel, no padding, stride 1, and 4x4 output. Its transpose will have 4x4 input, 3x3 kernel, padding 2, stride 1, and 6x6 output. The following code produces these convolutions explicitly.

```python
import jax
from jax import lax
import jax.numpy as jnp

# First, the direct convolution

in_chan = 256  # number of input channels
out_chan = 64  # number of output channels
batch_size = 32
kernel = jnp.ones((in_chan, out_chan, 3, 3))
images = jnp.ones((batch_size, in_chan, 6, 6))

dn = lax.conv_dimension_numbers(images.shape,  # Dimensions for the input
                                kernel.shape,  # Dimensions of the kernel
                                ('NCHW', 'IOHW', 'NCHW'))  # what each dimension is
                                                     # in the input, kernel, and output
    
out = lax.conv_general_dilated(images, # lhs = image tensor
                               kernel, # rhs = conv kernel tensor
                               (1,1),  # window strides
                               'VALID', # padding mode
                               (1,1),  # lhs/image dilation
                               (1,1),  # rhs/kernel dilation
                               dn)     # dimension_numbers

print(f"Output shape: {out.shape}")
```

The output produced is: `Output shape: (32, 64, 4, 4)`.

```python
# Next, the corresponding transpose convolution
images = jnp.ones((batch_size, in_chan, 4, 4))

dn = lax.conv_dimension_numbers(images.shape,  # Dimensions for the input
                                kernel.shape,  # Dimensions of the kernel
                                ('NCHW', 'IOHW', 'NCHW'))  # what each dimension is
                                                     # in the input, kernel, and output
    
out = lax.conv_general_dilated(images, # lhs = image tensor
                               kernel, # rhs = conv kernel tensor
                               (1,1),  # window strides
                               [(2,2), (2,2)],  # padding mode
                               (1,1),  # lhs/image dilation
                               (1,1),  # rhs/kernel dilation
                               dn)     # dimension_numbers

print(f"Output shape: {out.shape}")
```

The output produced is: `Output shape: (32, 64, 6, 6)`.

Let's now try a more complex example of direct-transpose convolutions. The direct convolution will have input size 64x64, 5x5 kernel, stride 3, dilation 2, and padding 2 on both sides. This leads to an output size of 20x20. 

```python
# First, the direct convolution

in_chan = 256  # number of input channels
out_chan = 64  # number of output channels
batch_size = 32
kernel = jnp.ones((in_chan, out_chan, 5, 5))
images = jnp.ones((batch_size, in_chan, 64, 64))

dn = lax.conv_dimension_numbers(images.shape,  # Dimensions for the input
                                kernel.shape,  # Dimensions of the kernel
                                ('NCHW', 'IOHW', 'NCHW'))  # what each dimension is
                                                     # in the input, kernel, and output

out = lax.conv_general_dilated(images, # lhs = image tensor
                               kernel, # rhs = conv kernel tensor
                               (3,3),  # window strides
                               [(2,2), (2,2)], # padding mode
                               (1,1),  # lhs/image dilation
                               (2,2),  # rhs/kernel dilation
                               dn)     # dimension_numbers

print(f"Output shape: {out.shape}")
```

The output produced is: `Output shape: (32, 64, 20, 20)`. Let's now try to produce the transpose:

```python
# Next, the corresponding transpose convolution
images = jnp.ones((batch_size, in_chan, 20, 20))

dn = lax.conv_dimension_numbers(images.shape,  # Dimensions for the input
                                kernel.shape,  # Dimensions of the kernel
                                ('NCHW', 'IOHW', 'NCHW'))  # what each dimension is
                                                 # in the input, kernel, and output

out = lax.conv_general_dilated(images, # lhs = image tensor
                               kernel, # rhs = conv kernel tensor
                               (1,1),  # window strides
                               [(5,5), (5,5)],  # padding mode
                               (3,3),  # lhs/image dilation
                               (1,1),  # rhs/kernel dilation
                               dn)     # dimension_numbers

print(f"Output shape: {out.shape}")
```

The output is: `Output shape: (32, 64, 64, 64)`.

So that worked, but the arguments used for the transpose convolution may be mysterious, especially the padding and the lhs dilation (which creates fractionally-strided convolutions, as in section 4.5 of Dumolin and Visin). This is not particularly hard, as long as you understand what input dilation does. If you have an input dilation $\delta$, this means that in-between any two contiguous elements of the input we will insert $(\delta - 1)$ elements with zero value. Thus, the input will expand to a size $i + (i-1)(\delta - 1)$. Because of this, the size of the convolution's output at the relevant dimension will be:

$$ o = \lfloor \frac{i + (i-1)(\delta-1) + 2p -k - (k-1)(d-1)}{s} \rfloor + 1 $$

We need to update our padding function in order to handle input dilations:
```python

def padding2(input_dim, output_dim, stride, kernel_size, dilation, inp_dilation):
    """Calculate the padding for a convolution.

    The padding is calculated to attain the desired input and output
    dimensions. If an asymmetric padding is needed, the function will
    return a non-zero residual value r.

    You can specify the asymmetric padding in JAX's conv_general_dilated by using
    a padding argument like: ((pad + r, pad), (pad + r, pad))

    :param input_dim: size of the input along the relevant dimension
    :type input_dim: int
    :param output_dim: height or width of the convolution output
    :type output_dim: int
    :param stride: stride
    :type stride: int
    :param kernel_size: kernel size
    :type kernel_size: int
    :param dilation: dilation of the kernel
    :type dilation: int
    :param inp_dilation: dilation of the input
    :type inp_dilation: int
    :returns: padding for the convolution, residual value.
    :rtype: int, int
    """
    pad = 0.5 * (stride * (output_dim - 1) - input_dim - (input_dim - 1) * (inp_dilation - 1)
             + dilation * (kernel_size - 1) + 1)
    err_msg = f"Incompatible values: kernel:{kernel_size}, stride:{stride}, " + \
              f"input:{input_dim}, output:{output_dim} are incompatible"
    print(f"pad: {pad}")
    if pad > 0:
        r = math.ceil(pad % 1)
        pad = math.floor(pad)
        assert ( output_dim ==  
            math.floor((input_dim + (input_dim - 1) * (inp_dilation - 1) +
                        2 * pad + r - dilation * (kernel_size - 1) - 1) / stride) + 1
            ), err_msg
    else:
        raise ValueError(err_msg)
    return pad, r
```

With this you can handle even the trickiest pairs of direct-transpose convolutions, just remember to test that the shapes are correct before deploying.

One tool that can help in evaluating output shapes without actually performing the computations is [jax.eval_shape](https://jax.readthedocs.io/en/latest/_autosummary/jax.eval_shape.html). Being mindful that you [need to specify the static arguments](https://github.com/google/jax/discussions/19020), here's how one call may look:

```python
from functools import partial
jax.eval_shape(partial(lax.conv_general_dilated,
                       window_strides=s,
                       padding=p,
                       lhs_dilation=inp_dilation,
                       rhs_dilation=ker_dilation,
                       dimension_numbers=dn,
                      ),
                image,
                kernel,
              )
```
