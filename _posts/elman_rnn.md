This relates the experience of implementing an Elman network for modeling sequential information. 

The Elman network was introduced by Jeffrey L. Elman in a 1990 paper titled 
[Finding Structure in Time](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1). It uses ideas very similar to those introduced by
Michael I. Jordan in a 1986 [technical report](https://cseweb.ucsd.edu/~gary/PAPER-SUGGESTIONS/Jordan-TR-8604-OCRed.pdf).

The Jordan and Elman models address a basic limitation of multilayer perceptrons: the input size and number of layers (processing steps) is fixed.
This can be a problem when processing sequential inputs where the input size is variable (e.g. speech, sound, time series ...). Jordan's idea was to go
go through the inputs one at a time, and having a **state** layer that maintains information about the previous inputs up to this point.
To implement this state, the inputs to the network are expanded with a *state* layer whose inputs are the outputs of the network at the previous
time step. In other words, the inputs to the hidden layer at time $t$ include the outputs of the network at time $t-1$.

The Elman network, also known as the SRN or [Simple Recurrent Network](https://web.stanford.edu/group/pdplab/pdphandbook/handbookch8.html) is
very similar. The difference is that the input to the hidden layer, instead of receiving a copy of the network's output, receives a copy of
the hidden layer's activity at the previous time step.

The task for this implementation was to learn to recreate a set of hand-drawn traces:
![the figures to be traced](/assets/figures.png)

These traces were drawn by hand and exported into the `.svg` format. I wrote a function (link) to extract a sequence of $(x,y)$ coordinates
from the svg file, which were stored in Numpy array with roughly 1000 rows and 2 columns. The task of the network was to learn this sequence
of traces, but due to the noisy nature of hand drawing the network must create limit cycle attractors that roughly follow the mean
trajectory of the lines.
