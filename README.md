# FancierSleepingBears
A Deep Learning Final Project

## Progress Report

### Dataset
We have obtained the Kaggle headline data for classification and ran the following preprocessing steps;
1. Blew up contractions into component words.
2. Lowered all cases.
3. Replaced all numbers with Number Tokens.
4. Blew up Possessives into original word and a target noun token
5. Preserved full vocabulary size (no rare word replacement or extraction)

### Encoding
First we attempted to do an encoding of word to vector, implementing it within the pytorch framework. However, upon running over out entire dataset we found that even on a reasonably modern GPU time to run was too long to be useful. Instead we train the embedding layer in our model. This performs much quicker with no significant loss in accuracy.

### Test Networks
We built a network using the following layers:
1. Data embedding layer
2. RNN layers -Last hidden state->
3. Linear layers

This was able to achieve around 90% accuracy after 1 epoch (about 1 minute to run). We accidently had an interesting bug where we originally soft-maxing over the wrong dimension and still getting reasonable outputs. If we ran the test dataset through the network with the same batch size it would perform with 90% accuracy, but if batch size was modified accuracy would tank. We then fixed that and are getting ~92% on our model with variable batch size.

### Future Work

In the future plan to explore how modifying the RNNs embedding size, number of hidden units, LSTMs vs. GRUs, and number of layers within the net will affect our accuracy on the headlines dataset. Generalizing to the twitter dataset will be a little more challenging (e.g. how do you encode emojis) because of the need for improved pre-processing and the dataset being so much smaller. Finally, we plan to introduce convolutional based networks for comparison. For preprocessing we plan to test if eliminating single-used words will help to improve accuracy (the group is split its potential effectiveness).
