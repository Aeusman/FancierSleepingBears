# Experimental Summary Results

Results are reported as the highest validation accuracy reached and on what epoch.

## Experiment 1 - Model: None

### Embedding Size

16 - 93.71%, Epoch 2  
32 - 93.74%, Epoch 1  
64 - 93.76%, Epoch 2  
128 - 93.06%, Epoch 1  

## Experiment 2 - RNN Variations

### RNN Output False

#### 1 Bidirectional Layer

GRU - 36.14%, Epoch 0  
LSTM - 36.13%, Epoch 0  
RNN - 35.23%, Epoch 0  

#### 2 Layers

GRU -36.14%, Epoch 0  
LSTM - 36.14%, Epoch 0  
RNN - 35.60%, Epoch 1  

### RNN Output True

#### 1 Bidirectional Layer

GRU - 36.14%, Epoch 0  
LSTM - 36.13%, Epoch 0  
RNN - 35.23%, Epoch 0

#### 2 Layers

GRU -36.14%, Epoch 0  
LSTM - 36.14%, Epoch 0  
RNN - 35.60%, Epoch 1  

## Experiment 3 - More RNN Variations

### RNN Output False

#### Embedding - 16, Hidden - 8
GRU - 93.87%, Epoch 2  
LSTM - 93.91%, Epoch 2  
RNN - 92.97%, Epoch 1  

#### Embedding - 32, Hidden - 16
GRU - 93.92%, Epoch 1
LSTM - 93.89%, Epoch 1  
RNN - 92.56%, Epoch 2  

#### Embedding - 64, Hidden - 32
GRU - 94.11%, Epoch 2
LSTM - 94.11%, Epoch 2  
RNN - 91.52%, Epoch 4  

#### Embedding - 128, Hidden - 64
GRU - 94.03%, Epoch 1
LSTM - 94.09%, Epoch 2  
RNN - 91.84%, Epoch 1  

### RNN Output True

#### Embedding - 16, Hidden - 8
GRU - 93.38%, Epoch 1  
LSTM - 93.67%, Epoch 1  
RNN - 93.23%, Epoch 2  

#### Embedding - 32, Hidden - 16
GRU - 93.39%, Epoch 1
LSTM - 93.88%, Epoch 1  
RNN - 93.39%, Epoch 2  

#### Embedding - 64, Hidden - 32
GRU - 93.31%, Epoch 5
LSTM - 94.08%, Epoch 1  
RNN - 93.54%, Epoch 3  

#### Embedding - 128, Hidden - 64
GRU - 93.16%, Epoch 2
LSTM - 94.00%, Epoch 2  
RNN - 93.04%, Epoch 2  

## Experiment 4 - More RNN Variations
