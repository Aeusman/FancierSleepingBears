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

### Embedding - 16

#### SeqCNN

##### 1 Layer
3 Wide Conv - 92.27%, Epoch 1  
4 Wide Conv - 92.82%, Epoch 1  
5 Wide Conv - 91.88%, Epoch 2  

##### 2 Layers
3 Wide Conv - 92.07%, Epoch 1  
4 Wide Conv - 92.56%, Epoch 1  
5 Wide Conv - 92.27%, Epoch 3  

##### 3 Layers
3 Wide Conv - 92.37%, Epoch 2  
4 Wide Conv - 93.01%, Epoch 3  
5 Wide Conv - 92.10%, Epoch 4  

#### ParCNN

##### Min Conv 1

3 Max Conv - 93.72%, Epoch 1  
4 Max Conv - 93.80%, Epoch 1  
5 Max Conv - 93.87%, Epoch 2  

##### Min Conv 2

3 Max Conv - 93.72%, Epoch 1  
4 Max Conv - 93.85%, Epoch 1  
5 Max Conv - 93.76%, Epoch 2  

### Embedding - 32

#### SeqCNN

##### 1 Layer
3 Wide Conv - 93.12%, Epoch 1  
4 Wide Conv - 92.62%, Epoch 1  
5 Wide Conv - 92.52%, Epoch 1  

##### 2 Layers
3 Wide Conv - 92.35%, Epoch 1  
4 Wide Conv - 92.85%, Epoch 2  
5 Wide Conv - 92.76%, Epoch 3  

##### 3 Layers
3 Wide Conv - 92.49%, Epoch 2  
4 Wide Conv - 93.29%, Epoch 2  
5 Wide Conv - 91.88%, Epoch 3  

#### ParCNN

##### Min Conv 1

3 Max Conv - 93.93%, Epoch 1  
4 Max Conv - 93.87%, Epoch 1  
5 Max Conv - 93.80%, Epoch 2  

##### Min Conv 2

3 Max Conv - 93.88%, Epoch 1  
4 Max Conv - 93.83%, Epoch 1  
5 Max Conv - 93.87%, Epoch 2  

### Embedding - 64

#### SeqCNN

##### 1 Layer
3 Wide Conv - 93.08%, Epoch 2  
4 Wide Conv - 93.05%, Epoch 1  
5 Wide Conv - 92.67%, Epoch 2  

##### 2 Layers
3 Wide Conv - 92.80%, Epoch 3  
4 Wide Conv - 93.13%, Epoch 1  
5 Wide Conv - 92.79%, Epoch 2  

##### 3 Layers
3 Wide Conv - 92.23%, Epoch 7  
4 Wide Conv - 92.43%, Epoch 2  
5 Wide Conv - 92.26%, Epoch 3  

#### ParCNN

##### Min Conv 1

3 Max Conv - 93.51%, Epoch 2  
4 Max Conv - 93.36%, Epoch 1  
5 Max Conv - 93.19%, Epoch 2  

##### Min Conv 2

3 Max Conv - 93.47%, Epoch 2  
4 Max Conv - 93.54%, Epoch 2  
5 Max Conv - 93.56%, Epoch 1  

### Embedding - 128

#### SeqCNN

##### 1 Layer
3 Wide Conv - 93.03%, Epoch 1  
4 Wide Conv - 92.72%, Epoch 1  
5 Wide Conv - 92.54%, Epoch 1  

##### 2 Layers
3 Wide Conv - 92.92%, Epoch 1  
4 Wide Conv - 93.27%, Epoch 1  
5 Wide Conv - 92.95%, Epoch 1  

##### 3 Layers
3 Wide Conv - 92.44%, Epoch 4   
4 Wide Conv - 92.35%, Epoch 2  
5 Wide Conv - 89.78%, Epoch 1  

#### ParCNN

##### Min Conv 1

3 Max Conv - 93.06%, Epoch 1  
4 Max Conv - 92.89%, Epoch 1  
5 Max Conv - 92.96%, Epoch 1  

##### Min Conv 2

3 Max Conv - 93.32%, Epoch 1  
4 Max Conv - 93.11%, Epoch 1  
5 Max Conv - 92.90%, Epoch 1  
