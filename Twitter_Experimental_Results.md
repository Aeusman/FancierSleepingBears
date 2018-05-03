# Experimental Summary Results

Results are reported as the highest validation accuracy reached and on what epoch.

## Experiment 1 - Model: None

### Embedding Size

16 - 89.60%, Epoch 8  
32 - 91.93%, Epoch 7  
64 - 92.09%, Epoch 8  
128 - 92.79%, Epoch 6  

## Experiment 2 - RNN Variations

### RNN Output False

#### 1 Bidirectional Layer

GRU - 51.90%, Epoch 1  
LSTM - 54.15%, Epoch 6  
RNN - 52.44%, Epoch 6  

#### 2 Layers

GRU - 53.76%, Epoch 1  
LSTM - 54.23%, Epoch 6  
RNN - 50.04%, Epoch 6  

### RNN Output True

#### 1 Bidirectional Layer

GRU - NA  
LSTM - NA  
RNN - NA

#### 2 Layers

GRU - 90.69%, Epoch 7  
LSTM - 89.99%, Epoch 4  
RNN - 89.14%, Epoch 9  

## Experiment 3 - More RNN Variations

### RNN Output False

#### Embedding - 16, Hidden - 8
GRU - 89.84%, Epoch 9  
LSTM - 87.35%, Epoch 9  
RNN - 70.67%, Epoch 9  

#### Embedding - 32, Hidden - 16
GRU - 90.85%, Epoch 6  
LSTM - 90.15%, Epoch 6  
RNN - 87.90%, Epoch 6  

#### Embedding - 64, Hidden - 32
GRU - 91.00%, Epoch 8  
LSTM - 89.99%, Epoch 8  
RNN - 86.89%, Epoch 4  

#### Embedding - 128, Hidden - 64
GRU - 91.08%, Epoch 2  
LSTM - 91.08%, Epoch 9   
RNN - 87.43%, Epoch 3  

### RNN Output True

#### Embedding - 16, Hidden - 8
GRU - 89.60%, Epoch 4  
LSTM - 90.15%, Epoch 8  
RNN - 90.54%, Epoch 7  

#### Embedding - 32, Hidden - 16
GRU - 90.92%, Epoch 5  
LSTM - 90.46%, Epoch 6  
RNN - 89.84%, Epoch 7  

#### Embedding - 64, Hidden - 32
GRU - 91.54%, Epoch 2  
LSTM - 91.31%, Epoch 3  
RNN - 91.39%, Epoch 5  

#### Embedding - 128, Hidden - 64
GRU - 91.62%, Epoch 6  
LSTM - 91.08%, Epoch 4  
RNN - 90.54%, Epoch 4  

## Experiment 4 - More RNN Variations

### Embedding - 16

#### SeqCNN

##### 1 Layer
3 Wide Conv - 85.73%, Epoch 7  
4 Wide Conv - 86.97%, Epoch 8  
5 Wide Conv - 83.71%, Epoch 9  

##### 2 Layers
3 Wide Conv - 89.60%, Epoch 8  
4 Wide Conv - 86.27%, Epoch 8  
5 Wide Conv - 87.74%, Epoch 9  

##### 3 Layers
3 Wide Conv - 90.22%, Epoch 9  
4 Wide Conv - 84.79%, Epoch 9  
5 Wide Conv - 82.39%, Epoch 9  

#### ParCNN

##### Min Conv 1

3 Max Conv - 89.68%, Epoch 8  
4 Max Conv - 90.15%, Epoch 8  
5 Max Conv - 89.76%, Epoch 9  

##### Min Conv 2

3 Max Conv - 90.22%, Epoch 9  
4 Max Conv - 89.45%, Epoch 9  
5 Max Conv - 90.92%, Epoch 8  

### Embedding - 32

#### SeqCNN

##### 1 Layer
3 Wide Conv - 89.76%, Epoch 1  
4 Wide Conv - 89.29%, Epoch 7  
5 Wide Conv - 87.20%, Epoch 7  

##### 2 Layers
3 Wide Conv - 88.91%, Epoch 7  
4 Wide Conv - 90.77%, Epoch 7  
5 Wide Conv - 87.74%, Epoch 6  

##### 3 Layers
3 Wide Conv - 87.35%, Epoch 3  
4 Wide Conv - 86.42%, Epoch 4  
5 Wide Conv - 88.05%, Epoch 6  

#### ParCNN

##### Min Conv 1

3 Max Conv - 92.01%, Epoch 7  
4 Max Conv - 91.78%, Epoch 8  
5 Max Conv - 91.47%, Epoch 9  

##### Min Conv 2

3 Max Conv - 91.78%, Epoch 6  
4 Max Conv - 90.77%, Epoch 8  
5 Max Conv - 91.23%, Epoch 7  

### Embedding - 64

#### SeqCNN

##### 1 Layer
3 Wide Conv - 90.22%, Epoch 6  
4 Wide Conv - 89.68%, Epoch 6  
5 Wide Conv - 89.29%, Epoch 4  

##### 2 Layers
3 Wide Conv - 89.37%, Epoch 6  
4 Wide Conv - 86.42%, Epoch 7  
5 Wide Conv - 85.10%, Epoch 5  

##### 3 Layers
3 Wide Conv - 51.05%, Epoch 2  
4 Wide Conv - 83.40%, Epoch 8  
5 Wide Conv - 84.95%, Epoch 9  

#### ParCNN

##### Min Conv 1

3 Max Conv - 92.09%, Epoch 8  
4 Max Conv - 92.71%, Epoch 6  
5 Max Conv - 92.32%, Epoch 5  

##### Min Conv 2

3 Max Conv - 92.63%, Epoch 6  
4 Max Conv - 91.93%, Epoch 7  
5 Max Conv - 91.31%, Epoch 8  

### Embedding - 128

#### SeqCNN

##### 1 Layer
3 Wide Conv - 90.69%, Epoch 9  
4 Wide Conv - 90.46%, Epoch 7  
5 Wide Conv - 88.44%, Epoch 4  

##### 2 Layers
3 Wide Conv - 85.57%, Epoch 7  
4 Wide Conv - 77.27%, Epoch 8  
5 Wide Conv - 83.17%, Epoch 5  

##### 3 Layers
3 Wide Conv - 87.04%, Epoch 7   
4 Wide Conv - 87.20%, Epoch 9  
5 Wide Conv - 78.67%, Epoch 9  

#### ParCNN

##### Min Conv 1

3 Max Conv - 92.94%, Epoch 4  
4 Max Conv - 92.32%, Epoch 7  
5 Max Conv - 91.78%, Epoch 8  

##### Min Conv 2

3 Max Conv - 92.24%, Epoch 3  
4 Max Conv - 91.85%, Epoch 5  
5 Max Conv - 91.62%, Epoch 3  
