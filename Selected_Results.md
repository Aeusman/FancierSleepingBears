# Selected Results From Experiments

## Twitter

Linear Model, Embedding 128: 92.79%
LSTM, Bidirectional layer, RNN Out False: 54.15%
GRU, 2 Layers, RNN Out True: 90.69%
GRU, Embedding 128, Hidden 64, RNN Out True: 91.62%
Sequential CNN, 1 Layer, 3 Conv. Kernel: 90.69%
Parallel CNN, Min Kernel 1, Max Kernel 4, Embedding 128: 92.94%

## Headlines

Linear Model, Embedding 64: 93.76%
LSTM, Bidirectional layer, RNN Output False: 36.13%
GRU, 2 Layers, RNN Out True: 36.14%
GRU, Embedding 64, Hidden 32, RNN Out False: 94.11%
Sequential CNN, 1 Layer, 3 Conv. Kernel: 93.12%
Parallel CNN, Min Kernel 1, Max Kernel 3, Embedding 32: 93.93%
