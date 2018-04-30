# Run experiments for Deep Learning

# Windows Setup
# SET LogDir=C:\Users\Lexa3_000\source\repos\PythonApplication1\PythonApplication1\FancierSleepingBears\ExperimentalLogs

# SET simple_rnn.py=C:\Users\Lexa3_000\source\repos\PythonApplication1\PythonApplication1\FancierSleepingBears\simple_rnn.py


#First Set of Experiments with no additional model
python %simple_rnn.py% --embed-size 16 --log_dir %LogDir%\Exp1_ModelNone_Em16
python %simple_rnn.py% --embed-size 32 --log_dir %LogDir%\Exp1_ModelNone_Em32
python %simple_rnn.py% --embed-size 64 --log_dir %LogDir%\Exp1_ModelNone_Em64
python %simple_rnn.py% --embed-size 128 --log_dir %LogDir%\Exp1_ModelNone_Em128

#Second set of experiments, different models and non-numeric hyper parameters
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --num-layers 2 --model rnn --log_dir %LogDir%\Exp2_RnOutTrue_2Lyrs_ModRNN
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --num-layers 2 --model lstm --log_dir %LogDir%\Exp2_RnOutTrue_2Lyrs_ModLSTM
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --num-layers 2 --model gru --log_dir %LogDir%\Exp2_RnOutTrue_2Lyrs_ModGRU
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --bidirectional 1 --model rnn --log_dir %LogDir%\Exp2_RnOutTrue_1Bid_ModRNN
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --bidirectional 1 --model lstm --log_dir %LogDir%\Exp2_RnOutTrue_1Bid_ModLSTM
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --bidirectional 1 --model gru --log_dir %LogDir%\Exp2_RnOutTrue_1Bid_ModGRU
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --num-layers 2 --model rnn --log_dir %LogDir%\Exp2_RnOutFalse_2Lyrs_ModRNN
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --num-layers 2 --model lstm --log_dir %LogDir%\Exp2_RnOutFalse_2Lyrs_ModLSTM
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --num-layers 2 --model gru --log_dir %LogDir%\Exp2_RnOutFalse_2Lyrs_ModGRU
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --bidirectional 1 --model rnn --log_dir %LogDir%\Exp2_RnOutFalse_1Bid_ModRNN
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --bidirectional 1 --model lstm --log_dir %LogDir%\Exp2_RnOutFalse_1Bid_ModLSTM
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --bidirectional 1 --model gru --log_dir %LogDir%\Exp2_RnOutFalse_1Bid_ModGRU

#Third set of experiments,
python %simple_rnn.py% --use-rnn-output 1 --embed-size 16 --hidden-size 8 --model rnn --log_dir %LogDir%\Exp3_RnOutTrue_Em16_Hid8_ModRNN
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --model rnn --log_dir %LogDir%\Exp3_RnOutTrue_Em32_Hid16_ModRNN
python %simple_rnn.py% --use-rnn-output 1 --embed-size 64 --hidden-size 32 --model rnn --log_dir %LogDir%\Exp3_RnOutTrue_Em64_Hid32_ModRNN
python %simple_rnn.py% --use-rnn-output 1 --embed-size 128 --hidden-size 64 --model rnn --log_dir %LogDir%\Exp3_RnOutTrue_Em128_Hid64_ModRNN
python %simple_rnn.py% --use-rnn-output 0 --embed-size 16 --hidden-size 8 --model rnn --log_dir %LogDir%\Exp3_RnOutFalse_Em16_Hid8_ModRNN
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --model rnn --log_dir %LogDir%\Exp3_RnOutFalse_Em32_Hid16_ModRNN
python %simple_rnn.py% --use-rnn-output 0 --embed-size 64 --hidden-size 32 --model rnn --log_dir %LogDir%\Exp3_RnOutFalse_Em64_Hid32_ModRNN
python %simple_rnn.py% --use-rnn-output 0 --embed-size 128 --hidden-size 64 --model rnn --log_dir %LogDir%\Exp3_RnOutFalse_Em128_Hid64_ModRNN

python %simple_rnn.py% --use-rnn-output 1 --embed-size 16 --hidden-size 8 --model gru --log_dir %LogDir%\Exp3_RnOutTrue_Em16_Hid8_ModGRU
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --model gru --log_dir %LogDir%\Exp3_RnOutTrue_Em32_Hid16_ModGRU
python %simple_rnn.py% --use-rnn-output 1 --embed-size 64 --hidden-size 32 --model gru --log_dir %LogDir%\Exp3_RnOutTrue_Em64_Hid32_ModGRU
python %simple_rnn.py% --use-rnn-output 1 --embed-size 128 --hidden-size 64 --model gru --log_dir %LogDir%\Exp3_RnOutTrue_Em128_Hid64_ModGRU
python %simple_rnn.py% --use-rnn-output 0 --embed-size 16 --hidden-size 8 --model gru --log_dir %LogDir%\Exp3_RnOutFalse_Em16_Hid8_ModGRU
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --model gru --log_dir %LogDir%\Exp3_RnOutFalse_Em32_Hid16_ModGRU
python %simple_rnn.py% --use-rnn-output 0 --embed-size 64 --hidden-size 32 --model gru --log_dir %LogDir%\Exp3_RnOutFalse_Em64_Hid32_ModGRU
python %simple_rnn.py% --use-rnn-output 0 --embed-size 128 --hidden-size 64 --model gru --log_dir %LogDir%\Exp3_RnOutFalse_Em128_Hid64_ModGRU

python %simple_rnn.py% --use-rnn-output 1 --embed-size 16 --hidden-size 8 --model lstm --log_dir %LogDir%\Exp3_RnOutTrue_Em16_Hid8_ModLSTM
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --model lstm --log_dir %LogDir%\Exp3_RnOutTrue_Em32_Hid16_ModLSTM
python %simple_rnn.py% --use-rnn-output 1 --embed-size 64 --hidden-size 32 --model lstm --log_dir %LogDir%\Exp3_RnOutTrue_Em64_Hid32_ModLSTM
python %simple_rnn.py% --use-rnn-output 1 --embed-size 128 --hidden-size 64 --model lstm --log_dir %LogDir%\Exp3_RnOutTrue_Em128_Hid64_ModLSTM
python %simple_rnn.py% --use-rnn-output 0 --embed-size 16 --hidden-size 8 --model lstm --log_dir %LogDir%\Exp3_RnOutFalse_Em16_Hid8_ModLSTM
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --model lstm --log_dir %LogDir%\Exp3_RnOutFalse_Em32_Hid16_ModLSTM
python %simple_rnn.py% --use-rnn-output 0 --embed-size 64 --hidden-size 32 --model lstm --log_dir %LogDir%\Exp3_RnOutFalse_Em64_Hid32_ModLSTM
python %simple_rnn.py% --use-rnn-output 0 --embed-size 128 --hidden-size 64 --model lstm --log_dir %LogDir%\Exp3_RnOutFalse_Em128_Hid64_ModLSTM
