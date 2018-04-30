# Run experiments for Deep Learning

# Windows Setup
# SET LogDir=C:\Users\Lexa3_000\source\repos\PythonApplication1\PythonApplication1\FancierSleepingBears\ExperimentalLogs

# SET simple_rnn.py=C:\Users\Lexa3_000\source\repos\PythonApplication1\PythonApplication1\FancierSleepingBears\simple_rnn.py


#First Set of Experiments with no additional model
python %simple_rnn.py% --embed-size 16 --log_dir %LogDir%\Exp1_ModelNone_Em16 > %LogDir%\Exp1_ModelNone_Em16_Result.txt
python %simple_rnn.py% --embed-size 32 --log_dir %LogDir%\Exp1_ModelNone_Em32 > %LogDir%\Exp1_ModelNone_Em32_Result.txt
python %simple_rnn.py% --embed-size 64 --log_dir %LogDir%\Exp1_ModelNone_Em64 > %LogDir%\Exp1_ModelNone_Em64_Result.txt
python %simple_rnn.py% --embed-size 128 --log_dir %LogDir%\Exp1_ModelNone_Em128 > %LogDir%\Exp1_ModelNone_Em128_Result.txt

#Second set of experiments, different models and non-numeric hyper parameters
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --num-layers 2 --model rnn --log_dir %LogDir%\Exp2_RnOutTrue_2Lyrs_ModRNN > %LogDir%\Exp2_RnOutTrue_2Lyrs_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --num-layers 2 --model lstm --log_dir %LogDir%\Exp2_RnOutTrue_2Lyrs_ModLSTM > %LogDir%\Exp2_RnOutTrue_2Lyrs_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --num-layers 2 --model gru --log_dir %LogDir%\Exp2_RnOutTrue_2Lyrs_ModGRU > %LogDir%\Exp2_RnOutTrue_2Lyrs_ModGRU_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --bidirectional 1 --model rnn --log_dir %LogDir%\Exp2_RnOutTrue_1Bid_ModRNN > %LogDir%\Exp2_RnOutTrue_1Bid_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --bidirectional 1 --model lstm --log_dir %LogDir%\Exp2_RnOutTrue_1Bid_ModLSTM > %LogDir%\Exp2_RnOutTrue_1Bid_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --bidirectional 1 --model gru --log_dir %LogDir%\Exp2_RnOutTrue_1Bid_ModGRU > %LogDir%\Exp2_RnOutTrue_1Bid_ModGRU_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --num-layers 2 --model rnn --log_dir %LogDir%\Exp2_RnOutFalse_2Lyrs_ModRNN > %LogDir%\Exp2_RnOutFalse_2Lyrs_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --num-layers 2 --model lstm --log_dir %LogDir%\Exp2_RnOutFalse_2Lyrs_ModLSTM > %LogDir%\Exp2_RnOutFalse_2Lyrs_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --num-layers 2 --model gru --log_dir %LogDir%\Exp2_RnOutFalse_2Lyrs_ModGRU > %LogDir%\Exp2_RnOutFalse_2Lyrs_ModGRU_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --bidirectional 1 --model rnn --log_dir %LogDir%\Exp2_RnOutFalse_1Bid_ModRNN > %LogDir%\Exp2_RnOutFalse_1Bid_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --bidirectional 1 --model lstm --log_dir %LogDir%\Exp2_RnOutFalse_1Bid_ModLSTM > %LogDir%\Exp2_RnOutFalse_1Bid_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --bidirectional 1 --model gru --log_dir %LogDir%\Exp2_RnOutFalse_1Bid_ModGRU > %LogDir%\Exp2_RnOutFalse_1Bid_ModGRU_Result.txt

#Third set of experiments,
python %simple_rnn.py% --use-rnn-output 1 --embed-size 16 --hidden-size 8 --model rnn --log_dir %LogDir%\Exp3_RnOutTrue_Em16_Hid8_ModRNN > %LogDir%\Exp3_RnOutTrue_Em16_Hid8_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --model rnn --log_dir %LogDir%\Exp3_RnOutTrue_Em32_Hid16_ModRNN > %LogDir%\Exp3_RnOutTrue_Em32_Hid16_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 64 --hidden-size 32 --model rnn --log_dir %LogDir%\Exp3_RnOutTrue_Em64_Hid32_ModRNN > %LogDir%\Exp3_RnOutTrue_Em64_Hid32_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 128 --hidden-size 64 --model rnn --log_dir %LogDir%\Exp3_RnOutTrue_Em128_Hid64_ModRNN > %LogDir%\Exp3_RnOutTrue_Em128_Hid64_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 16 --hidden-size 8 --model rnn --log_dir %LogDir%\Exp3_RnOutFalse_Em16_Hid8_ModRNN > %LogDir%\Exp3_RnOutFalse_Em16_Hid8_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --model rnn --log_dir %LogDir%\Exp3_RnOutFalse_Em32_Hid16_ModRNN > %LogDir%\Exp3_RnOutFalse_Em32_Hid16_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 64 --hidden-size 32 --model rnn --log_dir %LogDir%\Exp3_RnOutFalse_Em64_Hid32_ModRNN > %LogDir%\Exp3_RnOutFalse_Em64_Hid32_ModRNN_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 128 --hidden-size 64 --model rnn --log_dir %LogDir%\Exp3_RnOutFalse_Em128_Hid64_ModRNN > %LogDir%\Exp3_RnOutFalse_Em128_Hid64_ModRNN_Result.txt

python %simple_rnn.py% --use-rnn-output 1 --embed-size 16 --hidden-size 8 --model gru --log_dir %LogDir%\Exp3_RnOutTrue_Em16_Hid8_ModGRU > %LogDir%\Exp3_RnOutTrue_Em16_Hid8_ModGRU_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --model gru --log_dir %LogDir%\Exp3_RnOutTrue_Em32_Hid16_ModGRU > %LogDir%\Exp3_RnOutTrue_Em32_Hid16_ModGRU_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 64 --hidden-size 32 --model gru --log_dir %LogDir%\Exp3_RnOutTrue_Em64_Hid32_ModGRU > %LogDir%\Exp3_RnOutTrue_Em64_Hid32_ModGRU_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 128 --hidden-size 64 --model gru --log_dir %LogDir%\Exp3_RnOutTrue_Em128_Hid64_ModGRU > %LogDir%\Exp3_RnOutTrue_Em128_Hid64_ModGRU_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 16 --hidden-size 8 --model gru --log_dir %LogDir%\Exp3_RnOutFalse_Em16_Hid8_ModGRU > %LogDir%\Exp3_RnOutFalse_Em16_Hid8_ModGRU_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --model gru --log_dir %LogDir%\Exp3_RnOutFalse_Em32_Hid16_ModGRU > %LogDir%\Exp3_RnOutFalse_Em32_Hid16_ModGRU_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 64 --hidden-size 32 --model gru --log_dir %LogDir%\Exp3_RnOutFalse_Em64_Hid32_ModGRU > %LogDir%\Exp3_RnOutFalse_Em64_Hid32_ModGRU_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 128 --hidden-size 64 --model gru --log_dir %LogDir%\Exp3_RnOutFalse_Em128_Hid64_ModGRU > %LogDir%\Exp3_RnOutFalse_Em128_Hid64_ModGRU_Result.txt

python %simple_rnn.py% --use-rnn-output 1 --embed-size 16 --hidden-size 8 --model lstm --log_dir %LogDir%\Exp3_RnOutTrue_Em16_Hid8_ModLSTM > %LogDir%\Exp3_RnOutTrue_Em16_Hid8_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --model lstm --log_dir %LogDir%\Exp3_RnOutTrue_Em32_Hid16_ModLSTM > %LogDir%\Exp3_RnOutTrue_Em32_Hid16_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 64 --hidden-size 32 --model lstm --log_dir %LogDir%\Exp3_RnOutTrue_Em64_Hid32_ModLSTM > %LogDir%\Exp3_RnOutTrue_Em64_Hid32_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 1 --embed-size 128 --hidden-size 64 --model lstm --log_dir %LogDir%\Exp3_RnOutTrue_Em128_Hid64_ModLSTM > %LogDir%\Exp3_RnOutTrue_Em128_Hid64_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 16 --hidden-size 8 --model lstm --log_dir %LogDir%\Exp3_RnOutFalse_Em16_Hid8_ModLSTM > %LogDir%\Exp3_RnOutFalse_Em16_Hid8_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --model lstm --log_dir %LogDir%\Exp3_RnOutFalse_Em32_Hid16_ModLSTM > %LogDir%\Exp3_RnOutFalse_Em32_Hid16_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 64 --hidden-size 32 --model lstm --log_dir %LogDir%\Exp3_RnOutFalse_Em64_Hid32_ModLSTM > %LogDir%\Exp3_RnOutFalse_Em64_Hid32_ModLSTM_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 128 --hidden-size 64 --model lstm --log_dir %LogDir%\Exp3_RnOutFalse_Em128_Hid64_ModLSTM > %LogDir%\Exp3_RnOutFalse_Em128_Hid64_ModLSTM_Result.txt
