# Run experiments for Deep Learning

# Windows Setup
SET LogDir=C:\Users\Lexa3_000\source\repos\PythonApplication1\PythonApplication1\FancierSleepingBears\ExperimentalLogs
SET simple_rnn.py=C:\Users\Lexa3_000\source\repos\PythonApplication1\PythonApplication1\FancierSleepingBears\simple_rnn.py
SET twitter.py=C:\Users\Lexa3_000\source\repos\PythonApplication1\PythonApplication1\FancierSleepingBears\twitter.py

python %simple_rnn.py% --embed-size 64 --name Headlines_Exp1_ModelNone_Em64 > Headlines_Exp1_ModelNone_Em64_Result.txt
python %simple_rnn.py% --use-rnn-output 0 --embed-size 64 --hidden-size 32 --model gru --name Headlines_Exp3_RnOutFalse_Em64_Hid32_ModGRU > Headlines_Exp3_RnOutFalse_Em64_Hid32_ModGRU_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 3 --embed-size 32 --name Headlines_Exp4_Em32_Min1_Max3_ModParC > Headlines_Exp4_Em32_Min1_Max3_ModParC_Result.txt

python %twitter.py% --embed-size 128 --name Twitter_Exp1_ModelNone_Em128 > Twitter_Exp1_ModelNone_Em128_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 64 --hidden-size 32 --model gru --name Twitter_Exp3_RnOutTrue_Em64_Hid32_ModGRU > Twitter_Exp3_RnOutTrue_Em64_Hid32_ModGRU_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 4 --embed-size 64 --name Twitter_Exp4_Em64_Min1_Max4_ModParC > Twitter_Exp4_Em64_Min1_Max4_ModParC_Result.txt








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

#Fourth set of experiments, convolutional networks specifically
python %simple_rnn.py% --model seqcnn --embed-size 16 --num-layers 1 --conv-size 3 --log_dir %LogDir%\Exp4_Em16_Lay1_Conv3_ModSeqC > %LogDir%\Exp4_Em16_Lay1_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 16 --num-layers 1 --conv-size 4 --log_dir %LogDir%\Exp4_Em16_Lay1_Conv4_ModSeqC > %LogDir%\Exp4_Em16_Lay1_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 16 --num-layers 1 --conv-size 5 --log_dir %LogDir%\Exp4_Em16_Lay1_Conv5_ModSeqC > %LogDir%\Exp4_Em16_Lay1_Conv5_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 16 --num-layers 2 --conv-size 3 --log_dir %LogDir%\Exp4_Em16_Lay2_Conv3_ModSeqC > %LogDir%\Exp4_Em16_Lay2_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 16 --num-layers 2 --conv-size 4 --log_dir %LogDir%\Exp4_Em16_Lay2_Conv4_ModSeqC > %LogDir%\Exp4_Em16_Lay2_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 16 --num-layers 2 --conv-size 5 --log_dir %LogDir%\Exp4_Em16_Lay2_Conv5_ModSeqC > %LogDir%\Exp4_Em16_Lay2_Conv5_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 16 --num-layers 3 --conv-size 3 --log_dir %LogDir%\Exp4_Em16_Lay3_Conv3_ModSeqC > %LogDir%\Exp4_Em16_Lay3_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 16 --num-layers 3 --conv-size 4 --log_dir %LogDir%\Exp4_Em16_Lay3_Conv4_ModSeqC > %LogDir%\Exp4_Em16_Lay3_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 16 --num-layers 3 --conv-size 5 --log_dir %LogDir%\Exp4_Em16_Lay3_Conv5_ModSeqC > %LogDir%\Exp4_Em16_Lay3_Conv5_ModSeqC_Result.txt

python %simple_rnn.py% --model seqcnn --embed-size 32 --num-layers 1 --conv-size 3 --log_dir %LogDir%\Exp4_Em32_Lay1_Conv3_ModSeqC > %LogDir%\Exp4_Em32_Lay1_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 32 --num-layers 1 --conv-size 4 --log_dir %LogDir%\Exp4_Em32_Lay1_Conv4_ModSeqC > %LogDir%\Exp4_Em32_Lay1_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 32 --num-layers 1 --conv-size 5 --log_dir %LogDir%\Exp4_Em32_Lay1_Conv5_ModSeqC > %LogDir%\Exp4_Em32_Lay1_Conv5_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 32 --num-layers 2 --conv-size 3 --log_dir %LogDir%\Exp4_Em32_Lay2_Conv3_ModSeqC > %LogDir%\Exp4_Em32_Lay2_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 32 --num-layers 2 --conv-size 4 --log_dir %LogDir%\Exp4_Em32_Lay2_Conv4_ModSeqC > %LogDir%\Exp4_Em32_Lay2_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 32 --num-layers 2 --conv-size 5 --log_dir %LogDir%\Exp4_Em32_Lay2_Conv5_ModSeqC > %LogDir%\Exp4_Em32_Lay2_Conv5_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 32 --num-layers 3 --conv-size 3 --log_dir %LogDir%\Exp4_Em32_Lay3_Conv3_ModSeqC > %LogDir%\Exp4_Em32_Lay3_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 32 --num-layers 3 --conv-size 4 --log_dir %LogDir%\Exp4_Em32_Lay3_Conv4_ModSeqC > %LogDir%\Exp4_Em32_Lay3_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 32 --num-layers 3 --conv-size 5 --log_dir %LogDir%\Exp4_Em32_Lay3_Conv5_ModSeqC > %LogDir%\Exp4_Em32_Lay3_Conv5_ModSeqC_Result.txt

python %simple_rnn.py% --model seqcnn --embed-size 64 --num-layers 1 --conv-size 3 --log_dir %LogDir%\Exp4_Em64_Lay1_Conv3_ModSeqC > %LogDir%\Exp4_Em64_Lay1_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 64 --num-layers 1 --conv-size 4 --log_dir %LogDir%\Exp4_Em64_Lay1_Conv4_ModSeqC > %LogDir%\Exp4_Em64_Lay1_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 64 --num-layers 1 --conv-size 5 --log_dir %LogDir%\Exp4_Em64_Lay1_Conv5_ModSeqC > %LogDir%\Exp4_Em64_Lay1_Conv5_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 64 --num-layers 2 --conv-size 3 --log_dir %LogDir%\Exp4_Em64_Lay2_Conv3_ModSeqC > %LogDir%\Exp4_Em64_Lay2_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 64 --num-layers 2 --conv-size 4 --log_dir %LogDir%\Exp4_Em64_Lay2_Conv4_ModSeqC > %LogDir%\Exp4_Em64_Lay2_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 64 --num-layers 2 --conv-size 5 --log_dir %LogDir%\Exp4_Em64_Lay2_Conv5_ModSeqC > %LogDir%\Exp4_Em64_Lay2_Conv5_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 64 --num-layers 3 --conv-size 3 --log_dir %LogDir%\Exp4_Em64_Lay3_Conv3_ModSeqC > %LogDir%\Exp4_Em64_Lay3_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 64 --num-layers 3 --conv-size 4 --log_dir %LogDir%\Exp4_Em64_Lay3_Conv4_ModSeqC > %LogDir%\Exp4_Em64_Lay3_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 64 --num-layers 3 --conv-size 5 --log_dir %LogDir%\Exp4_Em64_Lay3_Conv5_ModSeqC > %LogDir%\Exp4_Em64_Lay3_Conv5_ModSeqC_Result.txt

python %simple_rnn.py% --model seqcnn --embed-size 128 --num-layers 1 --conv-size 3 --log_dir %LogDir%\Exp4_Em128_Lay1_Conv3_ModSeqC > %LogDir%\Exp4_Em128_Lay1_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 128 --num-layers 1 --conv-size 4 --log_dir %LogDir%\Exp4_Em128_Lay1_Conv4_ModSeqC > %LogDir%\Exp4_Em128_Lay1_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 128 --num-layers 1 --conv-size 5 --log_dir %LogDir%\Exp4_Em128_Lay1_Conv5_ModSeqC > %LogDir%\Exp4_Em128_Lay1_Conv5_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 128 --num-layers 2 --conv-size 3 --log_dir %LogDir%\Exp4_Em128_Lay2_Conv3_ModSeqC > %LogDir%\Exp4_Em128_Lay2_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 128 --num-layers 2 --conv-size 4 --log_dir %LogDir%\Exp4_Em128_Lay2_Conv4_ModSeqC > %LogDir%\Exp4_Em128_Lay2_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 128 --num-layers 2 --conv-size 5 --log_dir %LogDir%\Exp4_Em128_Lay2_Conv5_ModSeqC > %LogDir%\Exp4_Em128_Lay2_Conv5_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 128 --num-layers 3 --conv-size 3 --log_dir %LogDir%\Exp4_Em128_Lay3_Conv3_ModSeqC > %LogDir%\Exp4_Em128_Lay3_Conv3_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 128 --num-layers 3 --conv-size 4 --log_dir %LogDir%\Exp4_Em128_Lay3_Conv4_ModSeqC > %LogDir%\Exp4_Em128_Lay3_Conv4_ModSeqC_Result.txt
python %simple_rnn.py% --model seqcnn --embed-size 128 --num-layers 3 --conv-size 5 --log_dir %LogDir%\Exp4_Em128_Lay3_Conv5_ModSeqC > %LogDir%\Exp4_Em128_Lay3_Conv5_ModSeqC_Result.txt

# Now look at parcnn
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 3 --embed-size 16 --log_dir %LogDir%\Exp4_Em16_Min1_Max3_ModParC > %LogDir%\Exp4_Em16_Min1_Max3_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 3 --embed-size 32 --log_dir %LogDir%\Exp4_Em32_Min1_Max3_ModParC > %LogDir%\Exp4_Em32_Min1_Max3_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 3 --embed-size 64 --log_dir %LogDir%\Exp4_Em64_Min1_Max3_ModParC > %LogDir%\Exp4_Em64_Min1_Max3_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 3 --embed-size 128 --log_dir %LogDir%\Exp4_Em128_Min1_Max3_ModParC > %LogDir%\Exp4_Em128_Min1_Max3_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 4 --embed-size 16 --log_dir %LogDir%\Exp4_Em16_Min1_Max4_ModParC > %LogDir%\Exp4_Em16_Min1_Max4_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 4 --embed-size 32 --log_dir %LogDir%\Exp4_Em32_Min1_Max4_ModParC > %LogDir%\Exp4_Em32_Min1_Max4_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 4 --embed-size 64 --log_dir %LogDir%\Exp4_Em64_Min1_Max4_ModParC > %LogDir%\Exp4_Em64_Min1_Max4_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 4 --embed-size 128 --log_dir %LogDir%\Exp4_Em128_Min1_Max4_ModParC > %LogDir%\Exp4_Em128_Min1_Max4_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 5 --embed-size 16 --log_dir %LogDir%\Exp4_Em16_Min1_Max5_ModParC > %LogDir%\Exp4_Em16_Min1_Max5_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 5 --embed-size 32 --log_dir %LogDir%\Exp4_Em32_Min1_Max5_ModParC > %LogDir%\Exp4_Em32_Min1_Max5_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 5 --embed-size 64 --log_dir %LogDir%\Exp4_Em64_Min1_Max5_ModParC > %LogDir%\Exp4_Em64_Min1_Max5_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 1 --large-conv-size 5 --embed-size 128 --log_dir %LogDir%\Exp4_Em128_Min1_Max5_ModParC > %LogDir%\Exp4_Em128_Min1_Max5_ModParC_Result.txt

python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 3 --embed-size 16 --log_dir %LogDir%\Exp4_Em16_Min2_Max3_ModParC > %LogDir%\Exp4_Em16_Min2_Max3_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 3 --embed-size 32 --log_dir %LogDir%\Exp4_Em32_Min2_Max3_ModParC > %LogDir%\Exp4_Em32_Min2_Max3_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 3 --embed-size 64 --log_dir %LogDir%\Exp4_Em64_Min2_Max3_ModParC > %LogDir%\Exp4_Em64_Min2_Max3_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 3 --embed-size 128 --log_dir %LogDir%\Exp4_Em128_Min2_Max3_ModParC > %LogDir%\Exp4_Em128_Min2_Max3_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 4 --embed-size 16 --log_dir %LogDir%\Exp4_Em16_Min2_Max4_ModParC > %LogDir%\Exp4_Em16_Min2_Max4_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 4 --embed-size 32 --log_dir %LogDir%\Exp4_Em32_Min2_Max4_ModParC > %LogDir%\Exp4_Em32_Min2_Max4_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 4 --embed-size 64 --log_dir %LogDir%\Exp4_Em64_Min2_Max4_ModParC > %LogDir%\Exp4_Em64_Min2_Max4_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 4 --embed-size 128 --log_dir %LogDir%\Exp4_Em128_Min2_Max4_ModParC > %LogDir%\Exp4_Em128_Min2_Max4_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 5 --embed-size 16 --log_dir %LogDir%\Exp4_Em16_Min2_Max5_ModParC > %LogDir%\Exp4_Em16_Min2_Max5_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 5 --embed-size 32 --log_dir %LogDir%\Exp4_Em32_Min2_Max5_ModParC > %LogDir%\Exp4_Em32_Min2_Max5_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 5 --embed-size 64 --log_dir %LogDir%\Exp4_Em64_Min2_Max5_ModParC > %LogDir%\Exp4_Em64_Min2_Max5_ModParC_Result.txt
python %simple_rnn.py% --model parcnn --small-conv-size 2 --large-conv-size 5 --embed-size 128 --log_dir %LogDir%\Exp4_Em128_Min2_Max5_ModParC > %LogDir%\Exp4_Em128_Min2_Max5_ModParC_Result.txt

#Twitter Data
#First Set of twitter Experiments with no additional model
python %twitter.py% --embed-size 16 --log_dir %LogDir%\Twitter_Exp1_ModelNone_Em16 > %LogDir%\Twitter_Exp1_ModelNone_Em16_Result.txt
python %twitter.py% --embed-size 32 --log_dir %LogDir%\Twitter_Exp1_ModelNone_Em32 > %LogDir%\Twitter_Exp1_ModelNone_Em32_Result.txt
python %twitter.py% --embed-size 64 --log_dir %LogDir%\Twitter_Exp1_ModelNone_Em64 > %LogDir%\Twitter_Exp1_ModelNone_Em64_Result.txt
python %twitter.py% --embed-size 128 --log_dir %LogDir%\Twitter_Exp1_ModelNone_Em128 > %LogDir%\Twitter_Exp1_ModelNone_Em128_Result.txt

#Second set of experiments, different models and non-numeric hyper parameters
python %twitter.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --num-layers 2 --model rnn --log_dir %LogDir%\Twitter_Exp2_RnOutTrue_2Lyrs_ModRNN > %LogDir%\Twitter_Exp2_RnOutTrue_2Lyrs_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --num-layers 2 --model lstm --log_dir %LogDir%\Twitter_Exp2_RnOutTrue_2Lyrs_ModLSTM > %LogDir%\Twitter_Exp2_RnOutTrue_2Lyrs_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --num-layers 2 --model gru --log_dir %LogDir%\Twitter_Exp2_RnOutTrue_2Lyrs_ModGRU > %LogDir%\Twitter_Exp2_RnOutTrue_2Lyrs_ModGRU_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --bidirectional 1 --model rnn --log_dir %LogDir%\Twitter_Exp2_RnOutTrue_1Bid_ModRNN > %LogDir%\Twitter_Exp2_RnOutTrue_1Bid_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --bidirectional 1 --model lstm --log_dir %LogDir%\Twitter_Exp2_RnOutTrue_1Bid_ModLSTM > %LogDir%\Twitter_Exp2_RnOutTrue_1Bid_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --bidirectional 1 --model gru --log_dir %LogDir%\Twitter_Exp2_RnOutTrue_1Bid_ModGRU > %LogDir%\Twitter_Exp2_RnOutTrue_1Bid_ModGRU_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --num-layers 2 --model rnn --log_dir %LogDir%\Twitter_Exp2_RnOutFalse_2Lyrs_ModRNN > %LogDir%\Twitter_Exp2_RnOutFalse_2Lyrs_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --num-layers 2 --model lstm --log_dir %LogDir%\Twitter_Exp2_RnOutFalse_2Lyrs_ModLSTM > %LogDir%\Twitter_Exp2_RnOutFalse_2Lyrs_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --num-layers 2 --model gru --log_dir %LogDir%\Twitter_Exp2_RnOutFalse_2Lyrs_ModGRU > %LogDir%\Twitter_Exp2_RnOutFalse_2Lyrs_ModGRU_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --bidirectional 1 --model rnn --log_dir %LogDir%\Twitter_Exp2_RnOutFalse_1Bid_ModRNN > %LogDir%\Twitter_Exp2_RnOutFalse_1Bid_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --bidirectional 1 --model lstm --log_dir %LogDir%\Twitter_Exp2_RnOutFalse_1Bid_ModLSTM > %LogDir%\Twitter_Exp2_RnOutFalse_1Bid_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --bidirectional 1 --model gru --log_dir %LogDir%\Twitter_Exp2_RnOutFalse_1Bid_ModGRU > %LogDir%\Twitter_Exp2_RnOutFalse_1Bid_ModGRU_Result.txt

#Third set of experiments,
python %twitter.py% --use-rnn-output 1 --embed-size 16 --hidden-size 8 --model rnn --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em16_Hid8_ModRNN > %LogDir%\Twitter_Exp3_RnOutTrue_Em16_Hid8_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --model rnn --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em32_Hid16_ModRNN > %LogDir%\Twitter_Exp3_RnOutTrue_Em32_Hid16_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 64 --hidden-size 32 --model rnn --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em64_Hid32_ModRNN > %LogDir%\Twitter_Exp3_RnOutTrue_Em64_Hid32_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 128 --hidden-size 64 --model rnn --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em128_Hid64_ModRNN > %LogDir%\Twitter_Exp3_RnOutTrue_Em128_Hid64_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 16 --hidden-size 8 --model rnn --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em16_Hid8_ModRNN > %LogDir%\Twitter_Exp3_RnOutFalse_Em16_Hid8_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --model rnn --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em32_Hid16_ModRNN > %LogDir%\Twitter_Exp3_RnOutFalse_Em32_Hid16_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 64 --hidden-size 32 --model rnn --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em64_Hid32_ModRNN > %LogDir%\Twitter_Exp3_RnOutFalse_Em64_Hid32_ModRNN_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 128 --hidden-size 64 --model rnn --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em128_Hid64_ModRNN > %LogDir%\Twitter_Exp3_RnOutFalse_Em128_Hid64_ModRNN_Result.txt

python %twitter.py% --use-rnn-output 1 --embed-size 16 --hidden-size 8 --model gru --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em16_Hid8_ModGRU > %LogDir%\Twitter_Exp3_RnOutTrue_Em16_Hid8_ModGRU_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --model gru --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em32_Hid16_ModGRU > %LogDir%\Twitter_Exp3_RnOutTrue_Em32_Hid16_ModGRU_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 64 --hidden-size 32 --model gru --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em64_Hid32_ModGRU > %LogDir%\Twitter_Exp3_RnOutTrue_Em64_Hid32_ModGRU_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 128 --hidden-size 64 --model gru --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em128_Hid64_ModGRU > %LogDir%\Twitter_Exp3_RnOutTrue_Em128_Hid64_ModGRU_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 16 --hidden-size 8 --model gru --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em16_Hid8_ModGRU > %LogDir%\Twitter_Exp3_RnOutFalse_Em16_Hid8_ModGRU_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --model gru --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em32_Hid16_ModGRU > %LogDir%\Twitter_Exp3_RnOutFalse_Em32_Hid16_ModGRU_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 64 --hidden-size 32 --model gru --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em64_Hid32_ModGRU > %LogDir%\Twitter_Exp3_RnOutFalse_Em64_Hid32_ModGRU_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 128 --hidden-size 64 --model gru --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em128_Hid64_ModGRU > %LogDir%\Twitter_Exp3_RnOutFalse_Em128_Hid64_ModGRU_Result.txt

python %twitter.py% --use-rnn-output 1 --embed-size 16 --hidden-size 8 --model lstm --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em16_Hid8_ModLSTM > %LogDir%\Twitter_Exp3_RnOutTrue_Em16_Hid8_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 32 --hidden-size 16 --model lstm --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em32_Hid16_ModLSTM > %LogDir%\Twitter_Exp3_RnOutTrue_Em32_Hid16_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 64 --hidden-size 32 --model lstm --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em64_Hid32_ModLSTM > %LogDir%\Twitter_Exp3_RnOutTrue_Em64_Hid32_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 1 --embed-size 128 --hidden-size 64 --model lstm --log_dir %LogDir%\Twitter_Exp3_RnOutTrue_Em128_Hid64_ModLSTM > %LogDir%\Twitter_Exp3_RnOutTrue_Em128_Hid64_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 16 --hidden-size 8 --model lstm --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em16_Hid8_ModLSTM > %LogDir%\Twitter_Exp3_RnOutFalse_Em16_Hid8_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 32 --hidden-size 16 --model lstm --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em32_Hid16_ModLSTM > %LogDir%\Twitter_Exp3_RnOutFalse_Em32_Hid16_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 64 --hidden-size 32 --model lstm --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em64_Hid32_ModLSTM > %LogDir%\Twitter_Exp3_RnOutFalse_Em64_Hid32_ModLSTM_Result.txt
python %twitter.py% --use-rnn-output 0 --embed-size 128 --hidden-size 64 --model lstm --log_dir %LogDir%\Twitter_Exp3_RnOutFalse_Em128_Hid64_ModLSTM > %LogDir%\Twitter_Exp3_RnOutFalse_Em128_Hid64_ModLSTM_Result.txt

#Fourth set of experiments, convolutional networks specifically
python %twitter.py% --model seqcnn --embed-size 16 --num-layers 1 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em16_Lay1_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em16_Lay1_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 16 --num-layers 1 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em16_Lay1_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em16_Lay1_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 16 --num-layers 1 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em16_Lay1_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em16_Lay1_Conv5_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 16 --num-layers 2 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em16_Lay2_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em16_Lay2_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 16 --num-layers 2 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em16_Lay2_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em16_Lay2_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 16 --num-layers 2 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em16_Lay2_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em16_Lay2_Conv5_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 16 --num-layers 3 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em16_Lay3_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em16_Lay3_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 16 --num-layers 3 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em16_Lay3_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em16_Lay3_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 16 --num-layers 3 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em16_Lay3_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em16_Lay3_Conv5_ModSeqC_Result.txt

python %twitter.py% --model seqcnn --embed-size 32 --num-layers 1 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em32_Lay1_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em32_Lay1_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 32 --num-layers 1 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em32_Lay1_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em32_Lay1_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 32 --num-layers 1 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em32_Lay1_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em32_Lay1_Conv5_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 32 --num-layers 2 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em32_Lay2_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em32_Lay2_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 32 --num-layers 2 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em32_Lay2_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em32_Lay2_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 32 --num-layers 2 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em32_Lay2_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em32_Lay2_Conv5_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 32 --num-layers 3 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em32_Lay3_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em32_Lay3_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 32 --num-layers 3 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em32_Lay3_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em32_Lay3_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 32 --num-layers 3 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em32_Lay3_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em32_Lay3_Conv5_ModSeqC_Result.txt

python %twitter.py% --model seqcnn --embed-size 64 --num-layers 1 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em64_Lay1_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em64_Lay1_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 64 --num-layers 1 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em64_Lay1_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em64_Lay1_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 64 --num-layers 1 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em64_Lay1_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em64_Lay1_Conv5_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 64 --num-layers 2 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em64_Lay2_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em64_Lay2_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 64 --num-layers 2 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em64_Lay2_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em64_Lay2_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 64 --num-layers 2 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em64_Lay2_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em64_Lay2_Conv5_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 64 --num-layers 3 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em64_Lay3_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em64_Lay3_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 64 --num-layers 3 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em64_Lay3_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em64_Lay3_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 64 --num-layers 3 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em64_Lay3_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em64_Lay3_Conv5_ModSeqC_Result.txt

python %twitter.py% --model seqcnn --embed-size 128 --num-layers 1 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em128_Lay1_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em128_Lay1_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 128 --num-layers 1 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em128_Lay1_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em128_Lay1_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 128 --num-layers 1 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em128_Lay1_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em128_Lay1_Conv5_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 128 --num-layers 2 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em128_Lay2_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em128_Lay2_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 128 --num-layers 2 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em128_Lay2_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em128_Lay2_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 128 --num-layers 2 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em128_Lay2_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em128_Lay2_Conv5_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 128 --num-layers 3 --conv-size 3 --log_dir %LogDir%\Twitter_Exp4_Em128_Lay3_Conv3_ModSeqC > %LogDir%\Twitter_Exp4_Em128_Lay3_Conv3_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 128 --num-layers 3 --conv-size 4 --log_dir %LogDir%\Twitter_Exp4_Em128_Lay3_Conv4_ModSeqC > %LogDir%\Twitter_Exp4_Em128_Lay3_Conv4_ModSeqC_Result.txt
python %twitter.py% --model seqcnn --embed-size 128 --num-layers 3 --conv-size 5 --log_dir %LogDir%\Twitter_Exp4_Em128_Lay3_Conv5_ModSeqC > %LogDir%\Twitter_Exp4_Em128_Lay3_Conv5_ModSeqC_Result.txt

# Now look at parcnn
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 3 --embed-size 16 --log_dir %LogDir%\Twitter_Exp4_Em16_Min1_Max3_ModParC > %LogDir%\Twitter_Exp4_Em16_Min1_Max3_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 3 --embed-size 32 --log_dir %LogDir%\Twitter_Exp4_Em32_Min1_Max3_ModParC > %LogDir%\Twitter_Exp4_Em32_Min1_Max3_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 3 --embed-size 64 --log_dir %LogDir%\Twitter_Exp4_Em64_Min1_Max3_ModParC > %LogDir%\Twitter_Exp4_Em64_Min1_Max3_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 3 --embed-size 128 --log_dir %LogDir%\Twitter_Exp4_Em128_Min1_Max3_ModParC > %LogDir%\Twitter_Exp4_Em128_Min1_Max3_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 4 --embed-size 16 --log_dir %LogDir%\Twitter_Exp4_Em16_Min1_Max4_ModParC > %LogDir%\Twitter_Exp4_Em16_Min1_Max4_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 4 --embed-size 32 --log_dir %LogDir%\Twitter_Exp4_Em32_Min1_Max4_ModParC > %LogDir%\Twitter_Exp4_Em32_Min1_Max4_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 4 --embed-size 64 --log_dir %LogDir%\Twitter_Exp4_Em64_Min1_Max4_ModParC > %LogDir%\Twitter_Exp4_Em64_Min1_Max4_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 4 --embed-size 128 --log_dir %LogDir%\Twitter_Exp4_Em128_Min1_Max4_ModParC > %LogDir%\Twitter_Exp4_Em128_Min1_Max4_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 5 --embed-size 16 --log_dir %LogDir%\Twitter_Exp4_Em16_Min1_Max5_ModParC > %LogDir%\Twitter_Exp4_Em16_Min1_Max5_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 5 --embed-size 32 --log_dir %LogDir%\Twitter_Exp4_Em32_Min1_Max5_ModParC > %LogDir%\Twitter_Exp4_Em32_Min1_Max5_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 5 --embed-size 64 --log_dir %LogDir%\Twitter_Exp4_Em64_Min1_Max5_ModParC > %LogDir%\Twitter_Exp4_Em64_Min1_Max5_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 1 --large-conv-size 5 --embed-size 128 --log_dir %LogDir%\Twitter_Exp4_Em128_Min1_Max5_ModParC > %LogDir%\Twitter_Exp4_Em128_Min1_Max5_ModParC_Result.txt

python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 3 --embed-size 16 --log_dir %LogDir%\Twitter_Exp4_Em16_Min2_Max3_ModParC > %LogDir%\Twitter_Exp4_Em16_Min2_Max3_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 3 --embed-size 32 --log_dir %LogDir%\Twitter_Exp4_Em32_Min2_Max3_ModParC > %LogDir%\Twitter_Exp4_Em32_Min2_Max3_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 3 --embed-size 64 --log_dir %LogDir%\Twitter_Exp4_Em64_Min2_Max3_ModParC > %LogDir%\Twitter_Exp4_Em64_Min2_Max3_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 3 --embed-size 128 --log_dir %LogDir%\Twitter_Exp4_Em128_Min2_Max3_ModParC > %LogDir%\Twitter_Exp4_Em128_Min2_Max3_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 4 --embed-size 16 --log_dir %LogDir%\Twitter_Exp4_Em16_Min2_Max4_ModParC > %LogDir%\Twitter_Exp4_Em16_Min2_Max4_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 4 --embed-size 32 --log_dir %LogDir%\Twitter_Exp4_Em32_Min2_Max4_ModParC > %LogDir%\Twitter_Exp4_Em32_Min2_Max4_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 4 --embed-size 64 --log_dir %LogDir%\Twitter_Exp4_Em64_Min2_Max4_ModParC > %LogDir%\Twitter_Exp4_Em64_Min2_Max4_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 4 --embed-size 128 --log_dir %LogDir%\Twitter_Exp4_Em128_Min2_Max4_ModParC > %LogDir%\Twitter_Exp4_Em128_Min2_Max4_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 5 --embed-size 16 --log_dir %LogDir%\Twitter_Exp4_Em16_Min2_Max5_ModParC > %LogDir%\Twitter_Exp4_Em16_Min2_Max5_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 5 --embed-size 32 --log_dir %LogDir%\Twitter_Exp4_Em32_Min2_Max5_ModParC > %LogDir%\Twitter_Exp4_Em32_Min2_Max5_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 5 --embed-size 64 --log_dir %LogDir%\Twitter_Exp4_Em64_Min2_Max5_ModParC > %LogDir%\Twitter_Exp4_Em64_Min2_Max5_ModParC_Result.txt
python %twitter.py% --model parcnn --small-conv-size 2 --large-conv-size 5 --embed-size 128 --log_dir %LogDir%\Twitter_Exp4_Em128_Min2_Max5_ModParC > %LogDir%\Twitter_Exp4_Em128_Min2_Max5_ModParC_Result.txt
