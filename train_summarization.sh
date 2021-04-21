conda activate text-generation

python joint_train.py --data data-bin/t5-small/arxiv/ --src_lang original --trg_lang summary --g_learning_rate 1e-4 --d_learning_rate 1e-2 --joint-batch-size 4 --epochs 15 --fixed-max-len 256 --gpuid 1 --distributed-world-size 1 --model_file "./checkpoints" --model_name t5mle --sample-without-replacement 4000 --sample-val-without-replacement 1000
python joint_train.py --data data-bin/t5-small/arxiv/ --src_lang original --trg_lang summary --g_learning_rate 1e-4 --d_learning_rate 1e-2 --joint-batch-size 4 --epochs 20 --fixed-max-len 256 --gpuid 1 --distributed-world-size 1 --model_file "./checkpoints" --model_name t5rl --imp_smpl_epsilon 0.1 --d_pretraining 5  --sample-without-replacement 4000 --sample-val-without-replacement 1000
python joint_train.py --data data-bin/t5-small/arxiv/ --src_lang original --trg_lang summary --g_learning_rate 1e-4 --d_learning_rate 1e-2 --joint-batch-size 4 --epochs 20 --fixed-max-len 256 --gpuid 1 --distributed-world-size 1 --model_file "./checkpoints" --model_name t5gumbel --imp_smpl_epsilon 0.1 --d_pretraining 5  --sample-without-replacement 4000 --sample-val-without-replacement 1000

conda deactivate