conda activate text-generation

python joint-train.py --data data-bin/t5-small/arxiv/ --src_lang original --trg_lang summary --g_learning_rate 1e-5 --d_learning_rate 1e-2 --joint-batch-size 4 --epochs 15 --gpuid -1 --distributed-world-size 1 --model_file "./checkpoints" --model_name t5mle --sample-without-replacement 4000
python joint-train.py --data data-bin/t5-small/arxiv/ --src_lang original --trg_lang summary --g_learning_rate 1e-5 --d_learning_rate 1e-2 --joint-batch-size 4 --epochs 20 --gpuid -1 --distributed-world-size 1 --model_file "./checkpoints" --model_name t5rl --imp_smpl_epsilon 0.1 --d_pretraining 5  --sample-without-replacement 4000
python joint-train.py --data data-bin/t5-small/arxiv/ --src_lang original --trg_lang summary --g_learning_rate 1e-5 --d_learning_rate 1e-2 --joint-batch-size 4 --epochs 20 --gpuid -1 --distributed-world-size 1 --model_file "./checkpoints" --model_name t5dumbel --imp_smpl_epsilon 0.1 --d_pretraining 5  --sample-without-replacement 4000

conda deactivate