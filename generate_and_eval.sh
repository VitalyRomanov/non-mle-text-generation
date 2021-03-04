python generate.py --data "$1" --src_lang "$2" --trg_lang "$3" --batch-size 64 --gpuid 0 --model_name "$4" --model_file "$5"
bash postprocess.sh < real.txt > real_processed.txt
bash postprocess.sh < predictions.txt > predictions_processed.txt
perl scripts/multi-bleu.perl real_processed.txt < predictions_processed.txt




# bash generate_and_eval.sh data-bin/iwslt14.tokenized.de-en/ de en vae path_to_model
# bash generate_and_eval.sh path_to_data src_lng tgt_lng model_type path_to_model

