input_fasta=${1}
outputpath=${2}
model=${3}

if [ "$model" = "classification" ]; then
  task="cls"
  config="battleamp_SenseXAMP"
elif [ "$model" = "regression_ecoli" ]; then
  task="reg"
  config="ecoli_battleamp"
elif [ "$model" = "regression_saureus" ]; then
  task="reg"
  config="saureus_battleamp"
else
    echo "Unrecognized model $model"
    exit 1
fi

inputpath=${input_fasta%.fasta}.csv
filename=${inputpath##*/}
dirpath=${inputpath%/*}
rawname=${filename%%.*}

JSON=$(jq -n \
  --arg data_fpath "datasets/stc_datasets/$filename" \
  --arg embedding_fpath "datasets/esm_embeddings/all/$rawname.h5" \
  --arg stc_fpath "datasets/stc_info/$rawname.h5" \
  '$ARGS.named'
)

echo $JSON > configs/$task\_task/custom.json

python tools/convert_inputs.py --fasta_file $input_fasta --csv_file $inputpath --task $task &&

python tools/esm_emb_gen.py --dataset_path $inputpath &&

python tools/generate_stc_csv.py --inputpath $inputpath --outputpath datasets/stc_datasets/$filename --task $task &&

python tools/stc_gen.py --dataset_dir datasets/stc_datasets/ --datafile $filename --fname $rawname.h5 --task $task &&

python -m torch.distributed.launch run.py --config configs/$task\_task/$config.py --mode inference --task $task --output_path $outputpath &&

python tools/convert_outputs.py --input_file $input_fasta --output_file $outputpath --task $task &&

rm $inputpath
