
input_fasta=${1}
outputpath=${2}
model=${3}

set -e

if [ "$model" = "classification" ]; then
  task="cls"
  config="battleamp_SenseXAMP"
elif [ "$model" = "regression_ecoli" ]; then
  task="reg"
  config="ecoli_battleamp"
elif [ "$model" = "regression_saureus" ]; then
  task="reg"
  config="saureus_battleamp"
elif [ "$model" = "multioutput" ]; then
  task_classification="cls"
  config_classification="battleamp_SenseXAMP"
  output_classification="${outputpath%.*}-classifier.tsv"

  task_reg_ecoli="reg"
  config_reg_ecoli="ecoli_battleamp"
  output_reg_ecoli="${outputpath%.*}-ecoli.tsv"

  task_reg_saureus="reg"
  config_reg_saureus="saureus_battleamp"
  output_reg_saureus="${outputpath%.*}-saureus.tsv"

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

# Create the necessary JSON configuration for multi-output
if [ "$model" = "multioutput" ]; then

  echo $JSON > configs/$task_classification\_task/custom.json
  echo $JSON > configs/$task_reg_ecoli\_task/custom.json
  echo $JSON > configs/$task_reg_saureus\_task/custom.json
else
  echo $JSON > configs/$task\_task/custom.json
fi

echo "Converting inputs..."
python tools/convert_inputs.py --fasta_file $input_fasta --csv_file $inputpath 

echo "Generating embeddings..."
python tools/esm_emb_gen.py --dataset_path $inputpath 

echo "Generate structured peptide data..."
echo "$inputpath"
echo "datasets/stc_datasets/$filename"
python tools/generate_stc_csv.py --inputpath $inputpath --outputpath datasets/stc_datasets/$filename 

echo "Generating .h5 files..."
python tools/stc_gen.py --dataset_dir datasets/stc_datasets/ --datafile $filename --fname $rawname.h5 

if [ "$model" = "multioutput" ]; then
  echo "Running classification prediction..."
  python -m torch.distributed.launch run.py --config configs/$task_classification\_task/$config_classification.py --mode inference --task $task_classification --output_path $output_classification 

  echo "Running regression (ecoli) prediction..."
  python -m torch.distributed.launch run.py --config configs/$task_reg_ecoli\_task/$config_reg_ecoli.py --mode inference --task $task_reg_ecoli --output_path $output_reg_ecoli 

  echo "Running regression (saureus) prediction..."
  python -m torch.distributed.launch run.py --config configs/$task_reg_saureus\_task/$config_reg_saureus.py --mode inference --task $task_reg_saureus --output_path $output_reg_saureus
else
  echo "Running prediction..."
  python -m torch.distributed.launch run.py --config configs/$task\_task/$config.py --mode inference --task $task --output_path $outputpath
fi

echo "Converting outputs..."
if [ "$model" = "multioutput" ]; then
  python tools/convert_outputs.py --input_file $input_fasta --output_file $output_classification --task $task_classification
  python tools/convert_outputs.py --input_file $input_fasta --output_file $output_reg_ecoli --task $task_reg_ecoli
  python tools/convert_outputs.py --input_file $input_fasta --output_file $output_reg_saureus --task $task_reg_saureus
  python tools/merge_outputs.py --classifier $output_classification --ecoli $output_reg_ecoli --saureus $output_reg_saureus --output $outputpath
else
  python tools/convert_outputs.py --input_file $input_fasta --output_file $outputpath --task $task
fi

# Clean up intermediate files
rm $inputpath