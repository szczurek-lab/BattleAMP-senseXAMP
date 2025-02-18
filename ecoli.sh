input_fasta=${1}
inputpath=${input_fasta%.fasta}.csv
outputpath=${2}
filename=${inputpath##*/}
dirpath=${inputpath%/*}
rawname=${filename%%.*}

JSON=$(jq -n \
  --arg data_fpath "datasets/stc_datasets/$filename" \
  --arg embedding_fpath "datasets/esm_embeddings/all/$rawname.h5" \
  --arg stc_fpath "datasets/stc_info/$rawname.h5" \
  '$ARGS.named'
)

echo "$JSON" > "configs/reg_task/custom.json"

python tools/convert_inputs.py --fasta_file "$input_fasta" --csv_file "$inputpath" --task "reg" &&

python tools/esm_emb_gen.py --dataset_path "$inputpath" &&

python tools/generate_stc_csv.py --inputpath "$inputpath" --outputpath "datasets/stc_datasets/$filename" --task "reg" &&

python tools/stc_gen.py --dataset_dir "datasets/stc_datasets/" --datafile "$filename" --fname "$rawname.h5" &&

python -m torch.distributed.launch run.py --config configs/reg_task/ecoli_SenseXAMP.py --mode inference --task "reg" --output_path "$outputpath" &&

python tools/convert_outputs.py --input_file "$input_fasta" --output_file "$outputpath" --task "reg"
