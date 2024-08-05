inputpath=${1}
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

echo "$JSON" > "configs/cls_task/custom.json"

python tools/esm_emb_gen.py --dataset_path "$inputpath" &&

python tools/generate_stc_csv.py --inputpath "$inputpath" --outputpath "datasets/stc_datasets/$filename" &&

python tools/stc_gen.py --dataset_dir "datasets/stc_datasets/" --datafile "$filename" --fname "$rawname.h5" &&

python -m torch.distributed.launch run.py --config configs/cls_task/custom_SenseXAMP.py --mode inference --output_path "$outputpath"
