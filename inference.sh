python tools/esm_emb_gen.py --dataset_dir datasets/ori_datasets/custom/ --fname custom.h5

python tools/generate_stc_csv.py --data_dir datasets/ori_datasets/custom/ --out_dir datasets/stc_datasets/custom

python tools/stc_gen.py --dataset_dir datasets/stc_datasets/custom/ --datafile dummy.csv --fname custom.h5

python -m torch.distributed.launch run.py --config configs/cls_task/custom_SenseXAMP.py --mode inference --output_path resutls.tsv
