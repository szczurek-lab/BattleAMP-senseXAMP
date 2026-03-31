#!/usr/bin/env bash
set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE


INPUT_FASTA="$1"
OUTPUT_TSV="$2"

if [ -z "$INPUT_FASTA" ] || [ -z "$OUTPUT_TSV" ]; then
    echo "Usage: inference.sh <input.fasta> <output.tsv>" >&2
    exit 1
fi

if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: input FASTA not found: $INPUT_FASTA" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure conda's libstdc++ is found.
CONDA_LIB=$(python -c "import sys; print(sys.prefix + '/lib')")
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

OUTPUT_PREFIX="${OUTPUT_TSV%.tsv}"
FILTERED_FASTA="${OUTPUT_TSV}.filtered.fasta"

# --------------------------------------------------------------------------
# 1. Filter: keep only sequences with standard AAs, length 6-25.
# --------------------------------------------------------------------------
python3 -c "
from Bio import SeqIO
import sys

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
kept, skipped = 0, 0
with open(sys.argv[2], 'w') as out:
    for rec in SeqIO.parse(sys.argv[1], 'fasta'):
        seq = str(rec.seq).upper()
        if len(seq) < 6 or len(seq) > 25:
            skipped += 1
            continue
        if not all(c in STANDARD_AA for c in seq):
            skipped += 1
            continue
        out.write(f'>{rec.id}\n{seq}\n')
        kept += 1
print(f'SenseXAMP: kept {kept}, skipped {skipped} sequences', file=sys.stderr)
" "$INPUT_FASTA" "$FILTERED_FASTA"

# Bail if nothing left.
if [ ! -s "$FILTERED_FASTA" ]; then
    echo "SenseXAMP: no sequences pass filters, writing empty outputs" >&2
    echo -e "sequence\tPrediction\tProbability_score" > "${OUTPUT_PREFIX}-classifier.tsv"
    echo -e "sequence\tMIC\tMIC_unit" > "${OUTPUT_PREFIX}-ecoli.tsv"
    echo -e "sequence\tMIC\tMIC_unit" > "${OUTPUT_PREFIX}-saureus.tsv"
    cp "${OUTPUT_PREFIX}-classifier.tsv" "$OUTPUT_TSV"
    rm -f "$FILTERED_FASTA"
    exit 0
fi

# Derive names for intermediate files.
CLS_CSV="${OUTPUT_TSV}.cls.csv"
REG_CSV="${OUTPUT_TSV}.reg.csv"
CLS_CSVNAME=$(basename "$CLS_CSV")
REG_CSVNAME=$(basename "$REG_CSV")
CLS_RAWNAME="${CLS_CSVNAME%.*}"
REG_RAWNAME="${REG_CSVNAME%.*}"
EMB_H5="datasets/esm_embeddings/all/${CLS_RAWNAME}.h5"
STC_CLS_CSV="datasets/stc_datasets/${CLS_CSVNAME}"
STC_REG_CSV="datasets/stc_datasets/${REG_CSVNAME}"
STC_CLS_H5="datasets/stc_info/${CLS_RAWNAME}.h5"
STC_REG_H5="datasets/stc_info/${REG_RAWNAME}.h5"

# --------------------------------------------------------------------------
# 2. Convert FASTA to CSV for cls and reg tasks.
# --------------------------------------------------------------------------
echo "SenseXAMP: converting inputs..." >&2
python tools/convert_inputs.py --fasta_file "$FILTERED_FASTA" --csv_file "$CLS_CSV" --task cls
python tools/convert_inputs.py --fasta_file "$FILTERED_FASTA" --csv_file "$REG_CSV" --task reg

# --------------------------------------------------------------------------
# 3. Generate ESM-1b embeddings (shared across all variants).
# --------------------------------------------------------------------------
echo "SenseXAMP: generating ESM-1b embeddings..." >&2
python tools/esm_emb_gen.py --dataset_path "$CLS_CSV"

# --------------------------------------------------------------------------
# 4. Generate structural features for cls and reg.
# --------------------------------------------------------------------------
echo "SenseXAMP: generating structural features..." >&2
python tools/generate_stc_csv.py --inputpath "$CLS_CSV" --outputpath "$STC_CLS_CSV" --task cls
python tools/generate_stc_csv.py --inputpath "$REG_CSV" --outputpath "$STC_REG_CSV" --task reg

python tools/stc_gen.py --dataset_dir datasets/stc_datasets/ --datafile "$CLS_CSVNAME" --fname "${CLS_RAWNAME}.h5" --task cls
python tools/stc_gen.py --dataset_dir datasets/stc_datasets/ --datafile "$REG_CSVNAME" --fname "${REG_RAWNAME}.h5" --task reg

# --------------------------------------------------------------------------
# 5. Run classifier variant.
# --------------------------------------------------------------------------
echo "SenseXAMP: running classifier inference..." >&2
CLS_RAW="${OUTPUT_PREFIX}.cls_raw.tsv"

cat > configs/cls_task/custom.json <<EOF
{
    "data_fpath": "$CLS_CSV",
    "embedding_fpath": "$EMB_H5",
    "stc_fpath": "$STC_CLS_H5"
}
EOF

torchrun --nproc_per_node 1 run.py \
    --config configs/cls_task/battleamp_SenseXAMP.py \
    --mode inference --task cls --output_path "$CLS_RAW"

python tools/convert_outputs.py --input_file "$FILTERED_FASTA" --output_file "$CLS_RAW" --task cls

# Convert classifier to pipeline format.
python3 -c "
import pandas as pd, sys
df = pd.read_csv(sys.argv[1], sep='\t')
out = df[['Sequence', 'Prediction', 'Probability_score']].copy()
out.rename(columns={'Sequence': 'sequence'}, inplace=True)
out.to_csv(sys.argv[2], sep='\t', index=False)
" "$CLS_RAW" "${OUTPUT_PREFIX}-classifier.tsv"

# --------------------------------------------------------------------------
# 6. Run E. coli regression variant.
# --------------------------------------------------------------------------
echo "SenseXAMP: running E. coli regression..." >&2
ECOLI_RAW="${OUTPUT_PREFIX}.ecoli_raw.tsv"

cat > configs/reg_task/custom.json <<EOF
{
    "data_fpath": "$REG_CSV",
    "embedding_fpath": "$EMB_H5",
    "stc_fpath": "$STC_REG_H5"
}
EOF

torchrun --nproc_per_node 1 run.py \
    --config configs/reg_task/ecoli_battleamp.py \
    --mode inference --task reg --output_path "$ECOLI_RAW"

python tools/convert_outputs.py --input_file "$FILTERED_FASTA" --output_file "$ECOLI_RAW" --task reg

# Convert ecoli regression to pipeline format.
python3 -c "
import pandas as pd, sys
df = pd.read_csv(sys.argv[1], sep='\t')
out = pd.DataFrame({
    'sequence': df['Sequence'],
    'MIC': df['MIC'],
    'MIC_unit': df['MIC_unit'],
})
out.to_csv(sys.argv[2], sep='\t', index=False)
" "$ECOLI_RAW" "${OUTPUT_PREFIX}-ecoli.tsv"

# --------------------------------------------------------------------------
# 7. Run S. aureus regression variant.
# --------------------------------------------------------------------------
echo "SenseXAMP: running S. aureus regression..." >&2
SAUREUS_RAW="${OUTPUT_PREFIX}.saureus_raw.tsv"

torchrun --nproc_per_node 1 run.py \
    --config configs/reg_task/saureus_battleamp.py \
    --mode inference --task reg --output_path "$SAUREUS_RAW"

python tools/convert_outputs.py --input_file "$FILTERED_FASTA" --output_file "$SAUREUS_RAW" --task reg

# Convert saureus regression to pipeline format.
python3 -c "
import pandas as pd, sys
df = pd.read_csv(sys.argv[1], sep='\t')
out = pd.DataFrame({
    'sequence': df['Sequence'],
    'MIC': df['MIC'],
    'MIC_unit': df['MIC_unit'],
})
out.to_csv(sys.argv[2], sep='\t', index=False)
" "$SAUREUS_RAW" "${OUTPUT_PREFIX}-saureus.tsv"

# --------------------------------------------------------------------------
# 8. Cleanup intermediate files.
# --------------------------------------------------------------------------
rm -f "$FILTERED_FASTA" "$CLS_CSV" "$REG_CSV"
rm -f "$CLS_RAW" "$ECOLI_RAW" "$SAUREUS_RAW"
rm -f "$EMB_H5" "$STC_CLS_CSV" "$STC_REG_CSV" "$STC_CLS_H5" "$STC_REG_H5"

# Snakemake multioutput rule expects predictions.tsv to exist.
# Copy the primary variant (classifier) as the base file.
cp "${OUTPUT_PREFIX}-classifier.tsv" "$OUTPUT_TSV"

echo "SenseXAMP: inference complete" >&2
echo "  classifier: ${OUTPUT_PREFIX}-classifier.tsv" >&2
echo "  ecoli:      ${OUTPUT_PREFIX}-ecoli.tsv" >&2
echo "  saureus:    ${OUTPUT_PREFIX}-saureus.tsv" >&2

# The pipeline's multioutput rule expects a base predictions.tsv to exist.
# Copy the primary variant (classifier) as the base file.
cp "${OUTPUT_PREFIX}-classifier.tsv" "$OUTPUT_TSV"