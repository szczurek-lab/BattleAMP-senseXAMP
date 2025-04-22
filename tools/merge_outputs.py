import pandas as pd
import argparse

def merge_files(classifier_path, ecoli_path, saureus_path, output_path):
    # Load the TSV files
    classifier_df = pd.read_csv(classifier_path, sep="\t")
    ecoli_df = pd.read_csv(ecoli_path, sep="\t")
    saureus_df = pd.read_csv(saureus_path, sep="\t")

    # Rename columns in ecoli and saureus to reflect their source
    ecoli_df = ecoli_df.rename(columns={
        "Log10_MIC": "Log10_MIC_ecoli",
        "MIC": "MIC_ecoli"
    })

    saureus_df = saureus_df.rename(columns={
        "Log10_MIC": "Log10_MIC_saureus",
        "MIC": "MIC_saureus"
    })

    # Merge all dataframes on Sequence_id and Sequence
    merged_df = classifier_df.merge(ecoli_df, on=["Sequence_id", "Sequence"], how="outer") \
                             .merge(saureus_df, on=["Sequence_id", "Sequence", "MIC_unit"], how="outer")

    # Save the merged dataframe to a new TSV file
    merged_df.to_csv(output_path, sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge classifier, E. coli, and S. aureus data files.")
    parser.add_argument("--classifier", required=True, help="Path to sensexamp-classifier.tsv")
    parser.add_argument("--ecoli", required=True, help="Path to sensexamp-ecoli.tsv")
    parser.add_argument("--saureus", required=True, help="Path to sensexamp-saureus.tsv")
    parser.add_argument("--output", required=True, help="Path to save merged output TSV file")

    args = parser.parse_args()
    merge_files(args.classifier, args.ecoli, args.saureus, args.output)