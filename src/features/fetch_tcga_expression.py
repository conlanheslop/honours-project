"""Download TCGA-LAML RNA-seq data and extract expression for target proteins."""

import os
import io
import gzip
import tarfile
import requests
import pandas as pd

GENES = {
    "CCNA1": "ENSG00000133101",
    "HOXA9": "ENSG00000078399",
    "PRAME": "ENSG00000185686",
    "WT1":   "ENSG00000184937",
}

TUMOUR_SAMPLE_TYPES = {
    "Primary Blood Derived Cancer - Peripheral Blood",
    "Primary Blood Derived Cancer - Bone Marrow",
}

RAW_DIR = "data/raw/tcga"
BUNDLE_PATH = os.path.join(RAW_DIR, "laml_expression_files.tar.gz")
OUT_PATH = "data/processed/tcga_laml_expression_per_sample.csv"


def query_gdc_files():
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LAML"]}},
            {"op": "in", "content": {"field": "files.data_category", "value": ["Transcriptome Profiling"]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "files.experimental_strategy", "value": ["RNA-Seq"]}},
            {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}},
        ]
    }
    params = {
        "filters": str(filters).replace("'", '"'),
        "fields": "file_id,file_name,cases.case_id,cases.samples.sample_type",
        "format": "JSON",
        "size": "500",
    }
    resp = requests.get("https://api.gdc.cancer.gov/files", params=params)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]
    print(f"Found {len(hits)} candidate files")
    return hits


def filter_tumour_files(hits):
    tumour_files = []
    for h in hits:
        sample_types = {s["sample_type"] for c in h["cases"] for s in c.get("samples", [])}
        if sample_types & TUMOUR_SAMPLE_TYPES:
            tumour_files.append(h)
    print(f"{len(tumour_files)} files are tumour samples")
    return tumour_files


def download_bundle(file_ids):
    resp = requests.post(
        "https://api.gdc.cancer.gov/data",
        json={"ids": file_ids},
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    with open(BUNDLE_PATH, "wb") as f:
        f.write(resp.content)
    print(f"Downloaded bundle to {BUNDLE_PATH}")


def parse_bundle():
    records = []
    with tarfile.open(BUNDLE_PATH) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".tsv")]
        for i, member in enumerate(members):
            f = tar.extractfile(member)
            content = f.read()
            try:
                content = gzip.decompress(content)
            except OSError:
                pass

            df = pd.read_csv(
                io.BytesIO(content), sep="\t", header=None,
                names=["gene_id", "gene_name", "gene_type", "unstranded",
                       "stranded_first", "stranded_second",
                       "tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded"],
            )
            df = df[~df["gene_id"].astype(str).str.startswith("N_")]
            df = df[df["gene_id"].astype(str) != "gene_id"]

            for protein, ensembl_id in GENES.items():
                match = df[df["gene_id"].str.startswith(ensembl_id)]
                if len(match) == 1:
                    records.append({
                        "protein": protein,
                        "sample_file": member.name,
                        "tpm": float(match["tpm_unstranded"].iloc[0]),
                    })

            if (i + 1) % 25 == 0:
                print(f"  parsed {i + 1}/{len(members)} files")

    return pd.DataFrame(records)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    hits = query_gdc_files()
    tumour_files = filter_tumour_files(hits)

    if not os.path.exists(BUNDLE_PATH):
        download_bundle([f["file_id"] for f in tumour_files])
    else:
        print(f"{BUNDLE_PATH} already exists, skipping download")

    expr_long = parse_bundle()
    expr_long.to_csv(OUT_PATH, index=False)
    print(f"Extracted {len(expr_long)} gene-sample TPM values -> {OUT_PATH}")


if __name__ == "__main__":
    main()