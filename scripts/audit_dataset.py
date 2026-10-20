import _bootstrap
import argparse
from pathlib import Path
from src.common import ROOT
from src.data.loader import load_dataset
from src.data.validation import audit_dataset

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Audit the entire authorized source CSV.")
    parser.add_argument("--data",default=str(ROOT/"data/raw/FINAL_dataset_with_fitbit.csv"))
    args=parser.parse_args()
    print(audit_dataset(load_dataset(args.data),Path(args.data)))
