import _bootstrap
import argparse
from src.data.loader import load_dataset
from src.models.fusion_training import fit_fusion

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train mixed-source EHR/synthetic-sensor fusion network.")
    parser.add_argument("--data",default=None)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--max-train",type=int,default=4096)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    args=parser.parse_args()
    print(fit_fusion(load_dataset(args.data),args.epochs,args.max_train,args.device))
