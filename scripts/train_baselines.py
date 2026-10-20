import _bootstrap
import argparse
from threadpoolctl import threadpool_limits
from src.data.loader import load_dataset
from src.models.baselines import train_baselines

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train five model families on grouped partitions.")
    parser.add_argument("--data",default=None)
    parser.add_argument("--search-iterations",type=int,default=2,choices=[1,2])
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        train_baselines(load_dataset(args.data),args.search_iterations)
