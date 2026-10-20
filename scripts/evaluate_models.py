import _bootstrap
import argparse
from threadpoolctl import threadpool_limits
from src.data.loader import load_dataset
from src.evaluation.validation import evaluate_models,synthetic_evaluation

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Evaluate locked models and independent synthetic trajectories.")
    parser.add_argument("--data",default=None)
    parser.add_argument("--bootstrap",type=int,default=200)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        print(evaluate_models(load_dataset(args.data),args.bootstrap))
        print(synthetic_evaluation(repeats=args.bootstrap))
