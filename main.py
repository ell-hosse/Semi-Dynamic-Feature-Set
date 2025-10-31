#from examples.timeseries.SDFS_ETTh1 import run_example
from examples.timeseries.SDFS_AirQualityUCI import run_example
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example with custom dynamic input size.")
    parser.add_argument("--dynamic_input_size", type=int, default=5,
                        help="Number of dynamic features (default: 5)")
    args = parser.parse_args()

    run_example(dynamic_input_size=args.dynamic_input_size)