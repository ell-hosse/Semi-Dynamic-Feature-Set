#from examples.timeseries.SDFS_ETTh1 import run_example
#from examples.timeseries.SDFS_AirQualityUCI import run_example as run_example_sdfs
#from examples.timeseries.AirQualityUCI import run_example
#import argparse
#from examples.wine_quality_multiple_splits import run_random_splits
#from examples.wine_quality_k_fold import run_cv_80_20
from examples.diabetes_health_indicators import run_example


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="Run example with custom dynamic input size.")
    parser.add_argument("--dynamic_input_size", type=int, default=5,
                        help="Number of dynamic features (default: 5)")
    args = parser.parse_args()

    run_example(dynamic_input_size=args.dynamic_input_size)
    
    #run_random_splits(seeds=(1, 2, 3, 4, 5), dynamic_input_size=4)
    #run_cv_80_20(n_splits=5, val_size=0.2, epochs=100, dynamic_input_size=5, random_state=123)

    run_example()
    run_example_sdfs(dynamic_input_size=1)
    run_example_sdfs(dynamic_input_size=2)
    run_example_sdfs(dynamic_input_size=3)
    run_example_sdfs(dynamic_input_size=4)
    '''
    run_example()