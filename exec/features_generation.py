import os
import argparse
from ta.preprocess import get_data

DATA_DIR = os.environ['DATA_DIR']
RESULTS_DIR = os.environ['RESULTS_DIR']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']


def train(
        data_dir: str,
        results_dir: str,
        save_data: bool = False,
        countries_overwrite: bool = False,
        nrows: int = None,
):
    os.makedirs(results_dir, exist_ok=True)

    _, _, _ = get_data(
        data_dir,
        load_data=False,
        save_data=save_data,
        countries_overwrite=countries_overwrite,
        nrows=nrows,
        google_api_key=GOOGLE_API_KEY
    )

    print('DONE')


if __name__ == "__main__":
    """ Generate the training data set and, eventually, store it in  `$DATA_DIR/data.pkl`.
     
    Usage:
        source .env
        python exec/features_generation.py --nrows 200
        
        # this will overwrite ${DATA_DIR}/data.pkl
        python exec/features_generation.py --nrows 200 --save_data
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, dest="data_dir", default=DATA_DIR,
                        help="Data directory. The preprocessed data will be stored here, as well.")

    parser.add_argument("--results_dir", type=str, dest="results_dir", default=RESULTS_DIR,
                        help="Directory where all training artifacts will be saved.")

    parser.add_argument("--countries_overwrite", action="store_true",
                        help="Use the Google Maps API to get the Country names of "
                             "the origin and destination 3 letter abbreviations.\n"
                             "The results will be stored in `$DATA_DIR/iata_countries.csv`.\n"
                             "A valid GOOGLE_API_KEY is required.")

    parser.add_argument("--save_data", action="store_true",
                        help="Store the results in `$DATA_DIR/data.pkl`.")

    parser.add_argument("--nrows", type=int, dest="nrows", default=None,
                        help='Number of rows from the training data set that will be processed. '
                             "Used only to test the algorithm on a small subset.")

    args = parser.parse_args()

    for arg in vars(args):
        print("{0:34s} \t {1:20s}".format(arg, str(getattr(args, arg))))

    os.makedirs(args.results_dir, exist_ok=True)

    _, _, _ = get_data(
        data_dir=args.data_dir,
        load_data=False,
        save_data=args.save_data,
        countries_overwrite=args.countries_overwrite,
        nrows=args.nrows,
        google_api_key=GOOGLE_API_KEY
    )
