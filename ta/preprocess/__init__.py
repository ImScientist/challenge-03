import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .iata import preprocess_iata
from .events import preprocess_events, join_with_iata
from .events_group import \
    generate_features__user_id, \
    generate_features__user_id__n_bookings


def load_and_preprocess_data(
        data_dir: str,
        countries_overwrite: bool = False,
        google_api_key: str = None
):
    """ Load, preprocess and merge events and location data
    """
    events = pd.read_csv(os.path.join(data_dir, 'events_1_1.csv'))
    iata = pd.read_csv(os.path.join(data_dir, 'iata_1_1.csv'))

    events = preprocess_events(events)
    iata = preprocess_iata(
        iata,
        countries_path=os.path.join(data_dir, 'iata_countries.csv'),
        countries_overwrite=countries_overwrite,
        google_api_key=google_api_key)

    events = join_with_iata(events, iata)

    # encode 3-letter origin-destination abbreviations
    # encode country names
    le = LabelEncoder()
    le_countries = LabelEncoder()

    le.fit(events['origin'].append(events['destination']))
    le_countries.fit(events['countries_origin'].append(events['countries_destination']))

    events['origin'] = le.transform(events['origin'])
    events['destination'] = le.transform(events['destination'])

    events['countries_origin'] = le_countries.transform(events['countries_origin'])
    events['countries_destination'] = le_countries.transform(events['countries_destination'])

    # encode `search` and `book` events manually
    events['event_type'] = events['event_type'].map({'search': 0, 'book': 1})
    events = events.sort_values(by=['user_id', 'ts'])

    return events, le, le_countries


def generate_new_features(events, nrows: int = None):
    """ Generate features for every user_id
    """

    # TODO: this can be parallelized
    data = events[:nrows] \
        .groupby(['user_id'], group_keys=False) \
        .apply(lambda x: generate_features__user_id(x)) \
        .groupby(['user_id', 'n_bookings'], group_keys=False) \
        .apply(lambda x: generate_features__user_id__n_bookings(x))

    return data


def get_data(
        data_dir: str,
        load_data: bool,
        save_data: bool,
        countries_overwrite: bool,
        nrows: int = None,
        google_api_key: str = None
):
    """ Get the training data with all features
    """
    if load_data is True:
        with open(os.path.join(data_dir, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
        with open(os.path.join(data_dir, 'le.pkl'), 'rb') as f:
            le = pickle.load(f)
        with open(os.path.join(data_dir, 'le_countries.pkl'), 'rb') as f:
            le_countries = pickle.load(f)
    else:
        data, le, le_countries = load_and_preprocess_data(
            data_dir, countries_overwrite, google_api_key
        )
        data = generate_new_features(data, nrows)

    if save_data is True:
        with open(os.path.join(data_dir, 'data.pkl'), 'wb') as f:
            pickle.dump(data, f)
        with open(os.path.join(data_dir, 'le.pkl'), 'wb') as f:
            pickle.dump(le, f)
        with open(os.path.join(data_dir, 'le_countries.pkl'), 'wb') as f:
            pickle.dump(le_countries, f)

    # fill nans
    data = data.fillna(-1)
    data = data.reset_index(drop=True)

    return data, le, le_countries
