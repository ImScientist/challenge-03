from .utils import haversine_distance


def remove_consecutive_bookings_with_the_same_info(df):
    """

    :param df: dataframe where all elements have the same user_id
    :return:
    """
    init_columns = df.columns

    df['dt_date_from'] = (df['date_from'] - df['date_from'].shift(1)).apply(lambda x: x.days)
    df['dt_date_to'] = (df['date_to'] - df['date_to'].shift(1)).apply(lambda x: x.days)

    for col in ['event_type', 'origin', 'destination', 'num_adults', 'num_children']:
        df[f'dt_{col}'] = df[col] - df[col].shift(1)

    return df.loc[~(
            (df['dt_date_from'] == 0) &
            (df['dt_date_to'] == 0) &
            (df['dt_event_type'] == 0) &
            (df['dt_origin'] == 0) &
            (df['dt_destination'] == 0) &
            (df['dt_num_adults'] == 0) &
            (df['dt_num_children'] == 0) &
            (df['event_type'] == 1)
    ), init_columns]


def get_location_changes(df):
    """

    :param df: dataframe where all elements have the same user_id
    :return:
    """
    init_columns = df.columns.tolist()

    for col in ['lat_origin', 'lon_origin', 'lat_destination', 'lon_destination']:
        df[f'{col}_prev'] = df[col].shift(1)

    for col in ['origin', 'destination']:
        df[f'd_{col}_dist'] = df[[f'lat_{col}', f'lon_{col}', f'lat_{col}_prev', f'lon_{col}_prev']].apply(
            lambda x: haversine_distance(*x), 1)

    return df.loc[:, init_columns + ['d_origin_dist', 'd_destination_dist']]


def generate_features__user_id(df):
    """
    :param df: dataframe where all elements have the same user_id
    """
    # od_pair: ordered origin-destination pair;
    df['od_pair'] = df[['origin', 'destination']].apply(lambda x: min(x) * 1000 + max(x), 1)

    # n_bookings before this event
    df['n_bookings'] = df['event_type'].shift(1).fillna(0).expanding().sum().astype('int')

    df['distnace'] = df[['lat_origin', 'lon_origin', 'lat_destination', 'lon_destination']] \
        .apply(lambda x: int(haversine_distance(*x)), 1)

    df['date_range'] = (df['date_to'] - df['date_from']).apply(lambda x: x.days, 1)
    df['time_to_trip'] = (df['date_from'] - df['ts']).apply(lambda x: x.days)

    df['ts_hour'] = df['ts'].apply(lambda x: x.hour, 1)
    for col in ['ts', 'date_from', 'date_to']:
        df[f'{col}_month'] = df[f'{col}'].apply(lambda x: x.month, 1)
        df[f'{col}_weekday'] = df[f'{col}'].apply(lambda x: x.weekday(), 1)

    # check if there is any change in the value of a feature (in comparison to the previous registered event)
    for col in ['origin', 'destination', 'od_pair', 'num_adults', 'num_children']:
        df[f'd_{col}'] = (abs(df[col] - df[col].shift(1)) > 0).astype(int)

    # time difference btw two events
    df['dt_ts'] = (df['ts'] - df['ts'].shift(1)).apply(lambda x: x.total_seconds(), 1)

    df = get_location_changes(df)

    return df


def generate_features__user_id__n_bookings(df):
    """
    :param df: dataframe where all elements have the same user_id, n_bookings
    """
    # number of attempts (webpage visits) after the last book event
    df['attempt_n'] = df['ts'].rank(method="first").astype(int)

    # number of checked unique origin-destination paris after the last book event
    df['n_od_pairs'] = df['od_pair'].expanding().apply(lambda y: len(set(y))).astype(int)

    return df
