import pandas as pd


def get_feature_names(data):
    """ Get categorical/continuous columns
    and summary of the used and unused columns
    """
    target_col = 'event_type'

    contiuous_columns = [
        'date_from_month', 'date_from_weekday', 'date_to_month', 'date_to_weekday', 'ts_weekday',
        'num_adults', 'num_children', 'n_bookings', 'n_od_pairs', 'ts_hour',
        'distnace', 'date_range', 'time_to_trip', 'attempt_n', 'dt_ts',
        'd_origin_dist', 'd_destination_dist'
    ]

    categorical_columns = [
        'origin', 'destination', 'od_pair',
        'd_origin', 'd_destination', 'd_od_pair', 'd_num_adults', 'd_num_children'
    ]

    data_info = pd.DataFrame(index=data.columns)

    data_info['categorical'] = -1  # these columns are not used at all
    data_info.loc[categorical_columns, 'categorical'] = 1
    data_info.loc[contiuous_columns, 'categorical'] = 0
    data_info['n_unique'] = [len(data[col].unique()) for col in data_info.index]
    data_info['dtype'] = data.dtypes

    return contiuous_columns, categorical_columns, target_col, data_info


def get_loss_fct_weights(
        data: pd.DataFrame, condition: pd.Series, target_col: str
):
    """ Get the weights that can be used for the calculation of a loss function.

    w_pos = (N_pos + N_neg)/N_pos = N/N_pos
    w_neg = 1

    :param data:
    :param condition: filter used for data selection
    :param target_col:
    :return:
    """

    N = data.loc[condition, target_col].shape[0]
    Np = data.loc[condition, target_col].sum()

    weights = data.loc[condition, target_col] * (N / Np - 1)
    weights += 1

    return weights
