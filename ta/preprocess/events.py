import pandas as pd


def preprocess_events(events):
    events = events.copy(deep=True).drop_duplicates()

    events = events.assign(
        ts=pd.to_datetime(events['ts'], format='%Y-%m-%d %H:%M:%S'),
        date_from=pd.to_datetime(events['date_from'], format='%Y-%m-%d'),
        date_to=pd.to_datetime(events['date_to'], format='%Y-%m-%d')
    )

    # `date_from` has null values;
    # Assume that in this case the user does not care about the exact departure date
    # and leaves this search field blank;
    # In this case we see that num_adults = num_children = 0
    events.loc[events['date_from'].isnull(), 'date_from'] = \
        events.loc[events['date_from'].isnull(), 'ts'].values.copy()

    return events


def users_with_at_least_two_registered_events(events):
    """ Keep only information about users that have at least registered two events
    """
    # get registered events per user
    usr_events = events \
        .groupby(['user_id']) \
        .agg({'event_type': 'count'}) \
        .rename(columns={'event_type': 'counts'}) \
        .reset_index()

    # users that have at least two registered events
    relevant_usr = usr_events.loc[usr_events['counts'] > 1, 'user_id'].values

    return events.loc[events['user_id'].isin(relevant_usr)]


def join_with_iata(events, iata):
    return events \
        .join(iata[['iata_code', 'lat', 'lon', 'countries']]
              .rename(columns={'iata_code': 'origin',
                               'lat': 'lat_origin',
                               'lon': 'lon_origin',
                               'countries': 'countries_origin'
                               })
              .set_index(['origin']),
              on='origin', rsuffix='origin') \
        .join(iata[['iata_code', 'lat', 'lon', 'countries']]
              .rename(columns={'iata_code': 'destination',
                               'lat': 'lat_destination',
                               'lon': 'lon_destination',
                               'countries': 'countries_destination'
                               })
              .set_index(['destination']),
              on='destination', rsuffix='destination')
