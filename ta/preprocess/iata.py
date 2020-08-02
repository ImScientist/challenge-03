import urllib
import requests
import pandas as pd
import numpy as np
from typing import List, Tuple
from .utils import haversine_distance


def get_max_distance(lat_lon_list: List[Tuple[float, float]]):
    if len(lat_lon_list) == 1:
        return 0
    else:
        max_d = None
        for i1 in range(len(lat_lon_list)):
            for i2 in range(i1, len(lat_lon_list)):
                d = haversine_distance(*lat_lon_list[i1], *lat_lon_list[i2])
                if max_d is None or d > max_d:
                    max_d = d

        return max_d


def reduce_destination_locations(lat_lon_list: List[Tuple[float, float]]):
    lat_list = list(map(lambda x: x[0], lat_lon_list))
    lon_list = list(map(lambda x: x[1], lat_lon_list))
    return np.mean(lat_list), np.mean(lon_list)


def get_state_google_api(lat, lon, key):
    endpoint = 'https://maps.googleapis.com/maps/api/geocode/json'
    query = {
        'latlng': f'{lat},{lon}',
        'result_type': 'country',
        'key': key
    }
    url = endpoint + '?' + urllib.parse.urlencode(query)
    result = requests.post(url)

    if result.status_code == 200:
        try:
            result = result.json()['results'][0]['formatted_address']
        except:
            result = 'NA'
    else:
        result = 'NA'

    return result


def get_iata_countries(
        iata,
        save_path: str,
        overwrite: bool = False,
        google_api_key: str = None
):
    """ Use the google geocode API to get information about the countries
    to which the (lat, lon)-pairs belong, and save the results

    :param iata:
    :param key: Google API key
    :param save_path: save location
    :return:
    """

    if overwrite is True:
        print('Use the Google API to get the countries')
        countries = iata[['lat', 'lon']].copy()
        # 'iata_code',

        countries['cou  ntries'] = list(map(
            lambda x: get_state_google_api(*x, google_api_key),
            countries[['lat', 'lon']].values.tolist()))

        countries.to_csv(save_path, index=False)
    else:
        # load saved dataframe
        countries = pd.read_csv(save_path)

    countries['countries'] = countries['countries'].fillna('NA')

    return countries


def preprocess_iata(
        iata,
        countries_path: str,
        countries_overwrite: bool = False,
        google_api_key: str = None
):
    iata = iata.drop_duplicates().copy(deep=True)
    iata['lat_lon'] = iata[['lat', 'lon']].apply(lambda x: (x[0], x[1]), 1)
    iata = iata.groupby(['iata_code']).agg({'lat_lon': list})

    iata['count'] = iata['lat_lon'].apply(lambda x: len(x))
    iata['max_distance'] = iata['lat_lon'].apply(lambda x: get_max_distance(x))
    iata['lat_lon_avg'] = iata['lat_lon'].apply(lambda x: reduce_destination_locations(x), 1)

    iata['lat'] = iata['lat_lon_avg'].apply(lambda x: x[0], 1)
    iata['lon'] = iata['lat_lon_avg'].apply(lambda x: x[1], 1)

    # get the countries corresponding to the lat, lon pairs
    countries = get_iata_countries(
        iata, countries_path, countries_overwrite, google_api_key
    )

    iata = iata.join(
        countries.set_index(['iata_code'])[['countries']],
        on='iata_code'
    )

    iata = iata.drop(['lat_lon_avg'], 1).reset_index()

    return iata
