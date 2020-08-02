from math import sin, cos, sqrt, atan2, radians


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float):
    """ Get haversine distance [km]

    :param lat1: measured in degrees
    :param lon1: measured in degrees
    :param lat2: measured in degrees
    :param lon2: measured in degrees
    :return:
    """
    # approximate radius of earth in km
    radius = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = radius * c

    return distance
