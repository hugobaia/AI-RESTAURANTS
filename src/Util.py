import geopy.distance

def distanceBetween2Points(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)

    distance = geopy.distance.distance(coords_1, coords_2).km
    return distance

def calculateDistanceScore(users, rest):
    sum = 0
    lat = rest['location/lat']
    lon = rest['location/lng']
    for user in users:
        sum = sum + distanceBetween2Points(user["lat"], user["lon"], lat, lon)

    return (1 - (sum / (len(users)*20)))
