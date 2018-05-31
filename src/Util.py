import geopy.distance

def distanceBetween2Points(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)

    distance = geopy.distance.distance(coords_1, coords_2).km
    print ("%.1f" % distance)
    return distance