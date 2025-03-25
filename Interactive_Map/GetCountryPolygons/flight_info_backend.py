import requests
from geopy import distance
import time

# Cache to store country coordinates
coordinate_cache = {}


def get_country_coordinates(country_name):
    if country_name in coordinate_cache:
        return coordinate_cache[country_name]

    url = f"https://nominatim.openstreetmap.org/search?country={country_name}&format=json"
    response = requests.get(url)
    data = response.json()

    if data:
        coords = (float(data[0]['lat']), float(data[0]['lon']))
        coordinate_cache[country_name] = coords
        time.sleep(1)  # Respect Nominatim usage policy
        return coords
    return None


def calculate_distance(coord1, coord2):
    return distance.distance(coord1, coord2).km


def estimate_flight_time(distance_km, lat1, lat2):
    # Adjust flight speed based on latitude (slower near poles)
    avg_lat = (abs(lat1) + abs(lat2)) / 2
    speed_factor = 1 - (avg_lat / 90) * 0.2  # Reduce speed by up to 20% near poles
    avg_speed = 800 * speed_factor

    flight_hours = distance_km / avg_speed
    return flight_hours + 0.5  # Add 30 minutes for takeoff and landing


def get_flight_info(country1, country2):
    coord1 = get_country_coordinates(country1)
    coord2 = get_country_coordinates(country2)

    if coord1 is None or coord2 is None:
        return None

    distance_km = calculate_distance(coord1, coord2)
    flight_time = estimate_flight_time(distance_km, coord1[0], coord2[0])

    return {
        "from": country1,
        "to": country2,
        "distance_km": round(distance_km, 2),
        "flight_time_hours": round(flight_time, 2)
    }


if __name__ == "__main__":
    result = get_flight_info("France", "Japan")
    if result:
        print(f"Flight from {result['from']} to {result['to']}:")
        print(f"Distance: {result['distance_km']} km")
        print(f"Estimated flight time: {result['flight_time_hours']} hours")
    else:
        print("Could not calculate flight information.")