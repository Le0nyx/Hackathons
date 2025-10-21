import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry
import geocoder


class HappyMeteo:
    def __init__(self):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = retry_session)

    def get_current_location(self):
        """Get current location using IP geolocation"""
        try:
            g = geocoder.ip('me')
            if g.ok:
                latitude = g.latlng[0]
                longitude = g.latlng[1]
                print(f"Latitude: {latitude}")
                print(f"Longitude: {longitude}")
                print(f"Address: {g.address}")
                return latitude, longitude
            else:
                print("Could not determine location")
                return None, None
        except Exception as e:
            print(f"Error getting location: {e}")
            return None, None
        
def openMeteoCall(self, timeLapse):
    lat, lon = self.get_current_location()
    
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["weather_code", "temperature_2m_mean", "rain_sum", "showers_sum", "precipitation_sum", "daylight_duration", "relative_humidity_2m_mean"],
        "timezone": "auto",
        "forecast_days": timeLapse
    }
    responses = self.openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    
    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_weather_code = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(1).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(2).ValuesAsNumpy()
    daily_showers_sum = daily.Variables(3).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(4).ValuesAsNumpy()
    daily_daylight_duration = daily.Variables(5).ValuesAsNumpy()
    daily_relative_humidity_2m_mean = daily.Variables(6).ValuesAsNumpy()

    # Return comprehensive data structure
    return {
        "daily_data": {
            "weather_code": daily_weather_code.tolist(),
            "temperature_2m_mean": daily_temperature_2m_mean.tolist(),
            "rain_sum": daily_rain_sum.tolist(),
            "showers_sum": daily_showers_sum.tolist(),
            "precipitation_sum": daily_precipitation_sum.tolist(),
            "daylight_duration": daily_daylight_duration.tolist(),
            "relative_humidity_2m_mean": daily_relative_humidity_2m_mean.tolist()
        },
        "summary": {
            "avg_temperature": float(daily_temperature_2m_mean.mean()),
            "total_precipitation": float(daily_precipitation_sum.sum()),
            "avg_humidity": float(daily_relative_humidity_2m_mean.mean()),
            "total_daylight_hours": float(daily_daylight_duration.sum() / 3600)  # Convert seconds to hours
        }
    }

