import os
import requests
from dotenv import load_dotenv

load_dotenv()
params={"lat":37.57, "lon":127.0}
API_key=os.environ.get('WEATHER_API_KEY')
url=f'http://api.weatherapi.com/v1/current.json?key={API_key}&q=Seoul&aqi=yes'

response = requests.get(url,params=params)
print(response.content)
