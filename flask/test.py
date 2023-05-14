import os
import requests
import json
from ast import literal_eval
from dotenv import load_dotenv

load_dotenv()
params={"lat":37.57, "lon":127.0}
API_key=os.environ.get('WEATHER_API_KEY')
url=f'http://api.weatherapi.com/v1/current.json?key={API_key}&q=Seoul&aqi=yes'

response = requests.get(url,params=params)
data = literal_eval(response.content.decode('utf-8'))
with open('./apitest.json','w') as j:
    json.dump(data,j,indent=4)
# temp_c 현재기온(섭씨)
# is_day 낮밤
# wind_kph 풍속
# pressure_mb 기압
# precip_mm 강수량(mm)
# humidity 습도(%)
# cloud 구름낌(%)
# feelslike_c 체감온도(섭씨)
# 
