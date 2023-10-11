import requests 
from time import sleep
years = list(range(1992,2022))
url_start = "https://www.basketball-reference.com/awards/awards_{}.html"

for year in years:
    url = url_start.format(year)

    data = requests.get(url)

    with open("mvp/{}.html".format(year), "w+", encoding='utf-8') as f:
        f.write(data.text)
    sleep(30)
