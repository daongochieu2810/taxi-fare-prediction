# We are not using this script anymore because we can get data directly from the FTP server
# ftp://ftp.ncdc.noaa.gov/pub/data/
import json
import os
from urllib.request import Request, urlopen

base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2{}"
token = "rEAdHwCdGfjVdOixBpLgrlSNLoWYDOaG"
base_date = "{}-{}-{}"
min_year = 2009
max_year = 2021


def request(url):
    req = Request(url)
    req.add_header("token", token)
    content = urlopen(req).read()

    return json.loads(content.decode("utf8"))


def fetchData(dataset):
    dataset_id = dataset["id"]
    save_path = "weather-data/{}/{}"
    for year in range(min_year, max_year):
        file_path = save_path.format(dataset_id, str(year) + ".txt")
        start_date = base_date.format(year, "01", "01")
        end_date = base_date.format(year + 1, "01", "01")
        params = "?datasetid={}&startdate={}&enddate={}".format(
            dataset_id, start_date, end_date
        )
        url = base_url.format("/data") + params
        data = request(url)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        f = open(file_path, "wb")
        f.write(data)
        f.close()


datasets = request(base_url.format("/datasets"))
for dataset in datasets["results"]:
    fetchData(dataset)
