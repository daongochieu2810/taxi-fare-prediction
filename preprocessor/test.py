from .merger import Merger


def test():
    merger = Merger()
    merger.load_weather_data("sample-weather-data/2016_01.txt")
    merger.load_taxi_data("sample-data/2020/01/green_tripdata_2020-01.csv")

    merger.join()

    assert merger.merged_df is not None
