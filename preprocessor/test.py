import os
from .merger import Merger


def test():
    merger = Merger()
    merger.load_weather_data("sample-weather-data/2016_01.txt")
