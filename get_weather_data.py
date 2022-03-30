from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os

save_path = "weather-data/{}_{}.txt"
driver = webdriver.Firefox()
url = "https://nowdata.rcc-acis.org/okx/"
driver.get(url)

time.sleep(5)

datePicker = driver.find_element_by_id("tDatepicker")
locationSelect = driver.find_element_by_name("station")
goButton = driver.find_element_by_id("go")
results = driver.find_element_by_id("results_area")
calenderButton = driver.find_element_by_class_name("ui-datepicker-trigger")

columns = [
    "date",
    "tempMax",
    "tempMin",
    "tempAvg",
    "tempDeparture",
    "hdd",
    "cdd",
    "precipitation",
    "newSnow",
    "snowDepth",
]
options = locationSelect.find_elements_by_css_selector("option")
for year in range(2016, 2022):
    for i in range(1, 13):
        month = str(i)
        if i < 10:
            month = "0" + str(i)

        datePicker.clear()
        datePicker.send_keys(str(year) + "-" + month)
        webdriver.ActionChains(driver).send_keys(Keys.TAB).perform()
        time.sleep(0.5)

        dateData = {}
        for optionId in range(len(options)):
            option = options[optionId]
            if "NY" not in option.text:
                continue
            stationData = []
            try:
                option.click()
                goButton.click()
                time.sleep(5)

                table = results.find_element_by_css_selector("table")
                body = results.find_element_by_css_selector("tbody")
                rows = body.find_elements_by_css_selector("tr")
                for row in rows:
                    tds = row.find_elements_by_css_selector("td")
                    rowData = {}
                    for tdId in range(len(tds)):
                        td = tds[tdId]
                        rowData[columns[tdId]] = td.text
                    stationData.append(rowData)
                dateData[option.text] = stationData
            except Exception as e:
                print(e)
                print("Error! with " + str(year) + "-" + month + " and " + option.text)
            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)
            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)
            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)
            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)

        save_file = save_path.format(year, month)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        f = open(save_file, "w")
        f.write(str(dateData))
        f.close()

driver.close()
