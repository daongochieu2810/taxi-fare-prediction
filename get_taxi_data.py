import os
import urllib.request as urllib2
from html_parser import SimpleHTMLParser


def extract_meta_data(link):
    file_name = link.split("/")[-1]
    tag_pos = link.find(".csv")
    month = link[tag_pos - 2 : tag_pos]
    year = link[tag_pos - 7 : tag_pos - 3]
    return file_name, month, year


url = "https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
parser = SimpleHTMLParser()
page = urllib2.urlopen(url)
parser.feed(page.read().decode("utf8"))

save_path = "data/{}/{}/{}"
for link in parser.links:
    file_name, month, year = extract_meta_data(link)
    save_file = save_path.format(year, month, file_name)
    u = urllib2.urlopen(link)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    f = open(save_file, "wb")
    meta = u.info()
    file_size = int(meta.get_all("Content-Length")[0])
    # print("Downloading: %s Bytes: %s" % (file_name, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100.0 / file_size)
        status = status + chr(8) * (len(status) + 1)
        print(status)

    f.close()
