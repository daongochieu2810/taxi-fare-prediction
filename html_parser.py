from html.parser import HTMLParser


class SimpleHTMLParser(HTMLParser):
    links = []

    def handle_starttag(self, tag, attrs):
        link = ""
        if tag == "a":
            for attr in attrs:
                if attr[0] == "href":
                    link = attr[1]
                    break
        if link != "" and ".csv" in link and "trip+data" in link:
            self.links.append(link)
