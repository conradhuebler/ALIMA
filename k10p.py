import requests
from lxml import etree

# SRU-Abfrage
sru_url = "https://services.dnb.de/sru/dnb"
params = {
    "version": "1.1",
    "operation": "searchRetrieve",
    "query": "subject=NMR or subject=Bergbau",
    "recordSchema": "MARC21-xml"
}

# Anfrage senden
response = requests.get(sru_url, params=params)
xml_data = response.content

# XML parsen
root = etree.fromstring(xml_data)
namespaces = {"marc": "http://www.loc.gov/MARC21/slim"}

# Schlagworte extrahieren
for record in root.xpath("//marc:record", namespaces=namespaces):
    for subject in record.xpath(".//marc:datafield[@tag='650']", namespaces=namespaces):
        keyword = subject.xpath(".//marc:subfield[@code='a']/text()", namespaces=namespaces)
        if keyword:
            print(keyword[0])
