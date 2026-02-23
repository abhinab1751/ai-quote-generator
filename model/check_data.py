import json

with open("../data/quotes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Type of data:", type(data))

if isinstance(data, list):
    print("Total entries:", len(data))
    print("First item:\n", data[0])

elif isinstance(data, dict):
    print("Keys:", data.keys())