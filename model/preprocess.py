import json
import re

INPUT_PATH = "../data/quotes.json"
OUTPUT_PATH = "../data/quotes_clean.txt"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  
    text = text.strip()
    return text

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    quotes = []
    seen = set()

    for item in data:
        quote = item.get("Quote", "").strip()

        if not quote:
            continue

        quote = clean_text(quote)

        if quote not in seen:
            seen.add(quote)
            quotes.append(quote)

    print("Total cleaned quotes:", len(quotes))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for q in quotes:
            f.write(q + "\n")

    print("Saved to quotes_clean.txt")

if __name__ == "__main__":
    main()