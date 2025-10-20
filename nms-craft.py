#!/usr/bin/env python3

import json
import urllib.request

# Data sources (AssistantNMS GitHub repository)

URL_DIR_ASSISTANT_RAW = (
    "https://raw.githubusercontent.com/AssistantNMS/App/refs/heads/main/assets/json/en/"
)

URL_REFINERY = URL_DIR_ASSISTANT_RAW + "Refinery.lang.json"
URL_NUTRIENT = URL_DIR_ASSISTANT_RAW + "NutrientProcessor.lang.json"
URL_RAWMATS = URL_DIR_ASSISTANT_RAW + "RawMaterials.lang.json"
URL_PRODUCTS = URL_DIR_ASSISTANT_RAW + "Products.lang.json"
URL_CURIOSITY = URL_DIR_ASSISTANT_RAW + "Curiosity.lang.json"
URL_COOKING = URL_DIR_ASSISTANT_RAW + "Cooking.lang.json"


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def index_items(sources):
    return {
        it["Id"]: {
            "name": it["Name"],
            "value": (
                float(it["BaseValueUnits"]) if it["CurrencyType"] == "Credits" else 0.0
            ),
        }
        for src in sources
        for it in src
    }


def parse_machine_recipes(src):
    return [
        {
            "id": r["Id"],
            "time_s": float(r["Time"]),
            "inputs": [(e["Id"], float(e["Quantity"])) for e in r["Inputs"]],
            "output": (r["Output"]["Id"], float(r["Output"]["Quantity"])),
        }
        for r in src
    ]


def parse_crafting_recipes(src):
    return [
        {
            "id": it["Id"],
            "time_s": 1.0,
            "inputs": [(e["Id"], float(e["Quantity"])) for e in it["RequiredItems"]],
            "output": (it["Id"], float(it.get("CraftingOutputAmount", 1))),
        }
        for it in src
        if it["RequiredItems"]
    ]


def main():
    ref = fetch(URL_REFINERY)
    nut = fetch(URL_NUTRIENT)
    raw = fetch(URL_RAWMATS)
    prod = fetch(URL_PRODUCTS)
    cur = fetch(URL_CURIOSITY)
    cook = fetch(URL_COOKING)

    items = index_items([raw, prod, cur, cook])

    recipes = (
        parse_machine_recipes(ref)
        + parse_machine_recipes(nut)
        + parse_crafting_recipes(raw)
        + parse_crafting_recipes(prod)
        + parse_crafting_recipes(cur)
        + parse_crafting_recipes(cook)
    )

    print(f"Parsed {len(items)} items and {len(recipes)} recipes")


if __name__ == "__main__":
    main()
