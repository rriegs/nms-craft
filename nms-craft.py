#!/usr/bin/env python3

import json
import urllib.request

import pulp


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


# Configuration

MAX_CRAFTING_TIME = 60 * 30.0

initial_stock = {}


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
            "inputs": [(e["Id"], int(e["Quantity"])) for e in r["Inputs"]],
            "output": (r["Output"]["Id"], int(r["Output"]["Quantity"])),
        }
        for r in src
    ]


def parse_crafting_recipes(src):
    return [
        {
            "id": it["Id"],
            "time_s": 1.0,
            "inputs": [(e["Id"], int(e["Quantity"])) for e in it["RequiredItems"]],
            "output": (it["Id"], int(it.get("CraftingOutputAmount", 1))),
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

    profit_per_run = {
        r["id"]: r["output"][1] * items.get(r["output"][0], {"value": 0.0})["value"]
        - sum(q * items.get(iid, {"value": 0.0})["value"] for iid, q in r["inputs"])
        for r in recipes
    }

    print(f"Parsed {len(items)} items and {len(recipes)} recipes")

    # Create LP problem and variables representing recipe runs
    prob = pulp.LpProblem("nms_max_units", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", [r["id"] for r in recipes], lowBound=0)

    # Time constraint: total crafting time must not exceed the time budget
    prob += pulp.lpSum(r["time_s"] * x[r["id"]] for r in recipes) <= MAX_CRAFTING_TIME

    # Objective: maximize total units value produced
    prob += pulp.lpSum(profit_per_run[r["id"]] * x[r["id"]] for r in recipes)

    # Item stock nonnegativity constraints
    item_expr = {}
    for r in recipes:
        for iid, q in r["inputs"]:
            item_expr[iid] = item_expr.get(iid, 0) - q * x[r["id"]]
        out_id, out_q = r["output"]
        item_expr[out_id] = item_expr.get(out_id, 0) + out_q * x[r["id"]]
    for iid, expr in item_expr.items():
        prob += initial_stock.get(iid, 0) + expr >= 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    total_time = sum(r["time_s"] * (x[r["id"]].value() or 0) for r in recipes)
    chosen_recipes = []
    for r in recipes:
        xr = x[r["id"]].value() or 0
        if xr > 1e-9:
            out_name = items.get(r["output"][0], {"name": r["output"][0]})["name"]
            chosen_recipes.append(
                (r["id"], out_name, xr, r["time_s"] * xr, profit_per_run[r["id"]] * xr)
            )

    print(f"Parsed {len(items)} items and {len(recipes)} recipes")
    print("Status:", pulp.LpStatus[prob.status])
    print(f"Total profit: {pulp.value(prob.objective):.2f}")
    print(f"Crafting time: {total_time/60.0:.2f} min")
    print("Chosen recipes:")
    for rid, oname, xr, t, p in sorted(chosen_recipes, key=lambda t: -t[4]):
        print(f" - {rid}: runs={xr:.2f}, time={t:.1f}s, profit={p:.1f} -> {oname}")


if __name__ == "__main__":
    main()
