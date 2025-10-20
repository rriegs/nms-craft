#!/usr/bin/env python3

import argparse
import json
import sys
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


# Terminal colors

RESET_COLOR = "\033[0m"
INPUT_COLOR = "\033[36m"  # cyan
OUTPUT_COLOR = "\033[32m"  # green


# Configuration

MAX_CRAFTING_TIME = 60 * 30.0
SWITCHING_PENALTY = 40.0

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
    # Parse command-line options to override configuration
    parser = argparse.ArgumentParser(
        description="Optimize crafting for maximum profit within a time budget"
    )
    parser.add_argument(
        "-b",
        "--budget",
        metavar="MINUTES",
        type=float,
        default=MAX_CRAFTING_TIME / 60,
        help=f"maximum crafting time in minutes (default: {MAX_CRAFTING_TIME/60})",
    )
    parser.add_argument(
        "-p",
        "--penalty",
        metavar="AMOUNT",
        type=float,
        default=SWITCHING_PENALTY,
        help=f"per-recipe switching penalty in seconds (default: {SWITCHING_PENALTY})",
    )
    args = parser.parse_args()
    max_crafting_time = args.budget * 60
    switching_penalty = args.penalty

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
    b = pulp.LpVariable.dicts("b", [r["id"] for r in recipes], cat=pulp.LpBinary)

    # Track which recipes have nonzero usage counts
    for r in recipes:
        prob += r["time_s"] * x[r["id"]] <= max_crafting_time * b[r["id"]]

    # Time constraint: total crafting time must not exceed the time budget
    prob += (
        pulp.lpSum(r["time_s"] * x[r["id"]] for r in recipes)
        + switching_penalty * pulp.lpSum(b[r["id"]] for r in recipes)
        <= max_crafting_time
    )

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

    def fmt_item(iid, qty, color):
        name = items.get(iid, {"name": iid})["name"]
        if sys.stdout.isatty():
            return f"{qty:.0f} {color}{name}{RESET_COLOR}"
        else:
            return f"{qty:.0f} {name}"

    for r in recipes:
        xr = x[r["id"]].value() or 0
        if xr > 1e-9:
            time = r["time_s"] * xr
            profit = profit_per_run[r["id"]] * xr
            rstr = " + ".join(fmt_item(iid, q, INPUT_COLOR) for iid, q in r["inputs"])
            rstr += " -> " + fmt_item(r["output"][0], r["output"][1], OUTPUT_COLOR)
            chosen_recipes.append((r["id"], xr, time, profit, rstr))

    print("Status:", pulp.LpStatus[prob.status])
    print(f"Total profit: {pulp.value(prob.objective):.2f}")
    print(f"Crafting time: {total_time/60.0:.2f} min")
    print("Chosen recipes:")
    for rid, xr, t, p, rstr in sorted(chosen_recipes, key=lambda t: -t[3]):
        print(f"  {rid}: runs={xr:.2f}, time={t:.1f}s, profit={p:.1f}\n    {rstr}")


if __name__ == "__main__":
    main()
