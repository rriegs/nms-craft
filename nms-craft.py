#!/usr/bin/env python3

import argparse
from collections import defaultdict
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

BIG_M_CRAFTING_TIME = 60 * 60 * 10.0  # 10 hours
MAX_CRAFTING_TIME = 60 * 30.0
SWITCHING_PENALTY = 40.0

# Starting amounts of each item (by ID or name); negative values indicate desire
initial_stock = {
    "activated cadmium": 1053,
    "activated copper": 36,
    "activated indium": 1213,
    "atlanteum": 848,
    "cactus flesh": 2385,
    "cadmium": 237,
    "carbon": 9999,
    "cobalt": 72,
    "copper": 1439,
    "di-hydrogen": 423,
    "dioxide": 380,
    "emeril": 561,
    "ferrite dust": 9307,
    "frost crystal": 3455,
    "fungal mould": 464,
    "gold": 928 + 9999,
    "indium": 3405,
    "nitrogen": 1322,
    "oxygen": 9999,
    "parafinium": 30,
    "pugneum": 9949,
    "radon": 2083,
    "silicate powder": 283,
    "sodium": 2649,
    "sodium nitrate": 1440,
    "tritium": 4075,
    "uranium": 2046,
    "aloe flesh": 82,
    "chromatic metal": 5286,
    "craw milk": 4,
    "crystal sulfide": 7,
    "crystallised heart": 1,
    "frozen tubers": 24,
    "hadal core": 9,
    "hardframe engine": 19,
    "hex core": 3,
    "hexaberry": 10,
    "hypnotic eye": 5,
    "impluse beans": 21,
    "inverted mirror": 1,
    "jade peas": 104,
    "lava core": 32,
    "mordite": 7576 + 9999,
    "pirate transponder": 28,
    "pulpy roots": 11,
    "quad servo": 16,
    "radiant shard": 10,
    "rancid flesh": 64,
    "raw steak": 5,
    "repair kit": 2,
    "storm crystal": 34,
    "sweetroot": 55,
    "walker brain": 5,
}

# Recipes to ignore (by ID, output ID, or output name)
ignored_recipes = [
    "warp cell",
]


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
            "stack_size": float(it.get("MaxStackSize", 1.0)),
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
    parser.add_argument(
        "-i",
        "--integer",
        action="store_true",
        help="ensure recipe run counts are integers",
    )
    parser.add_argument(
        "-m",
        "--minimize",
        action="store_true",
        help="report minimal stocks needed to craft desired items",
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
    name_to_id = {v["name"].lower(): k for k, v in items.items()}

    # Normalize initial stock to use item IDs (if not already provided)
    s0 = defaultdict(float)
    for k, v in initial_stock.items():
        s0[name_to_id.get(k.lower(), k)] += v

    recipes = (
        parse_machine_recipes(ref)
        + parse_machine_recipes(nut)
        + parse_crafting_recipes(raw)
        + parse_crafting_recipes(prod)
        + parse_crafting_recipes(cur)
        + parse_crafting_recipes(cook)
    )

    # Filter recipes by ID, output ID, or output name
    ignored = set(name_to_id.get(k.lower(), k) for k in ignored_recipes)
    recipes = [
        r for r in recipes if r["id"] not in ignored and r["output"][0] not in ignored
    ]

    profit_per_run = {
        r["id"]: r["output"][1] * items.get(r["output"][0], {"value": 0.0})["value"]
        - sum(q * items.get(iid, {"value": 0.0})["value"] for iid, q in r["inputs"])
        for r in recipes
    }

    print(f"Parsed {len(items)} items and {len(recipes)} recipes")

    # Create LP problem and variables representing recipe runs
    if args.minimize:
        prob = pulp.LpProblem("nms_min_stock", pulp.LpMinimize)
    elif max_crafting_time <= 0:
        prob = pulp.LpProblem("nms_min_time", pulp.LpMinimize)
    else:
        prob = pulp.LpProblem("nms_max_units", pulp.LpMaximize)
    cat = pulp.LpInteger if args.integer else pulp.LpContinuous
    x = pulp.LpVariable.dicts("x", [r["id"] for r in recipes], lowBound=0, cat=cat)
    b = pulp.LpVariable.dicts("b", [r["id"] for r in recipes], cat=pulp.LpBinary)
    m = pulp.LpVariable.dicts("m", items.keys(), lowBound=0, cat=pulp.LpInteger)
    p = pulp.LpVariable.dicts(
        "p", items.keys(), lowBound=0, upBound=len(recipes), cat=pulp.LpContinuous
    )

    if switching_penalty > 0 or args.minimize:
        # Track which recipes have nonzero usage counts
        for r in recipes:
            prob += r["time_s"] * x[r["id"]] <= BIG_M_CRAFTING_TIME * b[r["id"]]

    if max_crafting_time > 0:
        # Time constraint: crafting time (and penalty) must not exceed the time budget
        if switching_penalty > 0:
            prob += (
                pulp.lpSum(r["time_s"] * x[r["id"]] for r in recipes)
                + switching_penalty * pulp.lpSum(b[r["id"]] for r in recipes)
                <= max_crafting_time
            )
        else:
            prob += (
                pulp.lpSum(r["time_s"] * x[r["id"]] for r in recipes)
                <= max_crafting_time
            )

    if args.minimize:
        # Objective: minimize total extra stock needed
        prob += pulp.lpSum(m[iid] for iid in items)

    elif max_crafting_time > 0:
        # Objective: maximize total units value produced (profit)
        prob += pulp.lpSum(profit_per_run[r["id"]] * x[r["id"]] for r in recipes)

    elif switching_penalty > 0:
        # Objective: minimize total crafting time and penalties (ignore profit)
        prob += pulp.lpSum(
            r["time_s"] * x[r["id"]] for r in recipes
        ) + switching_penalty * pulp.lpSum(b[r["id"]] for r in recipes)

    else:
        # Objective: just minimize total crafting time
        prob += pulp.lpSum(r["time_s"] * x[r["id"]] for r in recipes)

    # Item stock nonnegativity constraints
    item_expr = {}
    for r in recipes:
        for iid, q in r["inputs"]:
            item_expr[iid] = item_expr.get(iid, 0) - q * x[r["id"]]
        out_id, out_q = r["output"]
        item_expr[out_id] = item_expr.get(out_id, 0) + out_q * x[r["id"]]
    if args.minimize:
        for iid, expr in item_expr.items():
            stack_size = items.get(iid, {"stack_size": 1.0})["stack_size"]
            prob += s0.get(iid, 0) + expr + stack_size * m.get(iid, 0) >= 0
    else:
        for iid, expr in item_expr.items():
            prob += s0.get(iid, 0) + expr >= 0

    if args.minimize:
        # Break cycles
        for r in recipes:
            out_id, _ = r["output"]
            for iid, _ in r["inputs"]:
                prob += p[out_id] + len(recipes) * (1 - b[r["id"]]) >= p.get(iid, 0) + 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))

    total_time = sum(r["time_s"] * (x[r["id"]].value() or 0) for r in recipes)
    total_profit = sum(
        profit_per_run[r["id"]] * (x[r["id"]].value() or 0) for r in recipes
    )
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
    print(f"Total profit: {total_profit:.2f}")
    print(f"Crafting time: {total_time/60.0:.2f} min")
    print("Chosen recipes:")
    for rid, xr, t, p, rstr in sorted(chosen_recipes, key=lambda t: -t[3]):
        runs = f"{xr:.0f}" if args.integer else f"{xr:.2f}"
        print(f"  {rid}: runs={runs}, time={t:.1f}s, profit={p:.1f}\n    {rstr}")
    if args.minimize:
        print("Additional stock needed:")
        for iid in items:
            mi = m[iid].value() or 0
            if mi > 1e-9:
                print(f"  {fmt_item(iid, items[iid]["stack_size"] * mi, OUTPUT_COLOR)}")


if __name__ == "__main__":
    main()
