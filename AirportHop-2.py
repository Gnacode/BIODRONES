#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_H4_scenarios.py

Builds a 4-hop air travel infection spread model from seed airports
(AAL, BLL, CPH) under 3 exposure scenarios (2.5%, 5%, 10%).

NOW EXTENDED:
Also writes a "_mask" version assuming universal masking at 50% effectiveness.
That is applied as a 0.5 multiplier on infected passenger counts.

Inputs:
  - airports.dat (OpenFlights format)
  - routes.dat   (OpenFlights format)

Outputs (for each scenario A/B/C):
  - expected_infected_passengers_H4_scnX.csv
  - expected_infected_passengers_H4_scnX_mask.csv
        rows = seed airports (AAL,BLL,CPH) plus ALL_SEEDS_scnX
        cols = all airports in network
        values ~ expected infected pax sourced at those seeds
                 that reach airport j within 4 hops (sum day0..4)

  - cumulative_infection_risk_H4_scnX.csv
        square matrix (all airports x all airports)
        entry[i, j] ~ relative reach potential from i to j within 4 hops
        (probability-like score, not passenger-weighted)

ASSUMPTIONS:
- DAILY_PAX per seed airport = average daily passenger volume.
- EXPOSED_FRACTION of that day's pax are physically exposed to plume.
- SCENARIOS[...] = attack rate on those exposed.
- MAX_HOPS = 4 days/hops of movement.
- No secondary infection, just redistribution of initially infected travelers.
- Mask effectiveness = 50% means we cut expected infected travelers by 0.5.
"""

from __future__ import annotations
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

AIRPORTS_DAT_PATH = "airports.dat"
ROUTES_DAT_PATH   = "routes.dat"

# Approximate average daily pax through each seed airport.
DAILY_PAX = {
    "AAL": 4000,    # passengers / day through Aalborg
    "BLL": 15000,   # passengers / day through Billund
    "CPH": 80000,   # passengers / day through Copenhagen
}

# Fraction of that day's pax who are physically in plume during release window.
EXPOSED_FRACTION = 0.10  # 10%

# Infection scenarios (attack rates on exposed set)
SCENARIOS = {
    "A": 0.025,   # 2.5%
    "B": 0.05,    # 5%
    "C": 0.10,    # 10%
}

# Universal masking protection factor
MASK_EFFECTIVENESS = 0.5  # 50% reduction

# How many outbound destinations to keep per origin (avoid network explosion)
MAX_DEST_PER_ORIGIN = 4

# Geographic filter for "Europe-ish"
# (lon_min, lon_max, lat_min, lat_max)
EURO_BOUNDS = (-25.0, 45.0, 30.0, 72.0)

# Hops / days to propagate
MAX_HOPS = 4

# Seed airports (where exposure happens)
SEED_AIRPORTS = ["AAL", "BLL", "CPH"]


# ---------------------------------------------------------------------
# STEP 1. LOAD AIRPORTS
# ---------------------------------------------------------------------
def load_airports(airports_path: str):
    """
    airports.dat columns (OpenFlights):
      0 Airport ID
      1 Name
      2 City
      3 Country
      4 IATA
      5 ICAO
      6 Latitude
      7 Longitude
      ...
    We keep IATA, lat, lon for airports inside EURO_BOUNDS.
    """
    airports_info = {}
    with open(airports_path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 8:
                continue
            try:
                iata = row[4].strip().upper()
                lat  = float(row[6])
                lon  = float(row[7])
            except Exception:
                continue
            if not iata or iata == r"\"\"":
                continue
            # Europe-ish filter
            if (
                lon < EURO_BOUNDS[0] or lon > EURO_BOUNDS[1] or
                lat < EURO_BOUNDS[2] or lat > EURO_BOUNDS[3]
            ):
                continue

            country = row[3].strip('"')
            city    = row[2].strip('"')
            airports_info[iata] = {
                "lat": lat,
                "lon": lon,
                "country": country,
                "city": city,
            }
    return airports_info


# ---------------------------------------------------------------------
# STEP 2. LOAD ROUTES
# ---------------------------------------------------------------------
def load_routes(routes_path: str, airports_info: dict):
    """
    routes.dat columns (OpenFlights):
      0 Airline
      1 Airline ID
      2 Source airport (IATA)
      3 Source airport ID
      4 Destination airport (IATA)
      5 Destination airport ID
      ...
    Return { origin: [dest, dest, ...] } but only keep routes where both
    ends survived the Europe filter.
    """
    routes_dict = defaultdict(list)
    with open(routes_path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 6:
                continue
            src = row[2].strip().upper()
            dst = row[4].strip().upper()
            if src in airports_info and dst in airports_info and src and dst:
                routes_dict[src].append(dst)
    return routes_dict


def restrict_topk_routes(routes_dict: dict, k: int):
    """
    For each origin, keep up to k unique destinations in first-seen order.
    """
    new_routes = {}
    for src, dest_list in routes_dict.items():
        uniq = []
        for d in dest_list:
            if d not in uniq:
                uniq.append(d)
            if len(uniq) >= k:
                break
        new_routes[src] = uniq
    return new_routes


# ---------------------------------------------------------------------
# STEP 3. TRANSITION MATRIX
# ---------------------------------------------------------------------
def build_transition_matrix(airports_info: dict, routes_topk: dict):
    """
    Build row-stochastic P (N x N), where each row i splits evenly
    among that airport's kept outbound destinations.
    Returns:
      P            (np.ndarray [N,N])
      airport_ix   (dict IATA->row/col index)
      airports_list (list index->IATA)
    """
    all_airports = sorted(airports_info.keys())
    N = len(all_airports)
    airport_ix = {iata: idx for idx, iata in enumerate(all_airports)}

    P = np.zeros((N, N), dtype=float)

    for src, dests in routes_topk.items():
        if src not in airport_ix:
            continue
        i = airport_ix[src]
        if not dests:
            continue
        w = 1.0 / float(len(dests))
        for d in dests:
            if d not in airport_ix:
                continue
            j = airport_ix[d]
            P[i, j] += w

    return P, airport_ix, all_airports


# ---------------------------------------------------------------------
# STEP 4. POWERS OF P
# ---------------------------------------------------------------------
def matrix_powers(P: np.ndarray, max_hops: int):
    """
    Compute [I, P, P^2, ..., P^max_hops].
    powers[h] = P^h
    """
    N = P.shape[0]
    powers = [np.eye(N, dtype=float)]
    cur = np.eye(N, dtype=float)
    for h in range(1, max_hops + 1):
        cur = cur @ P
        powers.append(cur.copy())
    return powers


# ---------------------------------------------------------------------
# STEP 5. INITIAL INFECTED VECTOR
# ---------------------------------------------------------------------
def build_initial_infected_vector(airports_list, daily_pax, exposed_fraction, attack_rate):
    """
    Create I0_total (length N) giving initially infected travelers at each airport.
    We do:
        exposed = EXPOSED_FRACTION * DAILY_PAX[seed]
        infected = exposed * attack_rate
    for each seed airport, then sum across seeds.
    Also return per-seed vectors (seed alone).
    """
    N = len(airports_list)
    I0_total = np.zeros(N, dtype=float)
    per_seed_vectors = {}

    for seed in SEED_AIRPORTS:
        if seed not in airports_list:
            continue
        idx = airports_list.index(seed)

        pax_seed = daily_pax.get(seed, 0.0)
        exposed = exposed_fraction * pax_seed
        infected = exposed * attack_rate

        vec = np.zeros(N, dtype=float)
        vec[idx] = infected

        I0_total += vec
        per_seed_vectors[seed] = vec

    return I0_total, per_seed_vectors


# ---------------------------------------------------------------------
# STEP 6. PROPAGATION OVER HOPS
# ---------------------------------------------------------------------
def propagate_infected(I0_vec: np.ndarray, powers: list[np.ndarray]):
    """
    For each hop h (0..H):
      day_h = I0_vec @ P^h
    We also sum all days to get cumulative arrivals within <= H hops.
    """
    per_day = []
    for h, Ph in enumerate(powers):
        day_h = I0_vec @ Ph
        per_day.append(day_h)
    cumulative = np.sum(per_day, axis=0)
    return per_day, cumulative


# ---------------------------------------------------------------------
# STEP 7. EXPECTED INFECTED DF
# ---------------------------------------------------------------------
def make_expected_infected_df(per_seed_vectors, powers, airports_list):
    """
    One row per seed (AAL,BLL,CPH,...), columns = all airports.
    Values = total infected arriving within <= MAX_HOPS hops from that seed.
    """
    data = {}
    for seed, seed_vec in per_seed_vectors.items():
        _, cum = propagate_infected(seed_vec, powers)
        data[seed] = cum
    df = pd.DataFrame(data).T
    df.columns = airports_list
    return df


# ---------------------------------------------------------------------
# STEP 8. CUMULATIVE REACH DF
# ---------------------------------------------------------------------
def make_cumrisk_df(powers, airports_list):
    """
    CumReach = I + P + P^2 + ... + P^MAX_HOPS
    entry[i,j] ~ probability mass that a single infected traveler at i
                 can show up in j within <= MAX_HOPS hops.
    """
    CumReach = np.sum(powers, axis=0)
    df = pd.DataFrame(CumReach, index=airports_list, columns=airports_list)
    return df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # 1. Airports and routes
    airports_info = load_airports(AIRPORTS_DAT_PATH)
    raw_routes    = load_routes(ROUTES_DAT_PATH, airports_info)
    routes_topk   = restrict_topk_routes(raw_routes, MAX_DEST_PER_ORIGIN)

    # 2. Transition matrix
    P, airport_ix, airports_list = build_transition_matrix(airports_info, routes_topk)

    # 3. Powers of P up to MAX_HOPS
    powers = matrix_powers(P, MAX_HOPS)

    # 4. Cumulative reachability (same for mask/no-mask)
    cumrisk_df = make_cumrisk_df(powers, airports_list)

    # 5. For each scenario, compute infected passenger flows
    for scen_name, attack_rate in SCENARIOS.items():
        print(f"Scenario {scen_name}: attack_rate={attack_rate}")

        # Build initial infected travelers (unmasked case)
        I0_total, per_seed_vectors = build_initial_infected_vector(
            airports_list,
            DAILY_PAX,
            EXPOSED_FRACTION,
            attack_rate
        )

        # Propagate ALL seeds combined (unmasked)
        per_day_all, cumulative_all = propagate_infected(I0_total, powers)

        # Row for ALL_SEEDS
        cumulative_all_df = pd.DataFrame(
            [cumulative_all],
            index=[f"ALL_SEEDS_scn{scen_name}"],
            columns=airports_list,
        )

        # Per-seed breakdown (unmasked)
        expected_infected_df = make_expected_infected_df(
            per_seed_vectors,
            powers,
            airports_list
        )

        # Combine per-seed + ALL row (unmasked)
        combo_df = pd.concat([expected_infected_df, cumulative_all_df])

        # ----- MASKED VERSION -----
        # Universal 50% mask means we cut expected infected travelers by (1 - MASK_EFFECTIVENESS)
        mask_scale = (1.0 - MASK_EFFECTIVENESS)  # e.g. 0.5
        combo_mask_df = combo_df * mask_scale

        # 6. Save CSVs
        expected_path        = f"expected_infected_passengers_H4_scn{scen_name}.csv"
        expected_mask_path   = f"expected_infected_passengers_H4_scn{scen_name}_mask.csv"
        cumrisk_path         = f"cumulative_infection_risk_H4_scn{scen_name}.csv"
        cumrisk_mask_path    = f"cumulative_infection_risk_H4_scn{scen_name}_mask.csv"

        # Note: cumrisk_df itself does not depend on absolute passenger counts,
        # but downstream code expects a *_mask.csv too, so we can reuse it unchanged.
        combo_df.to_csv(expected_path, float_format="%.6f")
        combo_mask_df.to_csv(expected_mask_path, float_format="%.6f")
        cumrisk_df.to_csv(cumrisk_path, float_format="%.6f")
        cumrisk_df.to_csv(cumrisk_mask_path, float_format="%.6f")

        print(f"  wrote {expected_path}")
        print(f"  wrote {expected_mask_path}")
        print(f"  wrote {cumrisk_path}")
        print(f"  wrote {cumrisk_mask_path}")


if __name__ == "__main__":
    main()
