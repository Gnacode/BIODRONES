import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from math import sqrt
from geopy.distance import great_circle
import base64
from io import BytesIO
import logging
import time
import sys

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

t_start = time.time()
def log_timing(msg, t0=[t_start]):
    elapsed = time.time() - t0[0]
    logging.info(f"[{elapsed:7.2f} s] {msg}")

logging.info("=== Script started ===")

# -----------------------
# Parameters
# -----------------------
# Each tuple: (file stem for CSVs, human-readable label, suffix for filenames, mask_flag)
# mask_flag is just for color grouping in charts later.
scenarios = [
    ("scnA",      "Low (no mask)",        "low",         False),
    ("scnB",      "Medium (no mask)",     "medium",      False),
    ("scnC",      "High (no mask)",       "high",        False),
    ("scnA_mask", "Low (mask 50%)",       "low_mask",    True),
    ("scnB_mask", "Medium (mask 50%)",    "medium_mask", True),
    ("scnC_mask", "High (mask 50%)",      "high_mask",   True),
]

infectivity_profile = {1: 0.05, 2: 0.25, 3: 0.60, 4: 0.85}
CRUISING_SPEED_KMH = 850.0

logging.info(f"Scenarios: {[s[1] for s in scenarios]}")
logging.info("Loading airport coordinates...")

# -----------------------
# Load airport coordinates
# -----------------------
airports_df = pd.read_csv("airports.dat", header=None)
airports_df.columns = [
    "Airport ID", "Name", "City", "Country", "IATA", "ICAO",
    "Latitude", "Longitude", "Altitude", "Timezone", "DST",
    "Tz database time zone", "Type", "Source"
]

coords = {
    row.IATA: (row.Latitude, row.Longitude)
    for _, row in airports_df.iterrows()
    if pd.notna(row.IATA)
}

logging.info(f"Loaded {len(coords)} airports with IATA codes.")
log_timing("Finished loading airport data")

# -----------------------
# Precompute duration matrix and weight matrix
# -----------------------
logging.info("Precomputing duration matrix (this can be slow for many airports)...")
iata_list = list(coords.keys())

duration_matrix = pd.DataFrame(index=iata_list, columns=iata_list, dtype=float)

pairs = len(iata_list) * len(iata_list)
logging.info(f"Computing great-circle travel times for ~{pairs} pairs...")

route_count = 0
for src in iata_list:
    latlon_src = coords[src]
    for dst in iata_list:
        latlon_dst = coords[dst]
        dist = great_circle(latlon_src, latlon_dst).km
        duration_hr = dist / CRUISING_SPEED_KMH
        duration_matrix.loc[src, dst] = duration_hr
        route_count += 1
        if route_count % 10000 == 0:
            logging.info(
                f"...computed {route_count}/{pairs} pairs "
                f"({100.0*route_count/pairs:4.1f}%)"
            )

log_timing("Finished duration matrix")

def get_duration_weight(d):
    if d >= 2.0:
        return 1.0
    elif d >= 1.0:
        return 0.75
    elif d > 0:
        return 0.25
    return 0.0

logging.info("Computing weight matrix from duration matrix...")
weight_matrix = duration_matrix.applymap(
    lambda d: get_duration_weight(d) if pd.notna(d) else 0.0
)
log_timing("Finished weight matrix")

# -----------------------
# Storage for results
# -----------------------
scenario_maps          = {}  # scen_label -> (folium.Map, suffix)
scenario_summaries     = {}  # scen_label -> Series(day1..day4 totals)
scenario_day_details   = {}  # scen_label -> {day: Series(IATA->infected arrivals)}

# -----------------------
# Process each scenario
# -----------------------
for (scen_key, scen_label, scen_suffix, is_mask) in scenarios:
    logging.info(f"--- Scenario: {scen_label} ({scen_key}) ---")
    t_scn_start = time.time()

    fname_risk = f"cumulative_infection_risk_H4_{scen_key}.csv"
    fname_pass = f"expected_infected_passengers_H4_{scen_key}.csv"

    logging.info(f"Loading infect_risk matrix: {fname_risk}")
    infect_risk = pd.read_csv(fname_risk, index_col=0)

    logging.info(f"Loading passenger_vol matrix: {fname_pass}")
    passenger_vol = pd.read_csv(fname_pass, index_col=0)

    # Compute infections per day
    logging.info("Calculating infections per day (days 1-4)...")
    day_infections = {}
    for day in range(1, 5):
        logging.info(f" Day {day}: computing weighted infections...")
        risk_day = infect_risk * weight_matrix * infectivity_profile[day]
        day_infections[day] = (risk_day * passenger_vol).sum(axis=0)

    # Store per-day detail for this scenario
    scenario_day_details[scen_label] = day_infections

    # Total infected arrivals per day across all airports
    scenario_summaries[scen_label] = (
        pd.DataFrame.from_dict(day_infections, orient="index")
        .sum(axis=1)
        .astype(int)
    )
    logging.info(f"Summary for {scen_label}:\n{scenario_summaries[scen_label]}")

    # Build folium map (map only, we'll save separately)
    logging.info("Rendering folium map...")
    m = folium.Map(location=[55.0, 10.0], zoom_start=4, tiles="CartoDB positron")

    colors_by_day = {1: "blue", 2: "green", 3: "orange", 4: "red"}
    scales_by_day = {1: 3, 2: 5, 3: 7, 4: 9}

    point_counter = 0
    # draw Day 4 first so Day 1 sits on top
    for day in [4, 3, 2, 1]:
        for iata, infections in day_infections[day].items():
            if infections <= 0:
                continue
            if iata not in coords:
                continue

            lat, lon = coords[iata]
            radius = sqrt(infections) * scales_by_day[day]
            popup_text = (
                f"{iata} - Day {day}: {int(infections)} infected arrivals"
            )

            folium.CircleMarker(
                location=(lat, lon),
                radius=radius,
                color=colors_by_day[day],
                fill=True,
                fill_opacity=0.5,
                popup=popup_text
            ).add_to(m)

            point_counter += 1

    logging.info(f"Added {point_counter} circle markers to map for {scen_label}.")
    scenario_maps[scen_label] = (m, scen_suffix)

    log_timing(f"Finished scenario {scen_label} (took {time.time() - t_scn_start:0.2f} s)")

# -----------------------
# Build grouped bar chart (side-by-side bars)
# -----------------------
logging.info("Generating grouped bar chart PNG (matplotlib)...")

# We want: x-axis = Day 1..4, and for each day we show 6 bars:
#   Low (no mask), Medium (no mask), High (no mask),
#   Low (mask),    Medium (mask),    High (mask)
#
# Colors:
#   no mask → blues
#   mask → grays

bar_colors = {
    "Low (no mask)":        "#4f8bff",  # blue medium
    "Medium (no mask)":     "#2f5fcc",  # darker blue
    "High (no mask)":       "#15306e",  # very dark blue
    "Low (mask 50%)":       "#cfcfcf",  # light gray
    "Medium (mask 50%)":    "#9a9a9a",  # mid gray
    "High (mask 50%)":      "#5a5a5a",  # dark gray
}

# Put scenario labels in a stable order that matches how we think about them:
ordered_labels = [
    "Low (no mask)",
    "Medium (no mask)",
    "High (no mask)",
    "Low (mask 50%)",
    "Medium (mask 50%)",
    "High (mask 50%)",
]

# Build a matrix: rows = day 1..4, cols = ordered_labels
days = [1,2,3,4]
data_mat = []
for day in days:
    row_vals = []
    for scen_label in ordered_labels:
        row_vals.append(scenario_summaries[scen_label].loc[day])
    data_mat.append(row_vals)
data_mat = np.array(data_mat)  # shape (4 days, 6 scenarios)

num_days = len(days)
num_scen = len(ordered_labels)

x = np.arange(num_days)  # [0,1,2,3] for Day1..4
group_width = 0.8        # total width of the cluster
bar_width = group_width / num_scen

fig, ax = plt.subplots(figsize=(10,5))

for s_idx, scen_label in enumerate(ordered_labels):
    scen_vals = data_mat[:, s_idx]  # values for this scenario across days
    # offset each scenario within the group
    offsets = x - group_width/2 + (s_idx + 0.5)*bar_width
    ax.bar(
        offsets,
        scen_vals,
        width=bar_width,
        label=scen_label,
        color=bar_colors[scen_label],
        edgecolor="black",
        linewidth=0.5
    )

ax.set_xticks(x)
ax.set_xticklabels([f"Day {d}" for d in days])
ax.set_ylabel("Total infected arrivals (all airports)")
ax.set_title("Daily infected arrivals by scenario (side-by-side)")
ax.legend(fontsize="small", ncols=2)

buf = BytesIO()
plt.tight_layout()
plt.savefig(buf, format="png")
buf.seek(0)
chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
plt.close(fig)

log_timing("Grouped bar chart created")

# -----------------------
# Navigation helpers
# -----------------------
def nav_links(current_suffix=None):
    # current_suffix helps highlight which page you're on
    # We'll show both types: map and stats pages.
    links = []
    for (_, scen_label, scen_suffix, _) in scenarios:
        map_href   = f"infected_airport_map_{scen_suffix}.html"
        stats_href = f"infected_airport_stats_{scen_suffix}.html"
        if scen_suffix == current_suffix:
            links.append(
                f"<strong>{scen_label} [<span style='color:#444;'>map</span> | stats]</strong>"
            )
        else:
            links.append(
                f"<a href='{map_href}'>{scen_label} map</a> | "
                f"<a href='{stats_href}'>stats</a>"
            )
    links.append("<a href='infected_airport_map_summary.html'>Summary</a>")
    links.append("<a href='infected_airport_dashboard.html'>Dashboard</a>")
    return " &nbsp;||&nbsp; ".join(links)

# -----------------------
# Write per-scenario MAP pages (map only)
# -----------------------
logging.info("Writing per-scenario MAP pages (folium only)...")

for scen_label, (m, scen_suffix) in scenario_maps.items():
    out_file = f"infected_airport_map_{scen_suffix}.html"
    logging.info(f"Building {out_file} for {scen_label} ...")

    folium_html = m.get_root().render()
    lower_html = folium_html.lower()
    body_close_idx = lower_html.rfind("</body>")

    legend_block = """
<div style="font-family:Arial; font-size:0.8rem; background:white; border:1px solid #ccc;
            border-radius:6px; padding:0.5rem 0.75rem; max-width:260px;
            box-shadow:0 2px 4px rgba(0,0,0,0.15);">
  <div style="margin-bottom:0.5rem; font-weight:bold;">Legend</div>
  <div>Circle size ∝ sqrt(infected arrivals)</div>
  <div><span style='color:blue;'>●</span> Day 1 &nbsp;
       <span style='color:green;'>●</span> Day 2</div>
  <div><span style='color:orange;'>●</span> Day 3 &nbsp;
       <span style='color:red;'>●</span> Day 4</div>
</div>
"""

    inject_block = f"""
<div style="padding:1em; font-family:Arial;">
  <div style="margin-bottom:1em;">
    Navigation: {nav_links(scen_suffix)}
  </div>
  <h2 style="margin-top:0;">{scen_label}</h2>
  <p style="font-size:0.9em;color:#555; line-height:1.4em;">
    Interactive map. Zoom and pan. Hover/tap points for infected arrivals per airport.
  </p>
  <div style="position:absolute; top:80px; left:20px; z-index:9999;">
    {legend_block}
  </div>
</div>
"""

    if body_close_idx == -1:
        # Can't inject; fallback to raw folium
        final_html = folium_html
    else:
        final_html = (
            folium_html[:body_close_idx]
            + inject_block
            + folium_html[body_close_idx:]
        )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_html)

    logging.info(f"Wrote {out_file}")

# -----------------------
# Write per-scenario STATS pages (chart + table only)
# -----------------------
logging.info("Writing per-scenario STATS pages (chart + table)...")

for (scen_key, scen_label, scen_suffix, is_mask) in scenarios:
    out_file = f"infected_airport_stats_{scen_suffix}.html"
    logging.info(f"Building {out_file} ...")

    perday_text = scenario_summaries[scen_label].to_string()

    stats_html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{scen_label} - Infected Arrivals Stats</title>
  <style>
    body {{
      font-family: Arial;
      margin: 1em;
      color: #222;
      background: #f8f9fa;
    }}
    .card {{
      background:#fff;
      border:1px solid #ccc;
      border-radius:8px;
      box-shadow:0 2px 4px rgba(0,0,0,0.06);
      padding:1rem 1.25rem;
      margin-bottom:1rem;
    }}
    pre {{
      background:#f0f0f0;
      padding:0.5em;
      border-radius:6px;
      font-size:0.9rem;
      line-height:1.3em;
      overflow-x:auto;
    }}
    h2 {{
      margin-top:0;
      font-size:1.2rem;
    }}
    h3 {{
      margin-bottom:0.5rem;
      font-size:1rem;
    }}
    .nav {{
      font-size:0.85rem;
      margin-bottom:1rem;
      line-height:1.4em;
    }}
    img.chart {{
      max-width:700px;
      border:1px solid #ccc;
      border-radius:6px;
      box-shadow:0 2px 4px rgba(0,0,0,0.06);
    }}
  </style>
</head>
<body>

<div class="nav">
  Navigation: {nav_links(scen_suffix)}
</div>

<div class="card">
  <h2>{scen_label} – Summary stats</h2>
  <p style="font-size:0.9rem; color:#555;">
    Daily totals across all airports for this scenario.
  </p>
  <pre>{perday_text}</pre>
</div>

<div class="card">
  <h3>Daily infected arrivals by scenario (side-by-side)</h3>
  <p style="font-size:0.9rem; color:#555;">
    Blue shades = no mask (Low/Medium/High). Gray shades = mask 50% (Low/Medium/High).
  </p>
  <img class="chart" src="data:image/png;base64,{chart_base64}" />
</div>

<div class="card" style="font-size:0.8rem; color:#777;">
  <p>
    Bars show total modeled infected passengers arriving anywhere in the network on each day
    after the initial release. Masked scenarios apply a 50% effective universal masking
    assumption at exposure, which reduces onward infectious travelers.
  </p>
</div>

</body>
</html>
"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(stats_html)

    logging.info(f"Wrote {out_file}")

# -----------------------
# Global summary / index page (links, table, chart)
# -----------------------
summary_file = "infected_airport_map_summary.html"
logging.info(f"Building {summary_file} ...")

summary_table_df = pd.DataFrame({
    scen_label: scenario_summaries[scen_label] for (_, scen_label, _, _) in scenarios
})
summary_table_text = summary_table_df.to_string()

scenario_list_items = "\n".join(
    f'  <li>'
    f'<a href="infected_airport_map_{scen_suffix}.html">{scen_label} map</a> | '
    f'<a href="infected_airport_stats_{scen_suffix}.html">{scen_label} stats</a>'
    f'</li>'
    for (_, scen_label, scen_suffix, _) in scenarios
)

summary_html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Infected Arrivals – Summary</title>
  <style>
    body {{
      font-family: Arial;
      margin: 1em;
      background:#f8f9fa;
      color:#222;
    }}
    pre {{
      background:#f0f0f0;
      padding:0.5em;
      border-radius:6px;
      font-size:0.9rem;
      line-height:1.3em;
      overflow-x:auto;
    }}
    img.chart {{
      max-width:700px;
      border:1px solid #ccc;
      border-radius:6px;
      box-shadow:0 2px 4px rgba(0,0,0,0.06);
    }}
    ul {{
      line-height:1.5em;
    }}
    a {{
      color:#0645AD;
      text-decoration:none;
    }}
    a:hover {{
      text-decoration:underline;
    }}
    .nav {{
      font-size:0.85rem;
      margin-bottom:1rem;
    }}
  </style>
</head>
<body>

<div class="nav">
  Navigation: {nav_links(None)}
</div>

<h1 style="margin-top:0;">Infected Arrivals – Summary</h1>
<p style="font-size:0.9rem; color:#555;">
  Overview across all scenarios (Low/Medium/High) and both mitigation conditions
  (no mask vs 50% mask). Each scenario assumes the same plume event at the seed airports,
  different initial attack rates, and optional universal masking.
</p>

<h2>Daily infected arrivals by scenario</h2>
<img class="chart" src="data:image/png;base64,{chart_base64}" />

<h2>Totals per Day (all scenarios)</h2>
<pre>{summary_table_text}</pre>

<h2>Jump to scenario pages</h2>
<ul>
{scenario_list_items}
</ul>

<p style="font-size:0.8rem; color:#777;">
  "Map" pages are fully interactive Folium/Leaflet maps.<br/>
  "Stats" pages show numeric daily totals and the comparative bar chart.
</p>

</body>
</html>
"""

with open(summary_file, "w", encoding="utf-8") as f:
    f.write(summary_html)

logging.info(f"Wrote {summary_file}")

# -----------------------
# 3x2 dashboard with iframes (all six maps)
# -----------------------
dashboard_file = "infected_airport_dashboard.html"
logging.info(f"Building {dashboard_file} ...")

dashboard_cards = ""
# Top row: first 3 no-mask. Bottom row: 3 mask.
for (scen_key, scen_label, scen_suffix, is_mask) in scenarios:
    dashboard_cards += f"""
  <div class="panel">
    <div class="panel-header">{scen_label}</div>
    <div class="panel-body">
      <iframe src="infected_airport_map_{scen_suffix}.html"></iframe>
    </div>
  </div>
"""

dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Infected Passenger Spread Dashboard</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 1rem;
      background: #f8f9fa;
      color: #222;
    }}

    h1 {{
      font-size: 1.4rem;
      margin-bottom: 0.5rem;
      font-weight: 600;
    }}

    p.desc {{
      font-size: 0.9rem;
      max-width: 900px;
      line-height: 1.4;
      color: #444;
      margin-top: 0;
      margin-bottom: 1.5rem;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      grid-auto-rows: 400px;
      gap: 1rem;
    }}

    .panel {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.06);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    .panel-header {{
      background: #f0f2f5;
      border-bottom: 1px solid #ddd;
      padding: 0.5rem 0.75rem;
      font-size: 0.9rem;
      font-weight: 600;
      line-height: 1.2;
      color: #111;
    }}

    .panel-body {{
      flex: 1;
      position: relative;
    }}

    .panel-body iframe {{
      position: absolute;
      border: 0;
      width: 100%;
      height: 100%;
      left: 0;
      top: 0;
    }}

    @media (max-width: 1200px) {{
      .grid {{
        grid-template-columns: repeat(2, 1fr);
        grid-auto-rows: 400px;
      }}
    }}

    @media (max-width: 700px) {{
      .grid {{
        grid-template-columns: 1fr;
        grid-auto-rows: 400px;
      }}
    }}

    .legend-block {{
      margin-top: 1rem;
      font-size: 0.8rem;
      color: #444;
      line-height: 1.4;
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 0.75rem 1rem;
      max-width: 900px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }}

    .legend-swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 4px;
      vertical-align: middle;
    }}
    .day1 {{ background: blue; }}
    .day2 {{ background: green; }}
    .day3 {{ background: orange; }}
    .day4 {{ background: red; }}
  </style>
</head>
<body>

<h1>Infected Passenger Spread Dashboard</h1>
<p class="desc">
  Top row: no mask. Bottom row: 50% universal masking. Circles show infected arrivals
  at each airport. Larger = more. Color = day of arrival (Day 1 blue → Day 4 red).
</p>

<div class="grid">
{dashboard_cards}
</div>

<div class="legend-block">
  <div>
    <span class="legend-swatch day1"></span>Day 1
    &nbsp;&nbsp;
    <span class="legend-swatch day2"></span>Day 2
    &nbsp;&nbsp;
    <span class="legend-swatch day3"></span>Day 3
    &nbsp;&nbsp;
    <span class="legend-swatch day4"></span>Day 4
  </div>
  <div>
    Masked scenarios assume 50% effective universal masking at exposure,
    which cuts infected traveler export roughly in half.
  </div>
</div>

</body>
</html>
"""

with open(dashboard_file, "w", encoding="utf-8") as f:
    f.write(dashboard_html)

logging.info(f"Wrote {dashboard_file}")

log_timing("All pages written.")
logging.info("=== Script finished successfully ===")
