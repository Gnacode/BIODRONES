# Danish Airports Biological Warfare Spread Simulation

## **Overview**

This project models **aerosol dispersion and subsequent infection spread through air travel networks** using a combination of Lagrangian dispersion modeling (OpenDrift) and network-based infection propagation.
 It combines geospatial simulation, probabilistic infection modeling, and visualization of results for three major Danish airports (**AAL, BLL, and CPH**) over the period **24–29 September 2025**.

------

## **1. Dispersion Modeling with OpenDrift**

### **Purpose**

To estimate the initial **exposure area and intensity** of an aerosol release event near major airports in Denmark.

### **Model**

- **Tool:** [OpenDrift](https://opendrift.github.io/) using the `Leeway` model.
- **Meteorological Input:** ERA5 10 m wind fields (NetCDF CF-compliant).
- **Approach:** *Lagrangian particle tracking* — particles represent airborne pathogen carriers transported by wind.
- **Vertical Mixing Proxy:** Wind speed scaled by a factor
   [ U_{\text{boost}} = U_{10} \times (1 + \alpha |U_{10}|) ]
   with ( \alpha = 0.3 ), to mimic lofting into faster shear layers.

### **Simulation Parameters**

| Parameter             | Value         | Description                                 |
| --------------------- | ------------- | ------------------------------------------- |
| Particles per site    | 2000          | Each seeded near an airport                 |
| Time step             | 30 min        | Integration step                            |
| Duration              | 5 days        | Simulation span                             |
| Turbulent diffusivity | 25 m²/s       | Horizontal spreading                        |
| Mixing factor         | 0.3           | Wind speed–dependent vertical shear scaling |
| Seeds                 | AAL, BLL, CPH | Optional offshore duplicates                |

### **Outputs**

- **NetCDF:** `opendrift_air_aerosol_tracks.nc`
- **GeoJSON:** particle trajectories (`_points.geojson` & `_tails.geojson`)
- **Per-frame GeoJSONs:** `frames/frame_####_points.geojson` (for animation)
- **CSV:** particle coordinates over time

### **Rendering**

`DK-rend-4e.py` converts frames into **satellite basemap PNGs** using Esri World Imagery and `contextily`.
 It color-codes particles by source (AAL/BLL/CPH) and produces visual sequences of the aerosol plume dispersion across Denmark.


![Figure 1](https://github.com/Gnacode/EURSUR/blob/dee528019603d576365d743dc69f021a40ed305c/Figure%201.svg?raw=true)

------

## **2. Infection Network Modeling (4-Hop Propagation)**

### **Purpose**

To simulate how initially exposed passengers may propagate infections through the **European air route network**.

### **Tool**

`AirportHop-2.py` builds a **4-hop air-travel infection model**.

### **Inputs**

- `airports.dat` (OpenFlights format)
- `routes.dat` (OpenFlights format)
- Parameters for:
  - Passenger volumes per seed airport
  - Exposure fraction
  - Infection attack rates

### **Model Description**

For each seed airport, infected passengers are redistributed over the air network via route transitions up to 4 days (hops).
 Each day represents one leg of the network spread.

| Variable            | Meaning                                                  |
| ------------------- | -------------------------------------------------------- |
| DAILY_PAX           | Avg. daily passengers per airport (e.g., 80,000 for CPH) |
| EXPOSED_FRACTION    | Fraction of pax physically in plume (10%)                |
| SCENARIOS           | Attack rates of 2.5%, 5%, and 10% (Low, Medium, High)    |
| MASK_EFFECTIVENESS  | 50% reduction for masked population                      |
| MAX_HOPS            | 4 days                                                   |
| MAX_DEST_PER_ORIGIN | 4 destinations per airport to reduce network complexity  |

### **Outputs**

For each scenario (A/B/C and masked equivalents):

- `expected_infected_passengers_H4_scnX.csv`
- `cumulative_infection_risk_H4_scnX.csv`

Each matrix encodes the probability or expected number of infected passengers traveling from origin → destination within four hops.

------

## **3. Theoretical Background — Cox Proportional Hazards Model**

### **Purpose of the Cox Model**

The **Cox Proportional Hazards Model** provides a semi-parametric framework to describe how the probability (or *hazard*) of infection changes over time and with covariates such as mask use, airport origin, or infection rate.

The **hazard function** ( h(t) ) represents the instantaneous risk of infection at time ( t ):
 
$$
h(t) = h_0(t) \cdot \exp(\beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p)
$$

 where:

- ( h_0(t) ) is the baseline hazard (infection rate with no covariates),
- ( x_i ) are covariates (e.g., mask usage, infection seed rate),
- ( \beta_i ) are coefficients describing the relative risk associated with each covariate.

### **Partial Likelihood Function**

The Cox model is estimated via **partial likelihood** rather than full parametric likelihood, focusing on the relative ordering of infection events. The likelihood is:
 
$$
L(\beta) = \prod_{i=1}^{k} \frac{\exp(\beta^T x_i)}{\sum_{j \in R_i} \exp(\beta^T x_j)}
$$

 where:

- ( k ) = number of infection events,
- ( R_i ) = set of individuals still at risk just before time ( t_i ),
- ( \beta ) = vector of hazard coefficients.

This function provides relative risk comparisons without assuming a specific baseline hazard.

### **Artificial Parameterization in Our Model**

In our artificial model, we do **not** estimate ( \beta ) from data. Instead, we assign **synthetic coefficients** to emulate different infection severity scenarios:

| Scenario | Description      | Assigned ( \beta )  | Equivalent Attack Rate |
| -------- | ---------------- | ------------------- | ---------------------- |
| A        | Low infection    | ( \beta_A = 0.025 ) | 2.5%                   |
| B        | Medium infection | ( \beta_B = 0.05 )  | 5%                     |
| C        | High infection   | ( \beta_C = 0.10 )  | 10%                    |

These ( \beta ) values scale the baseline hazard through the exponential term ( \exp(\beta) ), defining three proportional hazard levels across the network. Masked scenarios apply a **multiplicative attenuation** of 0.5, reducing the effective hazard by 50%:
[
$$
\beta_{mask} = 0.5 \times \beta
$$



### **Interpretation**

- The **seed infections** at AAL, BLL, and CPH define the initial conditions — airports with non-zero hazard at ( t = 0 ).
- The infection risk propagates through the air-route network as passengers move (4 hops, 4 days).
- Each hop represents a new at-risk set ( R_i ), similar to exposure risk groups in a Cox process.

Thus, while our model is not a fully estimated Cox regression, it conceptually mirrors one by **defining relative hazards via synthetic ( \beta )** values and evolving these hazards through network transitions.

### **Illustrative Equation Diagram**

Below is a conceptual illustration of how the proportional hazards relate to infection intensity across scenarios:

```
            Infection Hazard (h)
                   ↑
                   │                 exp(β)
                   │      ┌──────────────┐
                   │      │              │
                   │      │ Scenario C   │  (β = 0.10)
                   │      │              │  High hazard
                   │      └──────────────┘
                   │
                   │      ┌──────────────┐
                   │      │ Scenario B   │  (β = 0.05)
                   │      │              │  Medium hazard
                   │      └──────────────┘
                   │
                   │      ┌──────────────┐
                   │      │ Scenario A   │  (β = 0.025)
                   │      │              │  Low hazard
                   │      └──────────────┘
                   │
                   └────────────────────────────────────────→  Time (t)
                               exp(βx) scales h₀(t)
```

This schematic illustrates that each scenario simply shifts the baseline hazard function vertically by a factor of ( e^{\beta} ), maintaining proportional hazards over time.

------

## **4. Infection Propagation and Visualization**

### **Purpose**

To estimate daily infection spread and visualize geographic infection clusters.

### **Tool**

`infected_airport_map_folium_6.py` compiles CSV outputs into daily infection risk estimates and visual HTML dashboards.

### **Workflow**

1. Load `airports.dat` for coordinates.

2. Load infection risk (`cumulative_infection_risk_H4_*.csv`) and passenger flow data.

3. Apply distance-based weighting using **great-circle travel times**.

4. Compute infections per day (Day 1–4) with an empirical **infectivity profile**:

   ```python
   {1: 0.05, 2: 0.25, 3: 0.60, 4: 0.85}
   ```

5. Visualize results using:

   - **Folium maps** (colored circles sized by infected arrivals)
   - **Grouped bar charts** comparing masked/unmasked scenarios
   - **HTML dashboards** (auto-linked across all scenarios)

### **Output Structure**

Each scenario generates:

- `infected_airport_map_<scenario>.html` — Interactive map
- `infected_airport_stats_<scenario>.html` — Summary tables & charts
- `infected_airport_map_summary.html` — Overview page with totals
- `infected_airport_dashboard.html` — Combined navigation dashboard

### **Color and Layout Scheme**

| Category       | Color                       | Description          |
| -------------- | --------------------------- | -------------------- |
| **No mask**    | Shades of blue              | Low, Medium, High    |
| **Mask (50%)** | Shades of gray              | Low, Medium, High    |
| **Days 1–4**   | Blue → Green → Orange → Red | Temporal progression |

------

## **5. Assumptions and Limitations**

- The infection modeling is **not epidemiologically calibrated**; attack rates (2.5–10%) are arbitrary for scenario illustration.
- The Cox model framework is used **conceptually**, not as a fitted survival model.
- Mask/no-mask distinctions are represented as multiplicative hazard reductions.
- No secondary infections are modeled—only redistribution of initially infected travelers.
- The OpenDrift dispersion field provides **relative exposure weighting**, not quantitative concentration estimates.
- ERA5 and OpenFlights datasets are used for **illustrative educational modeling**.

------

## **6. Directory Structure**

```
.
├── DK-airports-4n.py         # OpenDrift + ERA5 dispersion model
├── DK-rend-4e.py             # Satellite basemap renderer
├── AirportHop-2.py           # 4-hop infection network model
├── infected_airport_map_folium_6.py  # Visualization and dashboard builder
├── airports.dat              # OpenFlights airports database
├── routes.dat                # OpenFlights routes database
├── results/
│   ├── cumulative_infection_risk_H4_scnA.csv
│   ├── expected_infected_passengers_H4_scnA.csv
│   └── ...
└── outputs/
    ├── infected_airport_map_low.html
    ├── infected_airport_stats_low.html
    ├── infected_airport_map_summary.html
    └── frames_png/
```

------

## **7. Reproducibility**

**Requirements:**

```bash
pip install numpy pandas folium matplotlib geopy opendrift xarray contextily
```

**Run sequence:**

```bash
python DK-airports-4n.py          # Generate dispersion frames
python DK-rend-4e.py              # Render map PNGs
python AirportHop-2.py            # Build infection matrices
python infected_airport_map_folium_6.py  # Build visualizations
```

------

## **8. Example Interpretation**

- **Scenario A (2.5%)** — Minimal spread beyond immediate hubs.
- **Scenario B (5%)** — Noticeable continental spread within 3 days.
- **Scenario C (10%)** — Nearly pan-European dispersion by Day 4.
- **Masking (50%)** reduces total infected arrivals by approximately half across all timepoints.

------
