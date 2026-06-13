"""
app_airbnb.py
=============
Airbnb Listings Explorer — a self-contained Streamlit application.

What this app does
------------------
1.  Generates ~2,000 synthetic Airbnb listings scattered across real New York
    City neighbourhoods (realistic latitude/longitude clusters).
2.  Cleans the data with several clearly commented steps.
3.  Provides interactive exploratory data analysis:
      * a Plotly ``scatter_mapbox`` map (OpenStreetMap tiles — no token needed),
      * a price-distribution histogram,
      * price boxplots by room type and by neighbourhood,
      * an availability-vs-reviews scatter.
4.  Shows a live summary-statistics table.
5.  Offers rich sidebar filters: neighbourhood, room type and price range.

Run it with:
    streamlit run app_airbnb.py

Only the following libraries are used: streamlit, pandas, numpy, plotly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------- #
# Page configuration                                                          #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Airbnb Listings Explorer",
    page_icon="🏙️",
    layout="wide",
)

# Real NYC neighbourhood centroids (approx lat/lon) with a baseline price level
# and a sensible point spread (degrees) so listings cluster realistically.
NYC_NEIGHBOURHOODS = {
    "Manhattan - Midtown":      {"lat": 40.7549, "lon": -73.9840, "base": 250, "spread": 0.012},
    "Manhattan - Harlem":       {"lat": 40.8116, "lon": -73.9465, "base": 140, "spread": 0.015},
    "Brooklyn - Williamsburg":  {"lat": 40.7081, "lon": -73.9571, "base": 180, "spread": 0.013},
    "Brooklyn - Bushwick":      {"lat": 40.6944, "lon": -73.9213, "base": 110, "spread": 0.016},
    "Queens - Astoria":         {"lat": 40.7644, "lon": -73.9235, "base": 120, "spread": 0.015},
    "Queens - Flushing":        {"lat": 40.7675, "lon": -73.8331, "base": 95,  "spread": 0.018},
    "Bronx - Riverdale":        {"lat": 40.8907, "lon": -73.9120, "base": 90,  "spread": 0.020},
    "Staten Island - St. George": {"lat": 40.6437, "lon": -74.0765, "base": 85, "spread": 0.020},
}

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
# Multiplicative price effect of room type relative to the neighbourhood base.
ROOM_TYPE_PRICE_FACTOR = {
    "Entire home/apt": 1.0,
    "Private room": 0.6,
    "Shared room": 0.4,
    "Hotel room": 1.2,
}


# --------------------------------------------------------------------------- #
# 1. Synthetic data generation                                                #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Generating synthetic Airbnb listings…")
def generate_listings(n_listings: int = 2000, seed: int = 99) -> pd.DataFrame:
    """Create a synthetic NYC Airbnb listings table.

    Coordinates are sampled around each neighbourhood centroid; prices combine
    the neighbourhood baseline, a room-type multiplier and log-normal noise so
    the price distribution is realistically right-skewed.
    """
    rng = np.random.default_rng(seed)
    neighbourhood_names = list(NYC_NEIGHBOURHOODS.keys())
    rows = []

    for i in range(n_listings):
        nb = rng.choice(neighbourhood_names)
        cfg = NYC_NEIGHBOURHOODS[nb]
        room = rng.choice(ROOM_TYPES, p=[0.52, 0.38, 0.06, 0.04])

        # Scatter coordinates around the neighbourhood centre.
        lat = cfg["lat"] + rng.normal(0, cfg["spread"])
        lon = cfg["lon"] + rng.normal(0, cfg["spread"])

        # Price = base * room factor * log-normal noise.
        price = cfg["base"] * ROOM_TYPE_PRICE_FACTOR[room] * rng.lognormal(0, 0.35)

        rows.append(
            {
                "listing_id": i,
                "neighbourhood": nb,
                "borough": nb.split(" - ")[0],
                "room_type": room,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "price": round(price, 2),
                "minimum_nights": int(rng.choice([1, 2, 3, 5, 7, 30], p=[0.45, 0.2, 0.15, 0.1, 0.05, 0.05])),
                "number_of_reviews": int(rng.poisson(18)),
                "reviews_per_month": round(float(rng.gamma(1.5, 0.8)), 2),
                "availability_365": int(rng.integers(0, 366)),
                "host_listings_count": int(rng.choice([1, 1, 1, 2, 3, 8], )),
            }
        )

    data = pd.DataFrame(rows)

    # --- Inject "dirtiness" so cleaning is meaningful -----------------------
    # (a) Duplicate some listings.
    data = pd.concat([data, data.sample(50, random_state=seed)], ignore_index=True)
    # (b) Missing values in reviews_per_month (common in the real dataset).
    miss_idx = rng.choice(len(data), size=int(0.08 * len(data)), replace=False)
    data.loc[miss_idx, "reviews_per_month"] = np.nan
    # (c) A few zero / absurdly high prices (listing errors).
    bad_idx = rng.choice(len(data), size=25, replace=False)
    data.loc[bad_idx, "price"] = rng.choice([0.0, 99999.0], size=25)

    return data


# --------------------------------------------------------------------------- #
# 2. Data cleaning                                                            #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Cleaning listings…")
def clean_listings(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the listings table with transparent, commented steps.

    Cleaning steps:
      1. Drop duplicate listings by ``listing_id``.
      2. Remove invalid prices (<= 0) and cap extreme prices at the 99th pct.
      3. Impute missing ``reviews_per_month`` with 0 (no reviews logged).
      4. Add a derived ``price_per_night_bucket`` for easier segmentation.
    """
    df = raw.copy()

    # --- Cleaning step 1: drop duplicate listings ---------------------------
    df = df.drop_duplicates(subset="listing_id").reset_index(drop=True)

    # --- Cleaning step 2: fix invalid / extreme prices ----------------------
    # Prices must be positive; outrageous outliers are winsorised at p99.
    df = df[df["price"] > 0].reset_index(drop=True)
    price_cap = df["price"].quantile(0.99)
    df["price"] = df["price"].clip(upper=price_cap)

    # --- Cleaning step 3: impute missing reviews_per_month ------------------
    # A missing value here means the listing has simply never been reviewed.
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)

    # --- Cleaning step 4: engineer a price bucket for segmentation ----------
    df["price_bucket"] = pd.cut(
        df["price"],
        bins=[0, 75, 150, 300, np.inf],
        labels=["$ (<75)", "$$ (75-150)", "$$$ (150-300)", "$$$$ (300+)"],
    )

    return df


# --------------------------------------------------------------------------- #
# 3. Plotly helpers                                                           #
# --------------------------------------------------------------------------- #
def plot_map(df: pd.DataFrame, max_points: int = 1500) -> px.scatter_mapbox:
    """Scatter map of listings on OpenStreetMap tiles (no Mapbox token needed).

    For responsiveness we sample down to ``max_points`` markers when the
    filtered set is large.
    """
    plot_df = df if len(df) <= max_points else df.sample(max_points, random_state=0)
    fig = px.scatter_mapbox(
        plot_df,
        lat="latitude",
        lon="longitude",
        color="price",
        size="price",
        size_max=12,
        zoom=9.5,
        color_continuous_scale="Turbo",
        hover_name="neighbourhood",
        hover_data={"room_type": True, "price": ":.0f", "latitude": False, "longitude": False},
    )
    # OpenStreetMap style requires no access token.
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        title="Listing Locations (colour & size = price)",
    )
    return fig


def plot_map_static(df: pd.DataFrame, max_points: int = 1500) -> px.scatter:
    """Non-WebGL fallback map: a plain longitude/latitude scatter plot.

    ``scatter_mapbox`` relies on WebGL (mapbox-gl). In rare environments where
    WebGL is unavailable (some headless/remote browsers) the tiled map renders
    blank. This SVG-based scatter always renders and still conveys the spatial
    distribution of listings, coloured by price.
    """
    plot_df = df if len(df) <= max_points else df.sample(max_points, random_state=0)
    fig = px.scatter(
        plot_df,
        x="longitude",
        y="latitude",
        color="price",
        color_continuous_scale="Turbo",
        hover_name="neighbourhood",
        hover_data={"room_type": True, "price": ":.0f"},
        labels={"longitude": "Longitude", "latitude": "Latitude"},
        # Force SVG rendering: Plotly otherwise auto-switches to WebGL
        # ("scattergl") above ~1,000 points, which defeats the fallback.
        render_mode="svg",
    )
    fig.update_layout(
        title="Listing Locations (static fallback · colour = price)",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    # Keep the geographic aspect ratio roughly correct for NYC's latitude.
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
    return fig


def plot_price_histogram(df: pd.DataFrame) -> px.histogram:
    """Histogram of nightly price."""
    fig = px.histogram(
        df, x="price", nbins=50, color_discrete_sequence=["#ef4444"],
        labels={"price": "Price per night ($)"},
    )
    fig.update_layout(title="Price Distribution", yaxis_title="Number of listings")
    return fig


def plot_price_by_room(df: pd.DataFrame) -> px.box:
    """Boxplot of price by room type."""
    fig = px.box(
        df, x="room_type", y="price", color="room_type",
        labels={"room_type": "Room type", "price": "Price ($)"},
    )
    fig.update_layout(title="Price by Room Type", showlegend=False)
    return fig


def plot_price_by_neighbourhood(df: pd.DataFrame) -> px.box:
    """Boxplot of price by neighbourhood (sorted by median price)."""
    order = (
        df.groupby("neighbourhood")["price"].median().sort_values().index.tolist()
    )
    fig = px.box(
        df, x="price", y="neighbourhood", color="borough",
        category_orders={"neighbourhood": order},
        labels={"price": "Price ($)", "neighbourhood": ""},
    )
    fig.update_layout(title="Price by Neighbourhood", height=450)
    return fig


def plot_reviews_vs_availability(df: pd.DataFrame) -> px.scatter:
    """Scatter of availability against review volume, coloured by room type."""
    fig = px.scatter(
        df, x="availability_365", y="number_of_reviews",
        color="room_type", opacity=0.5,
        labels={
            "availability_365": "Days available / year",
            "number_of_reviews": "Number of reviews",
        },
    )
    fig.update_layout(title="Availability vs. Review Volume")
    return fig


# --------------------------------------------------------------------------- #
# 4. Sidebar controls                                                         #
# --------------------------------------------------------------------------- #
def build_sidebar(df: pd.DataFrame) -> dict:
    """Render the sidebar filters and return the chosen values."""
    st.sidebar.header("🔧 Filters")

    neighbourhoods = st.sidebar.multiselect(
        "Neighbourhood",
        options=sorted(df["neighbourhood"].unique()),
        default=sorted(df["neighbourhood"].unique()),
    )

    room_types = st.sidebar.multiselect(
        "Room type",
        options=ROOM_TYPES,
        default=ROOM_TYPES,
    )

    price_min = float(df["price"].min())
    price_max = float(df["price"].max())
    price_range = st.sidebar.slider(
        "Price range ($ / night)",
        min_value=round(price_min),
        max_value=round(price_max),
        value=(round(price_min), round(price_max)),
    )

    st.sidebar.markdown("---")
    max_nights = st.sidebar.slider("Maximum minimum-nights", 1, 30, 30)

    # Map rendering mode. The interactive OpenStreetMap view needs WebGL; the
    # static scatter is a universal fallback for environments that lack it.
    map_mode = st.sidebar.radio(
        "Map rendering",
        options=["Interactive (OpenStreetMap)", "Static scatter (no WebGL)"],
        help="Switch to the static scatter if the interactive map appears blank.",
    )

    return {
        "neighbourhoods": neighbourhoods,
        "room_types": room_types,
        "price_range": price_range,
        "max_nights": max_nights,
        "map_mode": map_mode,
    }


def apply_filters(df: pd.DataFrame, controls: dict) -> pd.DataFrame:
    """Apply all sidebar filters to the cleaned listings."""
    out = df.copy()
    if controls["neighbourhoods"]:
        out = out[out["neighbourhood"].isin(controls["neighbourhoods"])]
    if controls["room_types"]:
        out = out[out["room_type"].isin(controls["room_types"])]

    low, high = controls["price_range"]
    out = out[(out["price"] >= low) & (out["price"] <= high)]
    out = out[out["minimum_nights"] <= controls["max_nights"]]
    return out


# --------------------------------------------------------------------------- #
# 5. Main application body                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    """Assemble the listings-explorer page."""
    st.title("🏙️ NYC Airbnb Listings Explorer")
    st.markdown(
        "Explore ~2,000 synthetic Airbnb listings across New York City. Use the "
        "sidebar to filter by neighbourhood, room type and price."
    )

    raw = generate_listings()
    clean = clean_listings(raw)
    controls = build_sidebar(clean)
    filtered = apply_filters(clean, controls)

    if filtered.empty:
        st.warning("No listings match the current filters. Widen your selection.")
        st.stop()

    # --- Headline metrics ----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listings shown", f"{len(filtered):,}")
    c2.metric("Median price", f"${filtered['price'].median():,.0f}")
    c3.metric("Avg reviews", f"{filtered['number_of_reviews'].mean():.0f}")
    c4.metric("Avg availability", f"{filtered['availability_365'].mean():.0f} d/yr")

    st.markdown("---")

    # --- Map -----------------------------------------------------------------
    st.subheader("🗺️ Listing Map")
    if controls["map_mode"].startswith("Interactive"):
        st.plotly_chart(plot_map(filtered), use_container_width=True)
        st.caption(
            "Tip: if the map looks blank, your browser may lack WebGL — switch "
            "to *Static scatter (no WebGL)* in the sidebar."
        )
    else:
        st.plotly_chart(plot_map_static(filtered), use_container_width=True)

    # --- Distribution charts -------------------------------------------------
    st.subheader("📊 Price & Demand Analysis")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_price_histogram(filtered), use_container_width=True)
    with right:
        st.plotly_chart(plot_price_by_room(filtered), use_container_width=True)

    st.plotly_chart(plot_price_by_neighbourhood(filtered), use_container_width=True)
    st.plotly_chart(plot_reviews_vs_availability(filtered), use_container_width=True)

    st.markdown("---")

    # --- Summary statistics --------------------------------------------------
    st.subheader("📋 Summary Statistics")
    summary = (
        filtered.groupby("room_type")
        .agg(
            listings=("listing_id", "count"),
            median_price=("price", "median"),
            mean_price=("price", "mean"),
            avg_reviews=("number_of_reviews", "mean"),
            avg_availability=("availability_365", "mean"),
        )
        .round(1)
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)

    with st.expander("🔎 Browse the filtered listings"):
        st.dataframe(
            filtered[
                [
                    "neighbourhood", "room_type", "price", "minimum_nights",
                    "number_of_reviews", "availability_365", "price_bucket",
                ]
            ].head(200),
            use_container_width=True,
        )
        st.caption(f"Showing up to 200 of {len(filtered):,} filtered listings.")


if __name__ == "__main__":
    main()
