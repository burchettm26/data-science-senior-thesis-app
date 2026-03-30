import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='March Madness Predictor',
    page_icon='🏀', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():

    rdsr = pd.read_csv("./data/MRegularSeasonDetailedResults.csv")
    teams = pd.read_csv("./data/MTeams.csv")
    tourney = pd.read_csv("./data/MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv("./data/MNCAATourneySeeds.csv")

    return rdsr, teams, tourney, seeds


rdsr, teams, tourney, seeds = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# 🏀 March Madness Final Four Predictor

This app demonstrates a machine learning model trained to predict Final Four teams using historical NCAA data.
'''

# Season Selection
seasons = sorted(rdsr["Season"].unique())
selected_season = st.selectbox("Select Season", seasons)

# Region Selection

# Model Selection
model_choice = st.selectbox(
    "Choose Model",
    ["Non-PCA", "PCA", "Compare Both"]
)

season_df = rdsr[rdsr["Season"] == selected_season].copy()

# Show the data for a season
st.subheader(f"Data for {selected_season}")
st.dataframe(season_df)