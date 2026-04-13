import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from data_manipulation import get_season_totals, create_metrics, add_seed_and_region, add_FF, add_team_names

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='March Madness Predictor',
    page_icon='🏀',
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def load_data():

    rdsr = pd.read_csv("./data/MRegularSeasonDetailedResults.csv")
    teams = pd.read_csv("./data/MTeams.csv")
    tourney = pd.read_csv("./data/MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv("./data/MNCAATourneySeeds.csv")

    return rdsr, teams, tourney, seeds

@st.cache_data
def manipulate_data():
    season_stats = get_season_totals(rdsr)
    team_stats = create_metrics(season_stats)
    team_stats = add_seed_and_region(team_stats, seeds)
    stats_data = add_FF(team_stats, tourney)
    stats_data_with_names = add_team_names(stats_data, teams)
    return stats_data, stats_data_with_names


@st.cache_resource
def load_models():
    pca_model = joblib.load("./data/pca_pipeline.pkl")
    no_pca_model = joblib.load("./data/non_pca_pipeline.pkl")
    columns = joblib.load("./data/feature_columns.pkl")
    return pca_model, no_pca_model, columns


rdsr, teams, tourney, seeds = load_data()
stats_data, stats_data_with_names = manipulate_data()
pca_model, no_pca_model, columns = load_models()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# 🏀 March Madness Final Four Predictor

This app demonstrates a machine learning model trained to predict Final Four teams using historical NCAA data.
'''

'''
Choose any season from 2003 to 2025 to explore the data and predictions!
'''
# Season Selection
seasons = sorted(stats_data["Season"].unique())
selected_season = st.selectbox("Select Season", seasons)

# Filter by season FIRST
season_df = stats_data_with_names[
    stats_data_with_names["Season"] == selected_season
].copy()

st.subheader(f"Data for {selected_season}")
stats_df = season_df[["Team", "Region"] + columns]
st.dataframe(stats_df)

'''
Choose what statistics to compare between the average of the Final Four teams and any other teams. You can select as many as you like! 
'''

# Map FinalFour
season_df['FinalFour'] = season_df['FinalFour'].map({
    0: 'Not Final Four',
    1: 'Final Four'
})

# User selects stats to compare
selected_stats = st.multiselect(
    "Select stats to compare",
    columns,
    default=columns[:5]
)

'''
Now, choose a team to compare against the average Final Four team. 
The radar chart will show how that team ranks in the selected stats compared to 
the average of the Final Four teams from that season. Hover over each point to 
see the exact percentile and raw value for that stat!
'''

# select a team for the radar chart
selected_team = st.selectbox(
    "Select a team",
    season_df["Team"].unique()
)

percentile_df = season_df.copy()

for col in selected_stats:
    min_val = season_df[col].min()
    max_val = season_df[col].max()
    
    percentile_df[col + "_pct"] = (
        (season_df[col] - min_val) / (max_val - min_val)
    ) * 100

lower_is_better = ['DefRtg', 'OPPG', 'TORate']

for col in lower_is_better:
    if col in selected_stats:
        percentile_df[col + "_pct"] = 100 - percentile_df[col + "_pct"]

team_row = percentile_df[percentile_df["Team"] == selected_team].iloc[0]

categories = selected_stats

team_percentiles = [team_row[col + "_pct"] for col in selected_stats]
team_raw = [team_row[col] for col in selected_stats]

# Average Final Four team
ff_df = percentile_df[percentile_df["FinalFour"] == "Final Four"]

avg_percentiles = [
    ff_df[col + "_pct"].mean() for col in selected_stats
]

# Used to connect the lines in the radar chart around whatever stats the user selects. 
# We repeat the first stat at the end to close the loop.
categories = selected_stats + [selected_stats[0]]
team_percentiles += [team_percentiles[0]]
avg_percentiles += [avg_percentiles[0]]

fig = go.Figure()

# Team trace
fig.add_trace(go.Scatterpolar(
    r=team_percentiles,
    theta=categories,
    fill='toself',
    name=selected_team,
    hovertext=[
        f"{cat}<br>Percentile: {pct:.1f}<br>Value: {raw:.3f}"
        for cat, pct, raw in zip(categories, team_percentiles, team_raw)
    ],
    hoverinfo="text"
))

# Final Four average trace
fig.add_trace(go.Scatterpolar(
    r=avg_percentiles,
    theta=categories,
    fill='toself',
    name='Avg Final Four',
    line=dict(dash='dash')
))

fig.update_layout(
    title=f"{selected_team} Profile vs Final Four Teams ({selected_season})",
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )
    ),
    showlegend=True
)

st.plotly_chart(fig, width='stretch')

'''
Next, choose which model you want to use to predict the probabilities of each team making the Final Four.
You can view your selection in a table below or in the predictions plot at the bottom of the page!
'''

# Model Selection
model_choice = st.selectbox(
    "Choose Model",
    ["Non-PCA", "PCA", "Compare Both"]
)

features = season_df.drop(columns=["FinalFour", "Season", "Team", "Region"])

if model_choice == "Non-PCA":
    season_df["Probability"] = no_pca_model.predict_proba(features)[:, 1]

elif model_choice == "PCA":
    season_df["Probability"] = pca_model.predict_proba(features)[:, 1]

else:
    season_df["Non-PCA Prob"] = no_pca_model.predict_proba(features)[:, 1]
    season_df["PCA Prob"] = pca_model.predict_proba(features)[:, 1]

st.subheader("Predictions")

if model_choice == "Compare Both":
    season_df = season_df.sort_values(by="Non-PCA Prob", ascending=True)
    x = "Non-PCA Prob"
    st.dataframe(
        season_df[["Season", "Team", "Seed", "Region", "FinalFour", "Non-PCA Prob", "PCA Prob"]].sort_values("Non-PCA Prob", ascending=False)
    )
else:
    x = "Probability"
    season_df = season_df.sort_values(by="Probability", ascending=True)
    st.dataframe(
        season_df[["Season", "Team", "Seed", "Region", "FinalFour", "Probability"]].sort_values("Probability", ascending=False)
    )

st.subheader("Final Four Predictions Plot")

'''
Now, you can choose to filter the predictions plot by region. This will show you how the model ranks teams from different regions against each other.
'''

regions = ["All"] + sorted(season_df["Region"].unique())

selected_regions = st.multiselect(
    "Select Region(s)",
    options=sorted(season_df["Region"].unique()),
    default=sorted(season_df["Region"].unique())
)

filtered_df = season_df[season_df["Region"].isin(selected_regions)]

color_map = {
    "W": "blue",
    "X": "red",
    "Y": "green",
    "Z": "orange"
}

if model_choice == "Compare Both":
    
    # Reshape data
    df_long = filtered_df.melt(
        id_vars=["Team", "Region", "Seed", "FinalFour"],
        value_vars=["Non-PCA Prob", "PCA Prob"],
        var_name="Model",
        value_name="Probability"
    )

    # Clean model names
    df_long["Model"] = df_long["Model"].map({
        "Non-PCA Prob": "Non-PCA Model",
        "PCA Prob": "PCA Model"
    })

    # Sort
    df_sorted = df_long.sort_values("Probability", ascending=True)

    pred_fig = px.scatter(
        df_sorted,
        x="Probability",
        y="Team",
        color="Region",
        symbol="Model",
        hover_data=["Seed", "FinalFour", "Model"],
        category_orders={"Team": df_sorted["Team"].unique()},
        color_discrete_map=color_map
    )

    height = max(1, 25 * df_sorted["Team"].nunique())

else:
    # Single model behavior (same as before)
    df_sorted = filtered_df.sort_values("Probability", ascending=True)

    pred_fig = px.scatter(
        df_sorted,
        x="Probability",
        y="Team",
        color="Region",
        hover_data=["Seed", "FinalFour"],
        category_orders={"Team": df_sorted["Team"].tolist()},
        color_discrete_map=color_map
    )

    height = max(1, 25 * len(df_sorted))

container_height = min(height, 600)

st.container(height=container_height).plotly_chart(pred_fig, width='stretch', height=height)