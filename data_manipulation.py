import pandas as pd

def get_season_totals(game_data):
    winners = pd.DataFrame({
        "Season": game_data.Season,
        "TeamID": game_data.WTeamID,
        "Score": game_data.WScore,
        "OppScore": game_data.LScore,
        "FGM": game_data.WFGM,
        "FGA": game_data.WFGA,
        "FGM3": game_data.WFGM3,
        "FGA3": game_data.WFGA3,
        "FTM": game_data.WFTM,
        "FTA": game_data.WFTA,
        "OR": game_data.WOR,
        "DR": game_data.WDR,
        "Ast": game_data.WAst,
        "TO": game_data.WTO,
        "Stl": game_data.WStl,
        "Blk": game_data.WBlk,
        "PF": game_data.WPF
    })

    losers = pd.DataFrame({
        "Season": game_data.Season,
        "TeamID": game_data.LTeamID,
        "Score": game_data.LScore,
        "OppScore": game_data.WScore,
        "FGM": game_data.LFGM,
        "FGA": game_data.LFGA,
        "FGM3": game_data.LFGM3,
        "FGA3": game_data.LFGA3,
        "FTM": game_data.LFTM,
        "FTA": game_data.LFTA,
        "OR": game_data.LOR,
        "DR": game_data.LDR,
        "Ast": game_data.LAst,
        "TO": game_data.LTO,
        "Stl": game_data.LStl,
        "Blk": game_data.LBlk,
        "PF": game_data.LPF
    })

    games = pd.concat([winners, losers])

    games["Poss"] = games["FGA"] - games["OR"] + games["TO"] + 0.475 * games["FTA"]

    season_stats = games.groupby(["Season","TeamID"]).sum().reset_index()

    games_played = games.groupby(["Season","TeamID"]).size().reset_index(name="Games")

    season_stats = season_stats.merge(games_played, on=["Season","TeamID"])

    return season_stats

def create_metrics(season_stats):
    # FG percentage, 3-point FG percentage, 3-point FG attempt rate
    season_stats["FGPct"] = season_stats.FGM / season_stats.FGA
    season_stats["FG3Pct"] = season_stats.FGM3 / season_stats.FGA3
    season_stats["FG3Rate"] = season_stats.FGA3 / season_stats.FGA

    # FG 2-point makes, FG 2-point attempts, FG 2-point percentage, FG 2-point attempt rate
    season_stats["FG2M"] = season_stats.FGM - season_stats.FGM3
    season_stats["FG2A"] = season_stats.FGA - season_stats.FGA3
    season_stats["FG2Pct"] = season_stats["FG2M"] / season_stats["FG2A"]
    season_stats["FG2Rate"] = season_stats["FG2A"] / season_stats.FGA

    # Free Throw Percentage, Free Throw attempt rate
    season_stats["FTPct"] = season_stats.FTM / season_stats.FTA
    season_stats["FTRate"] = season_stats.FTA / season_stats.FGA

    # effective FG percentage, True Shooting Percentage
    season_stats["eFGPct"] = (season_stats.FGM + 0.5 * season_stats.FGM3) / season_stats.FGA
    season_stats["TSPct"] = season_stats.Score / (2 * (season_stats.FGA + 0.475 * season_stats.FTA))

    # Assist Rate, AstPG
    season_stats["AstRate"] = season_stats.Ast / season_stats.FGM
    season_stats["AstPG"] = season_stats.Ast / season_stats.Games

    # Possessions, Offensive Rating, Defensive Rating, Net Rating
    season_stats["OffRtg"] = 100 * season_stats.Score / season_stats["Poss"]
    season_stats["DefRtg"] = 100 * season_stats.OppScore / season_stats["Poss"]
    season_stats["NetRtg"] = season_stats["OffRtg"] - season_stats["DefRtg"]

    # Steals per game, Blocks per game, Steal Rate, Block Rate
    season_stats["StlPG"] = season_stats.Stl / season_stats.Games
    season_stats["BlkPG"] = season_stats.Blk / season_stats.Games
    season_stats["StlRate"] = season_stats.Stl / season_stats["Poss"]
    season_stats["BlkRate"] = season_stats.Blk / season_stats.FGA

    # Possessions per game
    season_stats["PossPG"] = season_stats["Poss"] / season_stats.Games

    # Turnover Rate, Turnovers per game
    season_stats["TORate"] = season_stats.TO / season_stats["Poss"]
    season_stats["TOPG"] = season_stats.TO / season_stats.Games

    # Offensive Rebound Rate, ORPG, DRPG
    season_stats["ORRate"] = season_stats.OR / (season_stats.OR + season_stats.DR)
    season_stats["ORPG"] = season_stats.OR / season_stats.Games
    season_stats["DRPG"] = season_stats.DR / season_stats.Games

    # Points Per game, Opponent Points per game, Point Differential, Points per possession
    season_stats["PPG"] = season_stats.Score / season_stats.Games
    season_stats["OPPG"] = season_stats.OppScore / season_stats.Games
    season_stats["PointDiff"] = season_stats.PPG - season_stats.OPPG
    season_stats["PPP"] = season_stats.Score / season_stats["Poss"]

    team_stats = season_stats.groupby(["Season","TeamID"]).agg({

        # scoring
        "PPG":"mean",
        "OPPG":"mean",
        "PointDiff":"mean",

        # shooting
        "FGPct":"mean",
        "FG2Pct":"mean",
        "FG3Pct":"mean",
        "eFGPct":"mean",

        # shot selection
        "FG3Rate":"mean",

        # turnovers
        "TORate":"mean",
        "TOPG":"mean",

        # rebounding
        "ORPG":"mean",
        "DRPG":"mean",
        "ORRate":"mean",

        # free throws
        "FTRate":"mean",
        "FTPct":"mean",

        # ball movement
        "AstPG":"mean",
        "AstRate":"mean",

        # defensive disruption
        "StlPG":"mean",
        "BlkPG":"mean",

        # efficiency
        "OffRtg":"mean",
        "DefRtg":"mean",

    }).reset_index()

    return team_stats

def add_seeds(team_stats, seeds):
    team_stats = team_stats.merge(
        seeds[["Season","TeamID","Seed"]],
        on=["Season","TeamID"],
        how="inner"
    )

    team_stats['Seed'] = team_stats['Seed'].str[1:3].astype(int)

    return team_stats

def add_FF(team_stats, tourney):
    elite8 = tourney[(tourney["DayNum"] >= 145) & (tourney["DayNum"] <= 146)]
    final_four_teams = elite8[["Season","WTeamID"]].rename(
        columns={"WTeamID":"TeamID"}
    )

    final_four_teams["FinalFour"] = 1

    stats_data = team_stats.merge(
        final_four_teams,
        on=["Season","TeamID"],
        how="left"
    )

    stats_data = stats_data[stats_data["Season"] != 2026]

    stats_data["FinalFour"] = stats_data["FinalFour"].fillna(0)

    return stats_data

def create_summary(stats_data, probs, teams, seeds, y, X):
    results = stats_data.loc[X.index, ['Season', 'TeamID', "Seed"]].copy()
    results['Prediction'] = probs
    results['Actual'] = y.values
    results = results.merge(teams[['TeamID','TeamName']], on='TeamID', how='left')
    results = results.merge(seeds[['Season', 'TeamID', 'Region']], on=['Season', 'TeamID'], how='left')
    results = results[['Season','TeamName','TeamID', 'Seed', 'Region', 'Prediction','Actual']]
    return results