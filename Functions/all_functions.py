from dataclasses import dataclass
import operator as op
from pathlib import Path
import re
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


DEFAULT_ELO = 1500.0
DEFAULT_HOME_ADVANTAGE = 45.0
DEFAULT_K_FACTOR = 24.0
DEFAULT_SEASON_CARRYOVER = 0.5
DEFAULT_FORM_LOOKBACK = 5
REMOVE_PLAYOFF = True

TEAM_NAME_MAP = {
    "Brisbane Broncos": "Broncos",
    "Canberra Raiders": "Raiders",
    "Canterbury Bulldogs": "Bulldogs",
    "Canterbury-Bankstown Bulldogs": "Bulldogs",
    "Cronulla Sharks": "Sharks",
    "Cronulla-Sutherland Sharks": "Sharks",
    "Dolphins": "Dolphins",
    "Gold Coast Titans": "Titans",
    "Manly Sea Eagles": "Sea Eagles",
    "Manly-Warringah Sea Eagles": "Sea Eagles",
    "Melbourne Storm": "Storm",
    "New Zealand Warriors": "Warriors",
    "Newcastle Knights": "Knights",
    "North QLD Cowboys": "Cowboys",
    "North Queensland Cowboys": "Cowboys",
    "Parramatta Eels": "Eels",
    "Penrith Panthers": "Panthers",
    "South Sydney Rabbitohs": "Rabbitohs",
    "St George Dragons": "Dragons",
    "St. George Illawarra Dragons": "Dragons",
    "Sydney Roosters": "Roosters",
    "Wests Tigers": "Wests Tigers",
}

CORE_TEAMS = [
    "Broncos",
    "Roosters",
    "Warriors",
    "Eels",
    "Dragons",
    "Rabbitohs",
    "Bulldogs",
    "Storm",
    "Sharks",
    "Sea Eagles",
    "Wests Tigers",
    "Raiders",
    "Panthers",
    "Cowboys",
    "Knights",
    "Titans",
    "Dolphins",
]

HISTORICAL_COLUMNS = [
    "Date",
    "Home Team",
    "Away Team",
    "Home Score",
    "Away Score",
    "Play Off Game?",
    "Home Odds",
    "Draw Odds",
    "Away Odds",
]

HISTORICAL_RENAME_MAP = {
    "Date": "date",
    "Home Team": "home_team",
    "Away Team": "away_team",
    "Home Score": "home_score",
    "Away Score": "away_score",
    "Play Off Game?": "play_off",
    "Home Odds": "home_odds",
    "Draw Odds": "draw_odds",
    "Away Odds": "away_odds",
}


@dataclass(frozen=True)
class EloConfig:
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    k_factor: float = DEFAULT_K_FACTOR
    season_carryover: float = DEFAULT_SEASON_CARRYOVER
    bet_value: float = 50.0
    perc_diff_upper_threshold: float = -20.0
    perc_diff_lower_threshold: float = -40.0
    initial_rating: float = DEFAULT_ELO
    form_lookback: int = DEFAULT_FORM_LOOKBACK
    elo_weight: float = 0.5
    form_weight: float = 0.3
    injury_weight: float = 0.2
    injury_penalty_per_point: float = 0.035
    rest_days_target: int = 7
    fatigue_penalty_per_day: float = 0.015
    upset_probability_cap: float = 0.72
    upset_risk_threshold: float = 0.55
    injury_file: str = "injury_data.csv"


def _download_historical_file() -> None:
    url = "http://www.aussportsbetting.com/historical_data/nrl.xlsx"
    header = {"User-Agent": "Mozilla/5.0"}
    local_file = requests.get(url, headers=header, timeout=30)
    local_file.raise_for_status()
    with open("NRL_Historical_Data.xlsx", "wb") as excel_file:
        excel_file.write(local_file.content)


def _normalise_historical_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    historic_df = raw_df[HISTORICAL_COLUMNS].rename(columns=HISTORICAL_RENAME_MAP).copy()
    historic_df["home_team"] = historic_df["home_team"].replace(TEAM_NAME_MAP)
    historic_df["away_team"] = historic_df["away_team"].replace(TEAM_NAME_MAP)
    historic_df["date"] = pd.to_datetime(historic_df["date"])
    historic_df["year"] = historic_df["date"].dt.year
    historic_df["result"] = np.sign(historic_df["home_score"] - historic_df["away_score"])
    historic_df = historic_df.sort_values("date").reset_index(drop=True)

    if REMOVE_PLAYOFF:
        historic_df = historic_df[historic_df["play_off"] != "Y"].reset_index(drop=True)

    return historic_df


def import_data(update_file: bool) -> pd.DataFrame:
    if update_file:
        _download_historical_file()
    else:
        try:
            pd.read_excel("NRL_Historical_Data.xlsx", nrows=1)
        except FileNotFoundError:
            print("Data file does not exist, creating a new file...")
            _download_historical_file()

    raw_df = pd.read_excel("NRL_Historical_Data.xlsx", header=1)
    return _normalise_historical_data(raw_df)


def load_injury_data(file_path: str = "injury_data.csv") -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "team", "injury_score", "notes"])

    injury_df = pd.read_csv(path)
    if injury_df.empty:
        return pd.DataFrame(columns=["date", "team", "injury_score", "notes"])

    injury_df = injury_df.rename(columns=str.lower).copy()
    required_columns = {"date", "team", "injury_score"}
    missing_columns = required_columns.difference(injury_df.columns)
    if missing_columns:
        raise ValueError(f"Missing injury columns: {sorted(missing_columns)}")

    if "notes" not in injury_df.columns:
        injury_df["notes"] = ""

    injury_df["team"] = injury_df["team"].replace(TEAM_NAME_MAP)
    injury_df["date"] = pd.to_datetime(injury_df["date"])
    injury_df["injury_score"] = pd.to_numeric(injury_df["injury_score"], errors="coerce").fillna(0.0)
    return injury_df.sort_values(["date", "team"]).reset_index(drop=True)


def get_team_injury_score(injury_df: pd.DataFrame, team: str, match_date: pd.Timestamp) -> tuple[float, str]:
    if injury_df.empty:
        return 0.0, ""

    eligible_rows = injury_df[(injury_df["team"] == team) & (injury_df["date"] <= match_date)]
    if eligible_rows.empty:
        return 0.0, ""

    latest_row = eligible_rows.iloc[-1]
    return float(latest_row["injury_score"]), str(latest_row.get("notes", ""))


def get_prior_season_data(year: int, update_file: bool, past_years: int) -> pd.DataFrame:
    all_data_df = import_data(update_file)
    start_year = year - past_years
    historic_df = all_data_df[(all_data_df["year"] >= start_year) & (all_data_df["year"] <= year)].copy()
    return historic_df.sort_values("date").reset_index(drop=True)


def get_current_round_data() -> pd.DataFrame:
    url = "https://www.nrl.com/draw/"
    req = Request(url)
    html_page = urlopen(req)
    soup = BeautifulSoup(html_page, "html.parser")
    round_data = soup.find("div", id="vue-draw")

    round_data_list = re.findall(r"(nickName)(.*?)(\d\.?\d*)", str(round_data))
    round_data_list = [item for group in round_data_list for item in group]
    del round_data_list[0::3]

    for idx, item in enumerate(round_data_list):
        item = item.replace("&quot;:&quot;", "")
        item = item.replace("&quot;,&quot;odds", "")
        item = item.replace("&quot;,&quot;score&quot;:", "")
        round_data_list[idx] = TEAM_NAME_MAP.get(item, item)

    home_team = []
    home_odds = []
    away_team = []
    away_odds = []

    while round_data_list:
        home_team.append(round_data_list.pop(0))
        home_odds.append(float(round_data_list.pop(0)))
        away_team.append(round_data_list.pop(0))
        away_odds.append(float(round_data_list.pop(0)))

    round_dict = {
        "home_team": home_team,
        "home_odds": home_odds,
        "away_team": away_team,
        "away_odds": away_odds,
    }
    return pd.DataFrame(round_dict)


def get_current_season_data(curr_round: int, year: int, update_file: bool):
    historic_df = get_prior_season_data(year, update_file, 0)
    if historic_df.empty:
        raise ValueError(f"No regular-season data found for {year}.")

    games_per_round = 4 if len(historic_df) < 192 else 8
    total_games = min(curr_round * games_per_round, len(historic_df))

    current_season_data_df = historic_df.iloc[:total_games].copy().reset_index(drop=True)
    current_round_data_df = current_season_data_df.tail(games_per_round).reset_index(drop=True)
    current_season_data_df = current_season_data_df.iloc[:-len(current_round_data_df)].reset_index(drop=True)

    return current_season_data_df, current_round_data_df


def setup_elo(initial_elo: float = DEFAULT_ELO, extra_teams=None) -> dict:
    teams = set(CORE_TEAMS)
    if extra_teams is not None:
        teams.update(extra_teams)
    return {team: float(initial_elo) for team in teams}


def expected_home_win_probability(home_elo: float, away_elo: float, home_advantage: float) -> float:
    return 1 / (1 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400))


def _clip_probability(probability: float) -> float:
    return float(np.clip(probability, 1e-6, 1 - 1e-6))


def logistic_from_edge(edge: float, scale: float) -> float:
    return _clip_probability(1 / (1 + np.exp(-(edge / scale))))


def margin_multiplier(margin: float) -> float:
    margin = abs(margin)
    if margin <= 1:
        return 1.0
    if margin <= 6:
        return 1.1
    if margin <= 12:
        return 1.3
    return 1.6


def get_team_recent_games(history_df: pd.DataFrame, team: str, before_date: pd.Timestamp, lookback: int) -> pd.DataFrame:
    team_games = history_df[
        ((history_df["home_team"] == team) | (history_df["away_team"] == team)) & (history_df["date"] < before_date)
    ].sort_values("date")
    if lookback > 0:
        team_games = team_games.tail(lookback)
    return team_games.reset_index(drop=True)


def calculate_recent_form_snapshot(history_df: pd.DataFrame, team: str, before_date: pd.Timestamp, lookback: int) -> dict:
    recent_games = get_team_recent_games(history_df, team, before_date, lookback)
    if recent_games.empty:
        return {
            "games": 0,
            "win_rate": 0.5,
            "average_margin": 0.0,
            "margin_std": 0.0,
            "rest_days": 7.0,
            "games_last_21_days": 0,
        }

    team_results = []
    team_margins = []
    for _, row in recent_games.iterrows():
        is_home = row["home_team"] == team
        team_score = row["home_score"] if is_home else row["away_score"]
        opp_score = row["away_score"] if is_home else row["home_score"]
        team_results.append(1.0 if team_score > opp_score else 0.5 if team_score == opp_score else 0.0)
        team_margins.append(float(team_score - opp_score))

    last_game_date = recent_games["date"].max()
    rest_days = max((before_date - last_game_date).days, 0)
    recent_window_start = before_date - pd.Timedelta(days=21)
    games_last_21_days = int((recent_games["date"] >= recent_window_start).sum())

    return {
        "games": int(len(recent_games)),
        "win_rate": float(np.mean(team_results)),
        "average_margin": float(np.mean(team_margins)),
        "margin_std": float(np.std(team_margins)),
        "rest_days": float(rest_days),
        "games_last_21_days": games_last_21_days,
    }


def form_probability(home_form: dict, away_form: dict) -> float:
    win_rate_edge = (home_form["win_rate"] - away_form["win_rate"]) * 30
    margin_edge = home_form["average_margin"] - away_form["average_margin"]
    rest_edge = home_form["rest_days"] - away_form["rest_days"]
    form_edge = win_rate_edge + margin_edge + (rest_edge * 1.5)
    return logistic_from_edge(form_edge, scale=12)


def injury_probability(home_injury_score: float, away_injury_score: float, home_rest_days: float, away_rest_days: float,
                       config: EloConfig) -> float:
    injury_edge = away_injury_score - home_injury_score
    fatigue_edge = away_rest_days - home_rest_days
    edge = (injury_edge * 8) + (fatigue_edge * 2)
    return logistic_from_edge(edge, scale=14)


def combine_expert_probabilities(elo_prob: float, form_prob: float, injury_prob: float, config: EloConfig) -> float:
    total_weight = config.elo_weight + config.form_weight + config.injury_weight
    combined_probability = (
        (elo_prob * config.elo_weight)
        + (form_prob * config.form_weight)
        + (injury_prob * config.injury_weight)
    ) / total_weight
    return _clip_probability(combined_probability)


def compute_upset_risk(home_form: dict, away_form: dict, home_injury_score: float, away_injury_score: float,
                       favorite_side: str, config: EloConfig) -> tuple[float, list[str]]:
    reasons = []
    if favorite_side == "home":
        favorite_form = home_form
        underdog_form = away_form
        favorite_injury = home_injury_score
        underdog_injury = away_injury_score
    else:
        favorite_form = away_form
        underdog_form = home_form
        favorite_injury = away_injury_score
        underdog_injury = home_injury_score

    fatigue_days = max(config.rest_days_target - favorite_form["rest_days"], 0)
    fatigue_risk = min(fatigue_days * config.fatigue_penalty_per_day, 0.3)
    if fatigue_risk > 0:
        reasons.append("short_rest")

    schedule_risk = 0.1 if favorite_form["games_last_21_days"] >= 3 else 0.0
    if schedule_risk > 0:
        reasons.append("busy_schedule")

    injury_gap = max(favorite_injury - underdog_injury, 0)
    injury_risk = min(injury_gap * config.injury_penalty_per_point, 0.35)
    if injury_risk > 0:
        reasons.append("injury_gap")

    form_risk = 0.12 if favorite_form["win_rate"] < underdog_form["win_rate"] else 0.0
    if form_risk > 0:
        reasons.append("recent_form")

    volatility_risk = 0.08 if favorite_form["margin_std"] >= 10 else 0.0
    if volatility_risk > 0:
        reasons.append("volatile_results")

    return fatigue_risk + schedule_risk + injury_risk + form_risk + volatility_risk, reasons


def apply_upset_adjustment(probability: float, upset_risk: float, config: EloConfig) -> tuple[float, bool]:
    adjusted_probability = probability
    upset_flag = upset_risk >= config.upset_risk_threshold
    if upset_flag:
        adjusted_probability = min(adjusted_probability, config.upset_probability_cap)
    elif upset_risk >= config.upset_risk_threshold / 2:
        adjusted_probability = min(adjusted_probability, config.upset_probability_cap + 0.04)

    return _clip_probability(adjusted_probability), upset_flag


def ensemble_vote(home_team: str, away_team: str, expert_probabilities: dict) -> dict:
    expert_picks = {
        expert_name: home_team if probability >= 0.5 else away_team
        for expert_name, probability in expert_probabilities.items()
    }
    home_votes = sum(pick == home_team for pick in expert_picks.values())
    away_votes = len(expert_picks) - home_votes
    majority_team = home_team if home_votes >= away_votes else away_team
    majority_votes = max(home_votes, away_votes)
    if majority_votes == len(expert_picks):
        confidence_label = "unanimous"
    elif majority_votes >= 2:
        confidence_label = "2_of_3"
    else:
        confidence_label = "split"

    return {
        "expert_picks": expert_picks,
        "home_votes": home_votes,
        "away_votes": away_votes,
        "majority_team": majority_team,
        "majority_votes": majority_votes,
        "confidence_label": confidence_label,
    }


def regress_ratings_to_mean(elo_dict: dict, initial_elo: float, carryover: float) -> dict:
    for team, rating in elo_dict.items():
        elo_dict[team] = initial_elo + ((rating - initial_elo) * carryover)
    return elo_dict


def update_elo(elo_dict: dict, home_team: str, away_team: str, home_score: float, away_score: float,
               k_factor: float, home_advantage: float) -> dict:
    home_current_elo = elo_dict[home_team]
    away_current_elo = elo_dict[away_team]
    prob_win_home = expected_home_win_probability(home_current_elo, away_current_elo, home_advantage)

    if home_score > away_score:
        actual_home = 1.0
    elif home_score < away_score:
        actual_home = 0.0
    else:
        actual_home = 0.5

    delta = k_factor * margin_multiplier(home_score - away_score) * (actual_home - prob_win_home)
    elo_dict[home_team] = home_current_elo + delta
    elo_dict[away_team] = away_current_elo - delta
    return elo_dict


def calculate_elo(data_df: pd.DataFrame, elo_dict: dict, k_factor: float, variable_k_factor: bool,
                  home_advantage: float = DEFAULT_HOME_ADVANTAGE, season_carryover: float = 1.0,
                  initial_elo: float = DEFAULT_ELO) -> dict:
    if data_df.empty:
        return elo_dict

    sorted_df = data_df.sort_values("date").reset_index(drop=True)
    previous_year = None

    for idx, row in sorted_df.iterrows():
        current_year = row["year"]
        if previous_year is not None and current_year != previous_year and season_carryover < 1.0:
            regress_ratings_to_mean(elo_dict, initial_elo, season_carryover)

        rating_k = 60 if variable_k_factor and idx < 48 else k_factor
        update_elo(
            elo_dict,
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_score=row["home_score"],
            away_score=row["away_score"],
            k_factor=rating_k,
            home_advantage=home_advantage,
        )
        previous_year = current_year

    return elo_dict


def _build_prediction_rows(round_data_df: pd.DataFrame, elo_dict: dict, all_data_df: pd.DataFrame, config: EloConfig,
                           injury_df: pd.DataFrame) -> pd.DataFrame:
    current_round_df = pd.DataFrame(
        columns=[
            "home_team",
            "calc_odds",
            "real_odds",
            "percent_diff",
            "exp_value_h",
            "away_team",
            "calc_odds_away",
            "real_odds_away",
            "percent_diff_away",
            "exp_value_a",
            "home_win_probability",
            "away_win_probability",
            "elo_probability",
            "form_probability",
            "injury_probability",
            "home_votes",
            "away_votes",
            "ensemble_confidence",
            "upset_risk",
            "upset_flag",
            "upset_reasons",
            "home_injury_score",
            "away_injury_score",
            "home_rest_days",
            "away_rest_days",
            "winner",
            "probability",
        ]
    )

    current_round_df["home_team"] = round_data_df["home_team"]
    current_round_df["away_team"] = round_data_df["away_team"]

    for idx in current_round_df.index:
        home_team = current_round_df.loc[idx, "home_team"]
        away_team = current_round_df.loc[idx, "away_team"]
        home_current_elo = elo_dict[home_team]
        away_current_elo = elo_dict[away_team]
        match_date = all_data_df["date"].max() + pd.Timedelta(days=1)
        home_form = calculate_recent_form_snapshot(all_data_df, home_team, match_date, config.form_lookback)
        away_form = calculate_recent_form_snapshot(all_data_df, away_team, match_date, config.form_lookback)
        home_injury_score, home_injury_notes = get_team_injury_score(injury_df, home_team, match_date)
        away_injury_score, away_injury_notes = get_team_injury_score(injury_df, away_team, match_date)

        elo_probability = expected_home_win_probability(home_current_elo, away_current_elo, config.home_advantage)
        form_prob = form_probability(home_form, away_form)
        injury_prob = injury_probability(
            home_injury_score,
            away_injury_score,
            home_form["rest_days"],
            away_form["rest_days"],
            config,
        )
        predict_home = combine_expert_probabilities(elo_probability, form_prob, injury_prob, config)

        favorite_side = "home" if predict_home >= 0.5 else "away"
        upset_risk, upset_reasons = compute_upset_risk(
            home_form,
            away_form,
            home_injury_score,
            away_injury_score,
            favorite_side=favorite_side,
            config=config,
        )
        predict_home, upset_flag = apply_upset_adjustment(
            predict_home if favorite_side == "home" else 1 - predict_home,
            upset_risk,
            config,
        )
        if favorite_side == "away":
            predict_home = 1 - predict_home

        predict_away = 1 - predict_home
        predict_home_odds = 1 / predict_home
        predict_away_odds = 1 / predict_away
        vote_summary = ensemble_vote(
            home_team,
            away_team,
            {
                "elo": elo_probability,
                "form": form_prob,
                "injury": injury_prob,
            },
        )

        home_odds = round_data_df.loc[idx, "home_odds"]
        away_odds = round_data_df.loc[idx, "away_odds"]

        home_percentage_diff = ((predict_home_odds - home_odds) / ((predict_home_odds + home_odds) / 2) * 100)
        away_percentage_diff = ((predict_away_odds - away_odds) / ((predict_away_odds + away_odds) / 2) * 100)

        exp_value_home = (predict_home * ((home_odds * config.bet_value) - config.bet_value)) - (predict_away * config.bet_value)
        exp_value_away = (predict_away * ((away_odds * config.bet_value) - config.bet_value)) - (predict_home * config.bet_value)

        current_round_df.at[idx, "calc_odds"] = round(predict_home_odds, 2)
        current_round_df.at[idx, "real_odds"] = home_odds
        current_round_df.at[idx, "percent_diff"] = round(home_percentage_diff, 2)
        current_round_df.at[idx, "exp_value_h"] = round(exp_value_home, 2)
        current_round_df.at[idx, "calc_odds_away"] = round(predict_away_odds, 2)
        current_round_df.at[idx, "real_odds_away"] = away_odds
        current_round_df.at[idx, "percent_diff_away"] = round(away_percentage_diff, 2)
        current_round_df.at[idx, "exp_value_a"] = round(exp_value_away, 2)
        current_round_df.at[idx, "home_win_probability"] = round(predict_home, 4)
        current_round_df.at[idx, "away_win_probability"] = round(predict_away, 4)
        current_round_df.at[idx, "elo_probability"] = round(elo_probability, 4)
        current_round_df.at[idx, "form_probability"] = round(form_prob, 4)
        current_round_df.at[idx, "injury_probability"] = round(injury_prob, 4)
        current_round_df.at[idx, "home_votes"] = vote_summary["home_votes"]
        current_round_df.at[idx, "away_votes"] = vote_summary["away_votes"]
        current_round_df.at[idx, "ensemble_confidence"] = vote_summary["confidence_label"]
        current_round_df.at[idx, "upset_risk"] = round(upset_risk, 3)
        current_round_df.at[idx, "upset_flag"] = upset_flag
        current_round_df.at[idx, "upset_reasons"] = ",".join(upset_reasons)
        current_round_df.at[idx, "home_injury_score"] = home_injury_score
        current_round_df.at[idx, "away_injury_score"] = away_injury_score
        current_round_df.at[idx, "home_rest_days"] = round(home_form["rest_days"], 1)
        current_round_df.at[idx, "away_rest_days"] = round(away_form["rest_days"], 1)
        if home_injury_notes:
            current_round_df.at[idx, "upset_reasons"] = ",".join(filter(None, [current_round_df.at[idx, "upset_reasons"], f"home_notes:{home_injury_notes}"]))
        if away_injury_notes:
            current_round_df.at[idx, "upset_reasons"] = ",".join(filter(None, [current_round_df.at[idx, "upset_reasons"], f"away_notes:{away_injury_notes}"]))
        current_round_df.at[idx, "winner"] = vote_summary["majority_team"]
        current_round_df.at[idx, "probability"] = round(max(predict_home, predict_away), 4)

    return current_round_df


def predict_current_round(round_df: pd.DataFrame, all_df: pd.DataFrame, bet_value: float,
                          home_advantage: float = DEFAULT_HOME_ADVANTAGE,
                          k_factor: float = DEFAULT_K_FACTOR,
                          season_carryover: float = DEFAULT_SEASON_CARRYOVER,
                          form_lookback: int = DEFAULT_FORM_LOOKBACK,
                          injury_file: str = "injury_data.csv") -> pd.DataFrame:
    all_data_df = all_df.sort_values("date").reset_index(drop=True)
    config = EloConfig(
        home_advantage=home_advantage,
        k_factor=k_factor,
        season_carryover=season_carryover,
        bet_value=bet_value,
        form_lookback=form_lookback,
        injury_file=injury_file,
    )
    injury_df = load_injury_data(injury_file)
    team_pool = set(all_data_df["home_team"]).union(all_data_df["away_team"]).union(round_df["home_team"]).union(round_df["away_team"])
    elo_dict = setup_elo(extra_teams=team_pool)
    elo_dict = calculate_elo(
        all_data_df,
        elo_dict,
        k_factor=config.k_factor,
        variable_k_factor=False,
        home_advantage=config.home_advantage,
        season_carryover=config.season_carryover,
    )

    elo_dict_sorted = sorted(elo_dict.items(), key=op.itemgetter(1), reverse=True)
    elo_ladder = pd.DataFrame(elo_dict_sorted, columns=["Team", "Elo"])
    elo_ladder["Elo"] = elo_ladder["Elo"].round(0).astype(int)
    print("\nTeams sorted by Elo rankings:\n" + str(elo_ladder) + "\n")

    current_round_df = _build_prediction_rows(round_df, elo_dict, all_data_df, config, injury_df)
    print(current_round_df.to_string())
    return current_round_df


def value_bets(current_round_df: pd.DataFrame, exp_value_threshold: float):
    value_home = current_round_df[current_round_df["exp_value_h"] > exp_value_threshold]
    value_away = current_round_df[current_round_df["exp_value_a"] > exp_value_threshold]

    print("\nValue Bets:")
    if not value_home.empty:
        print("Home Team")
        print(value_home.to_string())

    if not value_away.empty:
        print("\nAway Team")
        print(value_away.to_string())


def _summarise_backtest(results_df: pd.DataFrame, bet_value: float) -> dict:
    total_wagered = float(results_df["bet_placed"].sum() * bet_value)
    profit = float(results_df["bet_profit"].sum())
    roi = (profit / total_wagered) * 100 if total_wagered else 0.0
    bets_lost = int((results_df["bet_placed"] & ~results_df["bet_won"]).sum())
    bets_won = int(results_df["bet_won"].sum())
    average_odds_win = float(results_df.loc[results_df["bet_won"], "bet_odds"].mean()) if bets_won else 0.0
    accuracy = float(results_df["correct_tip"].mean())

    clipped_probs = results_df["home_win_probability"].clip(1e-6, 1 - 1e-6)
    actual_home = results_df["actual_home_win"]
    log_loss = float((-(actual_home * np.log(clipped_probs) + (1 - actual_home) * np.log(1 - clipped_probs))).mean())
    brier_score = float(((clipped_probs - actual_home) ** 2).mean())

    return {
        "profit": profit,
        "total_wagered": total_wagered,
        "roi": roi,
        "bets_placed": int(results_df["bet_placed"].sum()),
        "bets_lost": bets_lost,
        "average_odds_win": average_odds_win,
        "accuracy": accuracy,
        "log_loss": log_loss,
        "brier_score": brier_score,
    }


def walk_forward_backtest(start_year: int, end_year: int, update_file: bool = False, past_years: int | None = None,
                          show_game_data: bool = False, config: EloConfig | None = None):
    config = config or EloConfig()
    all_games_df = import_data(update_file)
    injury_df = load_injury_data(config.injury_file)

    if past_years is not None:
        min_year = max(start_year - past_years, int(all_games_df["year"].min()))
        all_games_df = all_games_df[all_games_df["year"] >= min_year].copy()

    evaluation_df = all_games_df[(all_games_df["year"] >= start_year) & (all_games_df["year"] <= end_year)].copy()
    if evaluation_df.empty:
        raise ValueError(f"No games found between {start_year} and {end_year}.")

    team_pool = set(all_games_df["home_team"]).union(all_games_df["away_team"])
    elo_dict = setup_elo(initial_elo=config.initial_rating, extra_teams=team_pool)
    results = []
    previous_year = None

    for _, row in all_games_df.sort_values("date").iterrows():
        current_year = int(row["year"])
        if previous_year is not None and current_year != previous_year:
            regress_ratings_to_mean(elo_dict, config.initial_rating, config.season_carryover)

        home_team = row["home_team"]
        away_team = row["away_team"]
        home_elo = elo_dict[home_team]
        away_elo = elo_dict[away_team]
        history_before_match = all_games_df[all_games_df["date"] < row["date"]]
        home_form = calculate_recent_form_snapshot(history_before_match, home_team, row["date"], config.form_lookback)
        away_form = calculate_recent_form_snapshot(history_before_match, away_team, row["date"], config.form_lookback)
        home_injury_score, _ = get_team_injury_score(injury_df, home_team, row["date"])
        away_injury_score, _ = get_team_injury_score(injury_df, away_team, row["date"])

        elo_probability = expected_home_win_probability(home_elo, away_elo, config.home_advantage)
        form_prob = form_probability(home_form, away_form)
        injury_prob = injury_probability(
            home_injury_score,
            away_injury_score,
            home_form["rest_days"],
            away_form["rest_days"],
            config,
        )
        predict_home = combine_expert_probabilities(elo_probability, form_prob, injury_prob, config)
        favorite_side = "home" if predict_home >= 0.5 else "away"
        upset_risk, upset_reasons = compute_upset_risk(
            home_form,
            away_form,
            home_injury_score,
            away_injury_score,
            favorite_side=favorite_side,
            config=config,
        )
        favorite_probability, upset_flag = apply_upset_adjustment(
            predict_home if favorite_side == "home" else 1 - predict_home,
            upset_risk,
            config,
        )
        predict_home = favorite_probability if favorite_side == "home" else 1 - favorite_probability
        predict_away = 1 - predict_home
        vote_summary = ensemble_vote(
            home_team,
            away_team,
            {
                "elo": elo_probability,
                "form": form_prob,
                "injury": injury_prob,
            },
        )

        if start_year <= current_year <= end_year:
            predict_home_odds = 1 / predict_home
            predict_away_odds = 1 / predict_away
            home_odds = float(row["home_odds"])
            away_odds = float(row["away_odds"])
            home_percentage_diff = ((predict_home_odds - home_odds) / ((predict_home_odds + home_odds) / 2) * 100)
            away_percentage_diff = ((predict_away_odds - away_odds) / ((predict_away_odds + away_odds) / 2) * 100)

            bet_side = None
            bet_profit = 0.0
            bet_odds = np.nan
            bet_won = False

            if config.perc_diff_lower_threshold < home_percentage_diff < config.perc_diff_upper_threshold:
                bet_side = "home"
                bet_odds = home_odds
            elif config.perc_diff_lower_threshold < away_percentage_diff < config.perc_diff_upper_threshold:
                bet_side = "away"
                bet_odds = away_odds

            if row["home_score"] > row["away_score"]:
                actual_home = 1.0
            elif row["home_score"] < row["away_score"]:
                actual_home = 0.0
            else:
                actual_home = 0.5

            if bet_side == "home":
                bet_won = actual_home == 1.0
                bet_profit = ((home_odds - 1) * config.bet_value) if bet_won else -config.bet_value
            elif bet_side == "away":
                bet_won = actual_home == 0.0
                bet_profit = ((away_odds - 1) * config.bet_value) if bet_won else -config.bet_value

            results.append(
                {
                    "date": row["date"],
                    "year": current_year,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": row["home_score"],
                    "away_score": row["away_score"],
                    "home_win_probability": predict_home,
                    "away_win_probability": predict_away,
                    "elo_probability": elo_probability,
                    "form_probability": form_prob,
                    "injury_probability": injury_prob,
                    "predicted_winner": vote_summary["majority_team"],
                    "probability": max(predict_home, predict_away),
                    "actual_winner": home_team if actual_home == 1.0 else away_team if actual_home == 0.0 else "Draw",
                    "correct_tip": vote_summary["majority_team"] == (home_team if actual_home == 1.0 else away_team if actual_home == 0.0 else "Draw"),
                    "actual_home_win": actual_home,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "home_percentage_diff": home_percentage_diff,
                    "away_percentage_diff": away_percentage_diff,
                    "bet_side": bet_side,
                    "bet_placed": bet_side is not None,
                    "bet_won": bet_won,
                    "bet_odds": bet_odds,
                    "bet_profit": bet_profit,
                    "home_elo_pre": home_elo,
                    "away_elo_pre": away_elo,
                    "home_votes": vote_summary["home_votes"],
                    "away_votes": vote_summary["away_votes"],
                    "ensemble_confidence": vote_summary["confidence_label"],
                    "upset_risk": upset_risk,
                    "upset_flag": upset_flag,
                    "upset_reasons": ",".join(upset_reasons),
                    "home_injury_score": home_injury_score,
                    "away_injury_score": away_injury_score,
                    "home_rest_days": home_form["rest_days"],
                    "away_rest_days": away_form["rest_days"],
                }
            )

            if show_game_data:
                print(
                    f"{row['date'].date()} {home_team} vs {away_team} | "
                    f"p_home={predict_home:.3f} home_odds={home_odds:.2f} away_odds={away_odds:.2f} "
                    f"bet={bet_side or 'none'} result={int(row['home_score'])}-{int(row['away_score'])}"
                )

        update_elo(
            elo_dict,
            home_team=home_team,
            away_team=away_team,
            home_score=row["home_score"],
            away_score=row["away_score"],
            k_factor=config.k_factor,
            home_advantage=config.home_advantage,
        )
        previous_year = current_year

    results_df = pd.DataFrame(results)
    summary = _summarise_backtest(results_df, config.bet_value)

    print(f"\nWalk-forward backtest: {start_year}-{end_year}")
    print(f"Accuracy: {summary['accuracy']:.3%}")
    print(f"Log loss: {summary['log_loss']:.4f}")
    print(f"Brier score: {summary['brier_score']:.4f}")
    print(f"Profit: {summary['profit']:.2f}")
    print(f"Bets Placed: {summary['bets_placed']}")
    print(f"Bets Lost: {summary['bets_lost']}")
    if summary["average_odds_win"]:
        print(f"Average odds on winning bet: {summary['average_odds_win']:.2f}")
    print(f"Total Wagered: {summary['total_wagered']:.2f}")
    print(f"ROI: {summary['roi']:.2f}%\n")

    return summary, results_df


def back_test(year: int, years_prior: int, bet_value: float, perc_diff_upper_threshold: float,
              perc_diff_lower_threshold: float, show_game_data: bool,
              home_advantage: float = DEFAULT_HOME_ADVANTAGE, k_factor: float = DEFAULT_K_FACTOR,
              season_carryover: float = DEFAULT_SEASON_CARRYOVER):
    config = EloConfig(
        home_advantage=home_advantage,
        k_factor=k_factor,
        season_carryover=season_carryover,
        bet_value=bet_value,
        perc_diff_upper_threshold=perc_diff_upper_threshold,
        perc_diff_lower_threshold=perc_diff_lower_threshold,
    )
    summary, _ = walk_forward_backtest(
        start_year=year,
        end_year=year,
        update_file=False,
        past_years=years_prior,
        show_game_data=show_game_data,
        config=config,
    )
    return summary["roi"], summary["total_wagered"], summary["profit"]


def tune_elo_model(start_year: int, end_year: int, home_advantages=None, k_factors=None, carryovers=None,
                   perc_threshold_pairs=None, bet_value: float = 50.0, update_file: bool = False) -> pd.DataFrame:
    home_advantages = home_advantages or [15, 25, 35, 45, 55]
    k_factors = k_factors or [16, 20, 24, 28, 32]
    carryovers = carryovers or [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    perc_threshold_pairs = perc_threshold_pairs or [(-40, -20), (-35, -20), (-30, -25), (-30, -20)]

    tuning_rows = []
    for home_advantage in home_advantages:
        for k_factor in k_factors:
            for carryover in carryovers:
                for lower_threshold, upper_threshold in perc_threshold_pairs:
                    config = EloConfig(
                        home_advantage=home_advantage,
                        k_factor=k_factor,
                        season_carryover=carryover,
                        bet_value=bet_value,
                        perc_diff_lower_threshold=lower_threshold,
                        perc_diff_upper_threshold=upper_threshold,
                    )
                    summary, _ = walk_forward_backtest(
                        start_year=start_year,
                        end_year=end_year,
                        update_file=update_file,
                        config=config,
                    )
                    tuning_rows.append(
                        {
                            "home_advantage": home_advantage,
                            "k_factor": k_factor,
                            "season_carryover": carryover,
                            "perc_diff_lower_threshold": lower_threshold,
                            "perc_diff_upper_threshold": upper_threshold,
                            **summary,
                        }
                    )

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        by=["log_loss", "accuracy", "roi"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return tuning_df


def average_stats(start_year: int, variable_k_factor: bool, years_prior: int, show_plot: bool = False):
    season_df = get_prior_season_data(start_year, variable_k_factor, years_prior)

    if show_plot:
        season_df.plot(x="home_score", y="away_score", kind="scatter", title=f"{start_year} Home vs Away Score")
        plt.show()

    total_points = season_df["home_score"] + season_df["away_score"]
    average_home_score = float(season_df["home_score"].mean())
    average_away_score = float(season_df["away_score"].mean())
    average_points_per_game = float(total_points.mean())
    percent_games_over_50 = float((total_points > 50).mean() * 100)

    print(f"\nAverage statistics across games in the {start_year}-{start_year - years_prior} regular season.")
    print(f"\nGames Evaluated: {len(season_df)}")
    print(f"Average points per game: {average_points_per_game:.2f}")
    print(f"Home: {average_home_score:.2f}")
    print(f"Away: {average_away_score:.2f}")
    print(f"Percent over 50: {percent_games_over_50:.2f}%")
