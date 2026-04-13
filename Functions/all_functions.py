from __future__ import annotations

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
DEFAULT_UPSET_CAP = 0.72
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
    "Tigers": "Wests Tigers",
    "Wests Tigers": "Wests Tigers",
}

CORE_TEAMS = [
    "Broncos", "Roosters", "Warriors", "Eels", "Dragons", "Rabbitohs", "Bulldogs",
    "Storm", "Sharks", "Sea Eagles", "Wests Tigers", "Raiders", "Panthers",
    "Cowboys", "Knights", "Titans", "Dolphins",
]

TEAM_BASE_REGION = {
    "Broncos": "queensland", "Bulldogs": "sydney", "Cowboys": "queensland", "Dolphins": "queensland",
    "Dragons": "nsw", "Eels": "sydney", "Knights": "nsw", "Panthers": "sydney",
    "Rabbitohs": "sydney", "Raiders": "canberra", "Roosters": "sydney", "Sea Eagles": "sydney",
    "Sharks": "sydney", "Storm": "melbourne", "Titans": "queensland", "Warriors": "new_zealand",
    "Wests Tigers": "sydney",
}

HISTORICAL_COLUMNS = [
    "Date", "Home Team", "Away Team", "Home Score", "Away Score", "Play Off Game?", "Home Odds", "Draw Odds", "Away Odds",
]

HISTORICAL_RENAME_MAP = {
    "Date": "date", "Home Team": "home_team", "Away Team": "away_team", "Home Score": "home_score",
    "Away Score": "away_score", "Play Off Game?": "play_off", "Home Odds": "home_odds",
    "Draw Odds": "draw_odds", "Away Odds": "away_odds",
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
    injury_file: str = "injury_data.csv"
    odds_file: str = "odds_data.csv"
    review_dir: str = "reports"
    rest_days_target: int = 7
    short_rest_threshold: int = 5
    upset_probability_cap: float = DEFAULT_UPSET_CAP
    soft_probability_cap: float = 0.75
    market_weight_early: float = 0.34
    elo_weight_early: float = 0.34
    form_weight_early: float = 0.18
    injury_weight_early: float = 0.14
    market_weight_late: float = 0.31
    elo_weight_late: float = 0.23
    form_weight_late: float = 0.26
    injury_weight_late: float = 0.20
    injury_weight_boost: float = 0.08
    heavy_injury_threshold: float = 2.0
    injury_gap_threshold: float = 1.5
    multiple_outs_threshold: float = 2.5


def _download_historical_file() -> None:
    url = "http://www.aussportsbetting.com/historical_data/nrl.xlsx"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    Path("NRL_Historical_Data.xlsx").write_bytes(response.content)


def _clip_probability(probability: float) -> float:
    return float(np.clip(probability, 1e-6, 1 - 1e-6))


def _normalise_team_name(team_name: str) -> str:
    return TEAM_NAME_MAP.get(team_name, team_name)


def infer_round_numbers(data_df: pd.DataFrame) -> pd.Series:
    round_numbers = pd.Series(index=data_df.index, dtype=int)
    for _, year_slice in data_df.groupby("year"):
        unique_dates = sorted(pd.to_datetime(year_slice["date"]).dt.normalize().unique())
        date_to_round: dict[pd.Timestamp, int] = {}
        current_round = 1
        previous_date = None
        for current_date in unique_dates:
            timestamp_date = pd.Timestamp(current_date)
            if previous_date is not None and (timestamp_date - previous_date).days > 4:
                current_round += 1
            date_to_round[timestamp_date] = current_round
            previous_date = timestamp_date
        mapped_rounds = pd.to_datetime(year_slice["date"]).dt.normalize().map(date_to_round)
        round_numbers.loc[year_slice.index] = mapped_rounds.astype(int)
    return round_numbers.astype(int)


def _normalise_historical_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    historic_df = raw_df[HISTORICAL_COLUMNS].rename(columns=HISTORICAL_RENAME_MAP).copy()
    historic_df["home_team"] = historic_df["home_team"].replace(TEAM_NAME_MAP)
    historic_df["away_team"] = historic_df["away_team"].replace(TEAM_NAME_MAP)
    historic_df["date"] = pd.to_datetime(historic_df["date"])
    historic_df["year"] = historic_df["date"].dt.year
    if REMOVE_PLAYOFF:
        historic_df = historic_df[historic_df["play_off"] != "Y"].copy()
    historic_df = historic_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    historic_df["round_number"] = infer_round_numbers(historic_df)
    return historic_df


def import_data(update_file: bool) -> pd.DataFrame:
    if update_file:
        _download_historical_file()
    else:
        try:
            pd.read_excel("NRL_Historical_Data.xlsx", nrows=1)
        except FileNotFoundError:
            _download_historical_file()
    return _normalise_historical_data(pd.read_excel("NRL_Historical_Data.xlsx", header=1))


def load_injury_data(file_path: str = "injury_data.csv") -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "team", "injury_score", "injury_burden", "notes"])
    injury_df = pd.read_csv(path).rename(columns=str.lower).copy()
    if injury_df.empty:
        return pd.DataFrame(columns=["date", "team", "injury_score", "injury_burden", "notes"])
    required = {"date", "team", "injury_score"}
    missing = required.difference(injury_df.columns)
    if missing:
        raise ValueError(f"Missing injury columns: {sorted(missing)}")
    if "notes" not in injury_df.columns:
        injury_df["notes"] = ""
    injury_df["team"] = injury_df["team"].map(_normalise_team_name)
    injury_df["date"] = pd.to_datetime(injury_df["date"])
    injury_df["injury_score"] = pd.to_numeric(injury_df["injury_score"], errors="coerce").fillna(0.0)
    injury_df["injury_burden"] = (-injury_df["injury_score"]).clip(lower=0)
    return injury_df.sort_values(["date", "team"]).reset_index(drop=True)


def get_team_injury_score(injury_df: pd.DataFrame, team: str, match_date: pd.Timestamp) -> tuple[float, float, str]:
    if injury_df.empty:
        return 0.0, 0.0, ""
    eligible = injury_df[(injury_df["team"] == team) & (injury_df["date"] <= match_date)]
    if eligible.empty:
        return 0.0, 0.0, ""
    latest = eligible.iloc[-1]
    return float(latest["injury_score"]), float(latest["injury_burden"]), str(latest["notes"])


def load_odds_data(file_path: str = "odds_data.csv") -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "round_number", "home_team", "away_team", "home_odds", "away_odds", "venue", "bookmaker", "source"])
    odds_df = pd.read_csv(path).rename(columns=str.lower).copy()
    if odds_df.empty:
        return pd.DataFrame(columns=["date", "round_number", "home_team", "away_team", "home_odds", "away_odds", "venue", "bookmaker", "source"])
    required = {"home_team", "away_team", "home_odds", "away_odds"}
    missing = required.difference(odds_df.columns)
    if missing:
        raise ValueError(f"Missing odds columns: {sorted(missing)}")
    odds_df["date"] = pd.to_datetime(odds_df.get("date"), errors="coerce")
    odds_df["round_number"] = pd.to_numeric(odds_df.get("round_number"), errors="coerce")
    for column in ["venue", "bookmaker", "source"]:
        if column not in odds_df.columns:
            odds_df[column] = ""
    odds_df["home_team"] = odds_df["home_team"].map(_normalise_team_name)
    odds_df["away_team"] = odds_df["away_team"].map(_normalise_team_name)
    odds_df["home_odds"] = pd.to_numeric(odds_df["home_odds"], errors="coerce")
    odds_df["away_odds"] = pd.to_numeric(odds_df["away_odds"], errors="coerce")
    return odds_df.dropna(subset=["home_odds", "away_odds"]).sort_values(["date", "round_number", "home_team", "away_team"]).reset_index(drop=True)

def market_probabilities_from_odds(home_odds: float, away_odds: float) -> tuple[float, float]:
    home_implied = 1 / home_odds
    away_implied = 1 / away_odds
    overround = home_implied + away_implied
    if overround <= 0:
        return 0.5, 0.5
    return _clip_probability(home_implied / overround), _clip_probability(away_implied / overround)


def find_market_row(market_df: pd.DataFrame, match_date: pd.Timestamp, round_number: int | None, home_team: str,
                    away_team: str) -> pd.Series | None:
    if market_df.empty:
        return None
    candidates = market_df[(market_df["home_team"] == home_team) & (market_df["away_team"] == away_team)].copy()
    if candidates.empty:
        return None
    if round_number is not None:
        round_matches = candidates[candidates["round_number"] == round_number]
        if not round_matches.empty:
            candidates = round_matches
    if pd.notna(match_date):
        dated_matches = candidates[candidates["date"].notna() & (candidates["date"] <= match_date)]
        if not dated_matches.empty:
            candidates = dated_matches
    return candidates.iloc[-1]


def get_prior_season_data(year: int, update_file: bool, past_years: int) -> pd.DataFrame:
    all_data_df = import_data(update_file)
    start_year = year - past_years
    return all_data_df[(all_data_df["year"] >= start_year) & (all_data_df["year"] <= year)].copy().reset_index(drop=True)


def get_current_round_data() -> pd.DataFrame:
    url = "https://www.nrl.com/draw/"
    html_page = urlopen(Request(url))
    soup = BeautifulSoup(html_page, "html.parser")
    round_data = soup.find("div", id="vue-draw")
    round_data_list = re.findall(r"(nickName)(.*?)(\d\.?\d*)", str(round_data))
    round_data_list = [item for group in round_data_list for item in group]
    del round_data_list[0::3]
    for idx, item in enumerate(round_data_list):
        item = item.replace("&quot;:&quot;", "")
        item = item.replace("&quot;,&quot;odds", "")
        item = item.replace("&quot;,&quot;score&quot;:", "")
        round_data_list[idx] = _normalise_team_name(item)
    home_team, home_odds, away_team, away_odds = [], [], [], []
    while round_data_list:
        home_team.append(round_data_list.pop(0))
        home_odds.append(float(round_data_list.pop(0)))
        away_team.append(round_data_list.pop(0))
        away_odds.append(float(round_data_list.pop(0)))
    return pd.DataFrame({"home_team": home_team, "home_odds": home_odds, "away_team": away_team, "away_odds": away_odds})


def get_current_season_data(curr_round: int, year: int, update_file: bool):
    season_df = get_prior_season_data(year, update_file, 0)
    current_season_df = season_df[season_df["round_number"] < curr_round].copy().reset_index(drop=True)
    current_round_df = season_df[season_df["round_number"] == curr_round].copy().reset_index(drop=True)
    return current_season_df, current_round_df


def setup_elo(initial_elo: float = DEFAULT_ELO, extra_teams=None) -> dict[str, float]:
    teams = set(CORE_TEAMS)
    if extra_teams is not None:
        teams.update(extra_teams)
    return {team: float(initial_elo) for team in teams}


def expected_home_win_probability(home_elo: float, away_elo: float, home_advantage: float) -> float:
    return _clip_probability(1 / (1 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400)))


def margin_multiplier(margin: float) -> float:
    margin = abs(margin)
    if margin <= 1:
        return 1.0
    if margin <= 6:
        return 1.1
    if margin <= 12:
        return 1.3
    return 1.6


def regress_ratings_to_mean(elo_dict: dict[str, float], initial_elo: float, carryover: float) -> dict[str, float]:
    for team, rating in elo_dict.items():
        elo_dict[team] = initial_elo + ((rating - initial_elo) * carryover)
    return elo_dict


def update_elo(elo_dict: dict[str, float], home_team: str, away_team: str, home_score: float, away_score: float,
               k_factor: float, home_advantage: float) -> dict[str, float]:
    home_rating = elo_dict[home_team]
    away_rating = elo_dict[away_team]
    expected_home = expected_home_win_probability(home_rating, away_rating, home_advantage)
    if home_score > away_score:
        actual_home = 1.0
    elif home_score < away_score:
        actual_home = 0.0
    else:
        actual_home = 0.5
    delta = k_factor * margin_multiplier(home_score - away_score) * (actual_home - expected_home)
    elo_dict[home_team] = home_rating + delta
    elo_dict[away_team] = away_rating - delta
    return elo_dict


def calculate_elo(data_df: pd.DataFrame, elo_dict: dict[str, float], k_factor: float, variable_k_factor: bool,
                  home_advantage: float = DEFAULT_HOME_ADVANTAGE, season_carryover: float = 1.0,
                  initial_elo: float = DEFAULT_ELO) -> dict[str, float]:
    previous_year = None
    for idx, row in data_df.sort_values("date").iterrows():
        current_year = int(row["year"])
        if previous_year is not None and current_year != previous_year and season_carryover < 1.0:
            regress_ratings_to_mean(elo_dict, initial_elo, season_carryover)
        applied_k = 60 if variable_k_factor and idx < 48 else k_factor
        update_elo(elo_dict, row["home_team"], row["away_team"], row["home_score"], row["away_score"], applied_k, home_advantage)
        previous_year = current_year
    return elo_dict


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
        return {"games": 0, "win_rate": 0.5, "average_margin": 0.0, "margin_std": 0.0, "rest_days": 7.0, "games_last_21_days": 0, "wins_last_3": 1, "losses_last_3": 1, "last_3_record": "1-1"}
    results, margins, dates = [], [], []
    for _, row in recent_games.iterrows():
        is_home = row["home_team"] == team
        team_score = row["home_score"] if is_home else row["away_score"]
        opp_score = row["away_score"] if is_home else row["home_score"]
        dates.append(row["date"])
        results.append(1.0 if team_score > opp_score else 0.0 if team_score < opp_score else 0.5)
        margins.append(float(team_score - opp_score))
    last_three = results[-3:]
    wins_last_3 = sum(result == 1.0 for result in last_three)
    losses_last_3 = sum(result == 0.0 for result in last_three)
    recent_window_start = before_date - pd.Timedelta(days=21)
    return {
        "games": int(len(recent_games)),
        "win_rate": float(np.mean(results)),
        "average_margin": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
        "rest_days": float(max((before_date - max(dates)).days, 0)),
        "games_last_21_days": int((recent_games["date"] >= recent_window_start).sum()),
        "wins_last_3": int(wins_last_3),
        "losses_last_3": int(losses_last_3),
        "last_3_record": f"{wins_last_3}-{losses_last_3}",
    }


def logistic_from_edge(edge: float, scale: float) -> float:
    return _clip_probability(1 / (1 + np.exp(-(edge / scale))))


def form_probability(home_form: dict, away_form: dict) -> float:
    form_edge = ((home_form["win_rate"] - away_form["win_rate"]) * 24)
    form_edge += home_form["average_margin"] - away_form["average_margin"]
    form_edge += (home_form["rest_days"] - away_form["rest_days"]) * 1.5
    form_edge += (home_form["wins_last_3"] - away_form["wins_last_3"]) * 4
    return logistic_from_edge(form_edge, scale=12)


def injury_probability(home_burden: float, away_burden: float, home_rest_days: float, away_rest_days: float) -> float:
    edge = (away_burden - home_burden) * 10
    edge += (home_rest_days - away_rest_days) * 1.8
    return logistic_from_edge(edge, scale=14)

def venue_region(venue: str) -> str:
    lowered = (venue or "").lower()
    if "optus" in lowered or "perth" in lowered:
        return "perth"
    if "mt smart" in lowered or "go media" in lowered or "auckland" in lowered:
        return "new_zealand"
    if "suncorp" in lowered or "lang park" in lowered or "queensland country bank" in lowered or "cbus" in lowered:
        return "queensland"
    if "aami" in lowered or "melbourne" in lowered:
        return "melbourne"
    if "gio" in lowered or "canberra" in lowered:
        return "canberra"
    if any(token in lowered for token in ["accor", "commbank", "campbelltown", "allianz", "kayo", "pointsbet", "shark park", "brookvale", "4 pines", "win stadium", "sydney", "newcastle", "leichhardt"]):
        return "nsw"
    return ""


def is_local_team(team: str, region: str) -> bool:
    if not region:
        return False
    team_region = TEAM_BASE_REGION.get(team, "")
    if region == "perth":
        return False
    if region == "nsw":
        return team_region in {"sydney", "nsw"}
    return team_region == region


def travel_penalty_for_favorite(home_team: str, away_team: str, favorite_side: str, venue: str) -> tuple[float, list[str]]:
    region = venue_region(venue)
    if not region:
        return 0.0, []
    reasons: list[str] = []
    favorite_team = home_team if favorite_side == "home" else away_team
    underdog_team = away_team if favorite_side == "home" else home_team
    if favorite_side == "away" and is_local_team(home_team, region):
        reasons.append("favorite_travel_local_underdog")
        return 0.14, reasons
    if favorite_side == "home" and is_local_team(away_team, region):
        reasons.append("underdog_local")
        return 0.08, reasons
    if favorite_side == "away" and region in {"perth", "new_zealand"}:
        reasons.append("long_haul_away_favorite")
        return 0.14, reasons
    if favorite_side == "home" and region in {"perth", "new_zealand"} and not is_local_team(favorite_team, region) and is_local_team(underdog_team, region):
        reasons.append("away_local_edge")
        return 0.14, reasons
    return 0.0, []


def resolve_match_market_info(match_row: pd.Series, market_df: pd.DataFrame) -> dict:
    home_odds = float(match_row["home_odds"])
    away_odds = float(match_row["away_odds"])
    venue = str(match_row.get("venue", ""))
    bookmaker = str(match_row.get("bookmaker", ""))
    source = str(match_row.get("source", ""))
    market_row = find_market_row(market_df, pd.to_datetime(match_row.get("date"), errors="coerce"), int(match_row["round_number"]) if pd.notna(match_row.get("round_number")) else None, match_row["home_team"], match_row["away_team"])
    if market_row is not None:
        home_odds = float(market_row["home_odds"])
        away_odds = float(market_row["away_odds"])
        venue = str(market_row.get("venue", venue))
        bookmaker = str(market_row.get("bookmaker", bookmaker))
        source = str(market_row.get("source", source))
    market_home, market_away = market_probabilities_from_odds(home_odds, away_odds)
    return {"home_odds": home_odds, "away_odds": away_odds, "venue": venue, "bookmaker": bookmaker, "source": source, "market_home_probability": market_home, "market_away_probability": market_away}


def determine_expert_weights(round_number: int, home_burden: float, away_burden: float, config: EloConfig) -> dict[str, float]:
    if round_number <= 5:
        weights = {"market": config.market_weight_early, "elo": config.elo_weight_early, "form": config.form_weight_early, "injury": config.injury_weight_early}
    else:
        weights = {"market": config.market_weight_late, "elo": config.elo_weight_late, "form": config.form_weight_late, "injury": config.injury_weight_late}
    if max(home_burden, away_burden) >= config.heavy_injury_threshold:
        weights["injury"] += config.injury_weight_boost
        weights["elo"] = max(weights["elo"] - (config.injury_weight_boost / 2), 0.05)
        weights["form"] = max(weights["form"] - (config.injury_weight_boost / 2), 0.05)
    total_weight = sum(weights.values())
    return {expert: weight / total_weight for expert, weight in weights.items()}


def combine_expert_probabilities(expert_probabilities: dict[str, float], weights: dict[str, float]) -> float:
    return _clip_probability(sum(expert_probabilities[name] * weights[name] for name in weights))


def expert_vote_summary(home_team: str, away_team: str, expert_probabilities: dict[str, float]) -> dict:
    expert_picks = {name: home_team if probability >= 0.5 else away_team for name, probability in expert_probabilities.items()}
    home_votes = sum(pick == home_team for pick in expert_picks.values())
    away_votes = len(expert_picks) - home_votes
    majority_team = home_team if home_votes > away_votes else away_team if away_votes > home_votes else ""
    majority_votes = max(home_votes, away_votes)
    if majority_votes == len(expert_picks):
        confidence = "unanimous_4_of_4"
    elif majority_votes == 3:
        confidence = "strong_3_of_4"
    else:
        confidence = "split_2_of_4"
    return {"expert_picks": expert_picks, "home_votes": home_votes, "away_votes": away_votes, "majority_team": majority_team, "majority_votes": majority_votes, "confidence_label": confidence}


def compute_upset_adjustment(home_team: str, away_team: str, favorite_side: str, favorite_probability: float,
                             home_form: dict, away_form: dict, home_burden: float, away_burden: float,
                             venue: str, config: EloConfig) -> tuple[float, float, bool, list[str]]:
    reasons: list[str] = []
    risk = 0.0
    if favorite_side == "home":
        favorite_form, underdog_form = home_form, away_form
        favorite_burden, underdog_burden = home_burden, away_burden
    else:
        favorite_form, underdog_form = away_form, home_form
        favorite_burden, underdog_burden = away_burden, home_burden
    if favorite_form["losses_last_3"] >= 2 and underdog_form["wins_last_3"] >= 2:
        risk += 0.22
        reasons.append("favorite_slump_vs_hot_underdog")
    if favorite_burden - underdog_burden >= config.injury_gap_threshold:
        risk += 0.22
        reasons.append("injury_gap")
    if favorite_burden >= config.multiple_outs_threshold:
        risk += 0.16
        reasons.append("multiple_outs")
    if favorite_form["rest_days"] <= config.short_rest_threshold:
        risk += 0.14
        reasons.append("short_rest")
    if favorite_form["games_last_21_days"] >= 3:
        risk += 0.10
        reasons.append("compressed_schedule")
    travel_risk, travel_reasons = travel_penalty_for_favorite(home_team, away_team, favorite_side, venue)
    risk += travel_risk
    reasons.extend(travel_reasons)
    upset_flag = bool(reasons)
    if not upset_flag:
        return favorite_probability, risk, False, reasons
    hard_trigger_count = sum(1 for trigger in ["favorite_slump_vs_hot_underdog", "injury_gap", "multiple_outs", "favorite_travel_local_underdog", "long_haul_away_favorite"] if trigger in reasons)
    cap = config.upset_probability_cap
    if hard_trigger_count >= 2:
        cap = min(cap, 0.70)
    if "multiple_outs" in reasons and "short_rest" in reasons:
        cap = min(cap, 0.68)
    adjusted_probability = min(favorite_probability, cap)
    if risk >= 0.20:
        adjusted_probability = min(adjusted_probability, config.soft_probability_cap)
    return _clip_probability(adjusted_probability), risk, True, reasons


def infer_current_round_number(history_df: pd.DataFrame) -> int:
    current_year_df = history_df[history_df["year"] == history_df["year"].max()]
    return 1 if current_year_df.empty else int(current_year_df["round_number"].max()) + 1


def _prepare_round_dataframe(round_df: pd.DataFrame, history_df: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    prepared = round_df.copy()
    prepared["home_team"] = prepared["home_team"].map(_normalise_team_name)
    prepared["away_team"] = prepared["away_team"].map(_normalise_team_name)
    prepared["home_odds"] = pd.to_numeric(prepared["home_odds"], errors="coerce")
    prepared["away_odds"] = pd.to_numeric(prepared["away_odds"], errors="coerce")
    if "date" not in prepared.columns:
        prepared["date"] = history_df["date"].max() + pd.Timedelta(days=1)
    else:
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
        prepared["date"] = prepared["date"].fillna(history_df["date"].max() + pd.Timedelta(days=1))
    if "round_number" not in prepared.columns:
        prepared["round_number"] = infer_current_round_number(history_df)
    else:
        prepared["round_number"] = pd.to_numeric(prepared["round_number"], errors="coerce")
        prepared["round_number"] = prepared["round_number"].fillna(infer_current_round_number(history_df)).astype(int)
    for column in ["venue", "bookmaker", "source"]:
        if column not in prepared.columns:
            prepared[column] = ""
    return prepared.reset_index(drop=True)

def build_match_prediction(match_row: pd.Series, elo_dict: dict[str, float], history_df: pd.DataFrame,
                           config: EloConfig, injury_df: pd.DataFrame, market_df: pd.DataFrame) -> dict:
    home_team = match_row["home_team"]
    away_team = match_row["away_team"]
    match_date = pd.to_datetime(match_row["date"])
    round_number = int(match_row["round_number"])
    home_form = calculate_recent_form_snapshot(history_df, home_team, match_date, config.form_lookback)
    away_form = calculate_recent_form_snapshot(history_df, away_team, match_date, config.form_lookback)
    home_injury_score, home_burden, home_notes = get_team_injury_score(injury_df, home_team, match_date)
    away_injury_score, away_burden, away_notes = get_team_injury_score(injury_df, away_team, match_date)
    market_info = resolve_match_market_info(match_row, market_df)
    elo_home_probability = expected_home_win_probability(elo_dict[home_team], elo_dict[away_team], config.home_advantage)
    form_home_probability = form_probability(home_form, away_form)
    injury_home_probability = injury_probability(home_burden, away_burden, home_form["rest_days"], away_form["rest_days"])
    expert_probabilities = {"market": market_info["market_home_probability"], "elo": elo_home_probability, "form": form_home_probability, "injury": injury_home_probability}
    weights = determine_expert_weights(round_number, home_burden, away_burden, config)
    home_probability = combine_expert_probabilities(expert_probabilities, weights)
    favorite_side = "home" if home_probability >= 0.5 else "away"
    favorite_probability = home_probability if favorite_side == "home" else 1 - home_probability
    adjusted_favorite_probability, upset_risk, upset_flag, upset_reasons = compute_upset_adjustment(home_team, away_team, favorite_side, favorite_probability, home_form, away_form, home_burden, away_burden, market_info["venue"], config)
    home_probability = adjusted_favorite_probability if favorite_side == "home" else 1 - adjusted_favorite_probability
    home_probability = _clip_probability(home_probability)
    away_probability = 1 - home_probability
    vote_summary = expert_vote_summary(home_team, away_team, expert_probabilities)
    predicted_winner = home_team if home_probability >= 0.5 else away_team
    final_probability = max(home_probability, away_probability)
    home_model_odds = 1 / home_probability
    away_model_odds = 1 / away_probability
    home_percentage_diff = ((home_model_odds - market_info["home_odds"]) / ((home_model_odds + market_info["home_odds"]) / 2) * 100)
    away_percentage_diff = ((away_model_odds - market_info["away_odds"]) / ((away_model_odds + market_info["away_odds"]) / 2) * 100)
    exp_value_home = (home_probability * ((market_info["home_odds"] * config.bet_value) - config.bet_value)) - (away_probability * config.bet_value)
    exp_value_away = (away_probability * ((market_info["away_odds"] * config.bet_value) - config.bet_value)) - (home_probability * config.bet_value)
    return {
        "date": match_date,
        "round_number": round_number,
        "venue": market_info["venue"],
        "bookmaker": market_info["bookmaker"],
        "source": market_info["source"],
        "home_team": home_team,
        "away_team": away_team,
        "home_odds": market_info["home_odds"],
        "away_odds": market_info["away_odds"],
        "calc_odds": round(home_model_odds, 2),
        "real_odds": market_info["home_odds"],
        "calc_odds_away": round(away_model_odds, 2),
        "real_odds_away": market_info["away_odds"],
        "percent_diff": round(home_percentage_diff, 2),
        "percent_diff_away": round(away_percentage_diff, 2),
        "exp_value_h": round(exp_value_home, 2),
        "exp_value_a": round(exp_value_away, 2),
        "home_win_probability": round(home_probability, 4),
        "away_win_probability": round(away_probability, 4),
        "probability": round(final_probability, 4),
        "winner": predicted_winner,
        "market_probability": round(market_info["market_home_probability"], 4),
        "elo_probability": round(elo_home_probability, 4),
        "form_probability": round(form_home_probability, 4),
        "injury_probability": round(injury_home_probability, 4),
        "market_weight": round(weights["market"], 3),
        "elo_weight": round(weights["elo"], 3),
        "form_weight": round(weights["form"], 3),
        "injury_weight": round(weights["injury"], 3),
        "home_votes": vote_summary["home_votes"],
        "away_votes": vote_summary["away_votes"],
        "ensemble_confidence": vote_summary["confidence_label"],
        "upset_risk": round(upset_risk, 3),
        "upset_flag": upset_flag,
        "upset_reasons": ",".join(upset_reasons),
        "home_injury_score": home_injury_score,
        "away_injury_score": away_injury_score,
        "home_injury_burden": round(home_burden, 2),
        "away_injury_burden": round(away_burden, 2),
        "home_injury_notes": home_notes,
        "away_injury_notes": away_notes,
        "home_rest_days": round(home_form["rest_days"], 1),
        "away_rest_days": round(away_form["rest_days"], 1),
        "home_last_3_record": home_form["last_3_record"],
        "away_last_3_record": away_form["last_3_record"],
        "market_pick": home_team if market_info["market_home_probability"] >= 0.5 else away_team,
        "elo_pick": home_team if elo_home_probability >= 0.5 else away_team,
        "form_pick": home_team if form_home_probability >= 0.5 else away_team,
        "injury_pick": home_team if injury_home_probability >= 0.5 else away_team,
    }


def predict_current_round(round_df: pd.DataFrame, all_df: pd.DataFrame, bet_value: float,
                          home_advantage: float = DEFAULT_HOME_ADVANTAGE,
                          k_factor: float = DEFAULT_K_FACTOR,
                          season_carryover: float = DEFAULT_SEASON_CARRYOVER,
                          form_lookback: int = DEFAULT_FORM_LOOKBACK,
                          injury_file: str = "injury_data.csv",
                          odds_file: str = "odds_data.csv") -> pd.DataFrame:
    all_data_df = all_df.sort_values("date").reset_index(drop=True)
    config = EloConfig(home_advantage=home_advantage, k_factor=k_factor, season_carryover=season_carryover, bet_value=bet_value, form_lookback=form_lookback, injury_file=injury_file, odds_file=odds_file)
    injury_df = load_injury_data(config.injury_file)
    market_df = load_odds_data(config.odds_file)
    prepared_round_df = _prepare_round_dataframe(round_df, all_data_df, config)
    team_pool = set(all_data_df["home_team"]).union(all_data_df["away_team"]).union(prepared_round_df["home_team"]).union(prepared_round_df["away_team"])
    elo_dict = setup_elo(extra_teams=team_pool)
    elo_dict = calculate_elo(all_data_df, elo_dict, k_factor=config.k_factor, variable_k_factor=False, home_advantage=config.home_advantage, season_carryover=config.season_carryover)
    elo_ladder = pd.DataFrame(sorted(elo_dict.items(), key=op.itemgetter(1), reverse=True), columns=["Team", "Elo"])
    elo_ladder["Elo"] = elo_ladder["Elo"].round(0).astype(int)
    print("\nTeams sorted by Elo rankings:\n" + str(elo_ladder) + "\n")
    predictions_df = pd.DataFrame([build_match_prediction(match_row, elo_dict, all_data_df, config, injury_df, market_df) for _, match_row in prepared_round_df.iterrows()])
    print(predictions_df.to_string(index=False))
    return predictions_df


def value_bets(current_round_df: pd.DataFrame, exp_value_threshold: float):
    value_home = current_round_df[current_round_df["exp_value_h"] > exp_value_threshold]
    value_away = current_round_df[current_round_df["exp_value_a"] > exp_value_threshold]
    print("\nValue Bets:")
    if not value_home.empty:
        print("Home Team")
        print(value_home.to_string(index=False))
    if not value_away.empty:
        print("\nAway Team")
        print(value_away.to_string(index=False))


def calibration_error(results_df: pd.DataFrame, probability_column: str = "probability", actual_column: str = "correct_tip") -> float:
    if results_df.empty:
        return 0.0
    calibration_df = results_df.copy()
    calibration_df["bucket"] = pd.cut(calibration_df[probability_column], bins=np.arange(0.5, 1.01, 0.1), include_lowest=True)
    bucket_stats = calibration_df.groupby("bucket", observed=False).agg(observed=(actual_column, "mean"), predicted=(probability_column, "mean"), count=(actual_column, "size"))
    bucket_stats = bucket_stats[bucket_stats["count"] > 0]
    if bucket_stats.empty:
        return 0.0
    return float(np.average(np.abs(bucket_stats["observed"] - bucket_stats["predicted"]), weights=bucket_stats["count"]))


def _summarise_backtest(results_df: pd.DataFrame, bet_value: float) -> dict:
    total_wagered = float(results_df["bet_placed"].sum() * bet_value)
    profit = float(results_df["bet_profit"].sum())
    roi = (profit / total_wagered) * 100 if total_wagered else 0.0
    actual_home = results_df["actual_home_win"]
    home_probabilities = results_df["home_win_probability"].clip(1e-6, 1 - 1e-6)
    log_loss = float((-(actual_home * np.log(home_probabilities) + (1 - actual_home) * np.log(1 - home_probabilities))).mean())
    brier_score = float(((home_probabilities - actual_home) ** 2).mean())
    return {"profit": profit, "total_wagered": total_wagered, "roi": roi, "bets_placed": int(results_df["bet_placed"].sum()), "bets_lost": int((results_df["bet_placed"] & ~results_df["bet_won"]).sum()), "average_odds_win": float(results_df.loc[results_df["bet_won"], "bet_odds"].mean()) if results_df["bet_won"].any() else 0.0, "accuracy": float(results_df["correct_tip"].mean()), "log_loss": log_loss, "brier_score": brier_score, "calibration_error": calibration_error(results_df), "upset_hit_rate": float(results_df.loc[results_df["upset_flag"], "correct_tip"].mean()) if results_df["upset_flag"].any() else 0.0}

def walk_forward_backtest(start_year: int, end_year: int, update_file: bool = False, past_years: int | None = None,
                          show_game_data: bool = False, config: EloConfig | None = None):
    config = config or EloConfig()
    all_games_df = import_data(update_file)
    injury_df = load_injury_data(config.injury_file)
    market_df = load_odds_data(config.odds_file)
    if past_years is not None:
        min_year = max(start_year - past_years, int(all_games_df["year"].min()))
        all_games_df = all_games_df[all_games_df["year"] >= min_year].copy().reset_index(drop=True)
    evaluation_df = all_games_df[(all_games_df["year"] >= start_year) & (all_games_df["year"] <= end_year)]
    if evaluation_df.empty:
        raise ValueError(f"No games found between {start_year} and {end_year}.")
    team_pool = set(all_games_df["home_team"]).union(all_games_df["away_team"])
    elo_dict = setup_elo(initial_elo=config.initial_rating, extra_teams=team_pool)
    results: list[dict] = []
    previous_year = None
    history_so_far = all_games_df.iloc[0:0].copy()
    for _, row in all_games_df.sort_values("date").iterrows():
        current_year = int(row["year"])
        if previous_year is not None and current_year != previous_year:
            regress_ratings_to_mean(elo_dict, config.initial_rating, config.season_carryover)
        prediction = build_match_prediction(row, elo_dict, history_so_far, config, injury_df, market_df)
        if start_year <= current_year <= end_year:
            home_score = float(row["home_score"])
            away_score = float(row["away_score"])
            actual_home = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
            actual_winner = row["home_team"] if actual_home == 1.0 else row["away_team"] if actual_home == 0.0 else "Draw"
            bet_side = None
            bet_odds = np.nan
            bet_profit = 0.0
            bet_won = False
            if config.perc_diff_lower_threshold < prediction["percent_diff"] < config.perc_diff_upper_threshold:
                bet_side = "home"
                bet_odds = prediction["home_odds"]
                bet_won = actual_home == 1.0
                bet_profit = ((bet_odds - 1) * config.bet_value) if bet_won else -config.bet_value
            elif config.perc_diff_lower_threshold < prediction["percent_diff_away"] < config.perc_diff_upper_threshold:
                bet_side = "away"
                bet_odds = prediction["away_odds"]
                bet_won = actual_home == 0.0
                bet_profit = ((bet_odds - 1) * config.bet_value) if bet_won else -config.bet_value
            favorite_from_market = row["home_team"] if prediction["market_probability"] >= 0.5 else row["away_team"]
            upset_occurred = actual_winner != favorite_from_market and actual_winner != "Draw"
            result_row = {**prediction, "year": current_year, "home_score": home_score, "away_score": away_score, "actual_home_win": actual_home, "actual_winner": actual_winner, "correct_tip": prediction["winner"] == actual_winner, "bet_side": bet_side, "bet_placed": bet_side is not None, "bet_odds": bet_odds, "bet_profit": bet_profit, "bet_won": bet_won, "market_favorite": favorite_from_market, "upset_occurred": upset_occurred}
            for expert_name, expert_pick_key in [("market", "market_pick"), ("elo", "elo_pick"), ("form", "form_pick"), ("injury", "injury_pick")]:
                result_row[f"{expert_name}_correct"] = prediction[expert_pick_key] == actual_winner
            results.append(result_row)
            if show_game_data:
                print(f"{row['date'].date()} {row['home_team']} vs {row['away_team']} | pick={prediction['winner']} prob={prediction['probability']:.3f} actual={actual_winner} upset={upset_occurred}")
        update_elo(elo_dict, row["home_team"], row["away_team"], row["home_score"], row["away_score"], config.k_factor, config.home_advantage)
        history_so_far = pd.concat([history_so_far, pd.DataFrame([row])], ignore_index=True)
        previous_year = current_year
    results_df = pd.DataFrame(results)
    summary = _summarise_backtest(results_df, config.bet_value)
    print(f"\nWalk-forward backtest: {start_year}-{end_year}")
    print(f"Accuracy: {summary['accuracy']:.3%}")
    print(f"Log loss: {summary['log_loss']:.4f}")
    print(f"Brier score: {summary['brier_score']:.4f}")
    print(f"Calibration error: {summary['calibration_error']:.4f}")
    print(f"Profit: {summary['profit']:.2f}")
    print(f"Bets Placed: {summary['bets_placed']}")
    print(f"Bets Lost: {summary['bets_lost']}")
    if summary["average_odds_win"]:
        print(f"Average odds on winning bet: {summary['average_odds_win']:.2f}")
    print(f"Total Wagered: {summary['total_wagered']:.2f}")
    print(f"ROI: {summary['roi']:.2f}%")
    print(f"Upset flag hit rate: {summary['upset_hit_rate']:.3%}\n")
    return summary, results_df


def back_test(year: int, years_prior: int, bet_value: float, perc_diff_upper_threshold: float,
              perc_diff_lower_threshold: float, show_game_data: bool,
              home_advantage: float = DEFAULT_HOME_ADVANTAGE, k_factor: float = DEFAULT_K_FACTOR,
              season_carryover: float = DEFAULT_SEASON_CARRYOVER):
    summary, _ = walk_forward_backtest(start_year=year, end_year=year, update_file=False, past_years=years_prior, show_game_data=show_game_data, config=EloConfig(home_advantage=home_advantage, k_factor=k_factor, season_carryover=season_carryover, bet_value=bet_value, perc_diff_upper_threshold=perc_diff_upper_threshold, perc_diff_lower_threshold=perc_diff_lower_threshold))
    return summary["roi"], summary["total_wagered"], summary["profit"]


def tune_hybrid_model(start_year: int = 2021, end_year: int = 2025, update_file: bool = False) -> pd.DataFrame:
    candidate_rows = []
    for home_advantage in [35, 45, 55]:
        for k_factor in [20, 24, 28]:
            for form_lookback in [3, 5, 7]:
                for upset_cap in [0.70, 0.72]:
                    for market_weight_early, elo_weight_early in [(0.34, 0.34), (0.38, 0.30)]:
                        config = EloConfig(home_advantage=home_advantage, k_factor=k_factor, form_lookback=form_lookback, upset_probability_cap=upset_cap, market_weight_early=market_weight_early, elo_weight_early=elo_weight_early, form_weight_early=0.18, injury_weight_early=1 - market_weight_early - elo_weight_early - 0.18, market_weight_late=0.31, elo_weight_late=0.23, form_weight_late=0.26, injury_weight_late=0.20)
                        summary, _ = walk_forward_backtest(start_year, end_year, update_file=update_file, config=config)
                        candidate_rows.append({"home_advantage": home_advantage, "k_factor": k_factor, "form_lookback": form_lookback, "upset_cap": upset_cap, "market_weight_early": market_weight_early, "elo_weight_early": elo_weight_early, **summary})
    return pd.DataFrame(candidate_rows).sort_values(by=["accuracy", "calibration_error", "log_loss", "roi"], ascending=[False, True, True, False]).reset_index(drop=True)


def tune_elo_model(start_year: int, end_year: int, home_advantages=None, k_factors=None, carryovers=None,
                   perc_threshold_pairs=None, bet_value: float = 50.0, update_file: bool = False) -> pd.DataFrame:
    return tune_hybrid_model(start_year=start_year, end_year=end_year, update_file=update_file)


def normalise_results_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    normalised = results_df.copy()
    normalised["home_team"] = normalised["home_team"].map(_normalise_team_name)
    normalised["away_team"] = normalised["away_team"].map(_normalise_team_name)
    normalised["home_score"] = pd.to_numeric(normalised["home_score"], errors="coerce")
    normalised["away_score"] = pd.to_numeric(normalised["away_score"], errors="coerce")
    normalised["actual_winner"] = np.where(normalised["home_score"] > normalised["away_score"], normalised["home_team"], np.where(normalised["away_score"] > normalised["home_score"], normalised["away_team"], "Draw"))
    return normalised


def generate_weekly_review_report(predictions_df: pd.DataFrame, results_df: pd.DataFrame, round_label: str,
                                  output_dir: str = "reports") -> tuple[pd.DataFrame, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    review_df = predictions_df.merge(normalise_results_dataframe(results_df)[["home_team", "away_team", "home_score", "away_score", "actual_winner"]], on=["home_team", "away_team"], how="inner").copy()
    review_df["correct_tip"] = review_df["winner"] == review_df["actual_winner"]
    review_df["market_favorite"] = np.where(review_df["market_probability"] >= 0.5, review_df["home_team"], review_df["away_team"])
    review_df["upset_occurred"] = review_df["actual_winner"] != review_df["market_favorite"]
    review_df["favorite_overconfidence_miss"] = ((review_df["probability"] >= 0.60) & (~review_df["correct_tip"]) & review_df["upset_occurred"])
    for expert_name, pick_key in [("market", "market_pick"), ("elo", "elo_pick"), ("form", "form_pick"), ("injury", "injury_pick")]:
        review_df[f"{expert_name}_correct"] = review_df[pick_key] == review_df["actual_winner"]
    csv_path = output_path / f"{round_label}_review.csv"
    review_df.to_csv(csv_path, index=False)
    summary_lines = [
        f"Round review: {round_label}",
        f"Tips correct: {int(review_df['correct_tip'].sum())}/{len(review_df)}",
        f"Upsets: {int(review_df['upset_occurred'].sum())}",
        f"Favorite overconfidence misses: {int(review_df['favorite_overconfidence_miss'].sum())}",
        f"Market expert correct: {int(review_df['market_correct'].sum())}/{len(review_df)}",
        f"Elo expert correct: {int(review_df['elo_correct'].sum())}/{len(review_df)}",
        f"Form expert correct: {int(review_df['form_correct'].sum())}/{len(review_df)}",
        f"Injury expert correct: {int(review_df['injury_correct'].sum())}/{len(review_df)}",
    ]
    markdown_path = output_path / f"{round_label}_review.md"
    markdown_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return review_df, str(markdown_path)


def average_stats(start_year: int, variable_k_factor: bool, years_prior: int, show_plot: bool = False):
    season_df = get_prior_season_data(start_year, variable_k_factor, years_prior)
    if show_plot:
        season_df.plot(x="home_score", y="away_score", kind="scatter", title=f"{start_year} Home vs Away Score")
        plt.show()
    total_points = season_df["home_score"] + season_df["away_score"]
    print(f"\nAverage statistics across games in the {start_year}-{start_year - years_prior} regular season.")
    print(f"\nGames Evaluated: {len(season_df)}")
    print(f"Average points per game: {float(total_points.mean()):.2f}")
    print(f"Home: {float(season_df['home_score'].mean()):.2f}")
    print(f"Away: {float(season_df['away_score'].mean()):.2f}")
    print(f"Percent over 50: {float((total_points > 50).mean() * 100):.2f}%")
