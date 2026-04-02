from dataclasses import dataclass
import operator as op
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


def margin_multiplier(margin: float) -> float:
    margin = abs(margin)
    if margin <= 1:
        return 1.0
    if margin <= 6:
        return 1.1
    if margin <= 12:
        return 1.3
    return 1.6


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


def _build_prediction_rows(round_data_df: pd.DataFrame, elo_dict: dict, bet_value: float,
                           home_advantage: float) -> pd.DataFrame:
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

        predict_home = expected_home_win_probability(home_current_elo, away_current_elo, home_advantage)
        predict_away = 1 - predict_home
        predict_home_odds = 1 / predict_home
        predict_away_odds = 1 / predict_away

        home_odds = round_data_df.loc[idx, "home_odds"]
        away_odds = round_data_df.loc[idx, "away_odds"]

        home_percentage_diff = ((predict_home_odds - home_odds) / ((predict_home_odds + home_odds) / 2) * 100)
        away_percentage_diff = ((predict_away_odds - away_odds) / ((predict_away_odds + away_odds) / 2) * 100)

        exp_value_home = (predict_home * ((home_odds * bet_value) - bet_value)) - (predict_away * bet_value)
        exp_value_away = (predict_away * ((away_odds * bet_value) - bet_value)) - (predict_home * bet_value)

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
        current_round_df.at[idx, "winner"] = home_team if predict_home >= predict_away else away_team
        current_round_df.at[idx, "probability"] = round(max(predict_home, predict_away), 4)

    return current_round_df


def predict_current_round(round_df: pd.DataFrame, all_df: pd.DataFrame, bet_value: float,
                          home_advantage: float = DEFAULT_HOME_ADVANTAGE,
                          k_factor: float = DEFAULT_K_FACTOR,
                          season_carryover: float = DEFAULT_SEASON_CARRYOVER) -> pd.DataFrame:
    all_data_df = all_df.sort_values("date").reset_index(drop=True)
    team_pool = set(all_data_df["home_team"]).union(all_data_df["away_team"]).union(round_df["home_team"]).union(round_df["away_team"])
    elo_dict = setup_elo(extra_teams=team_pool)
    elo_dict = calculate_elo(
        all_data_df,
        elo_dict,
        k_factor=k_factor,
        variable_k_factor=False,
        home_advantage=home_advantage,
        season_carryover=season_carryover,
    )

    elo_dict_sorted = sorted(elo_dict.items(), key=op.itemgetter(1), reverse=True)
    elo_ladder = pd.DataFrame(elo_dict_sorted, columns=["Team", "Elo"])
    elo_ladder["Elo"] = elo_ladder["Elo"].round(0).astype(int)
    print("\nTeams sorted by Elo rankings:\n" + str(elo_ladder) + "\n")

    current_round_df = _build_prediction_rows(round_df, elo_dict, bet_value, home_advantage)
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
        predict_home = expected_home_win_probability(home_elo, away_elo, config.home_advantage)
        predict_away = 1 - predict_home

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
                    "predicted_winner": home_team if predict_home >= predict_away else away_team,
                    "actual_winner": home_team if actual_home == 1.0 else away_team if actual_home == 0.0 else "Draw",
                    "correct_tip": (predict_home >= 0.5 and actual_home == 1.0) or (predict_home < 0.5 and actual_home == 0.0),
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


def average_stats(start_year: int, variable_k_factor: bool, years_prior: int):
    season_df = get_prior_season_data(start_year, variable_k_factor, years_prior)

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
