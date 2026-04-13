from pathlib import Path

import pandas as pd

import Functions.all_functions as functions


def main():
    config = functions.EloConfig(
        home_advantage=45,
        k_factor=24,
        season_carryover=0.5,
        bet_value=50,
        perc_diff_lower_threshold=-40,
        perc_diff_upper_threshold=-20,
        form_lookback=5,
        injury_file="injury_data.csv",
        odds_file="odds_data.csv",
    )

    functions.walk_forward_backtest(
        start_year=2021,
        end_year=2025,
        config=config,
    )

    try:
        round_data_df = functions.get_current_round_data()
        historical_data_df = functions.import_data(update_file=False)
        current_round_predicted = functions.predict_current_round(
            round_data_df,
            historical_data_df,
            bet_value=10,
            home_advantage=config.home_advantage,
            k_factor=config.k_factor,
            season_carryover=config.season_carryover,
            form_lookback=config.form_lookback,
            injury_file=config.injury_file,
            odds_file=config.odds_file,
        )

        functions.value_bets(current_round_predicted, exp_value_threshold=1.5)
    except Exception as exc:
        print(f"Skipping live round predictions because draw data could not be fetched: {exc}")

    round_6_odds = Path("odds_data.csv")
    round_6_results = Path("round_6_results.csv")
    if round_6_odds.exists() and round_6_results.exists():
        review_round_df = pd.read_csv("odds_data.csv")
        review_round_df = review_round_df[review_round_df["round_number"] == 6][["date", "round_number", "home_team", "away_team", "home_odds", "away_odds", "venue", "bookmaker", "source"]]
        historical_data_df = functions.import_data(update_file=False)
        round_6_predictions = functions.predict_current_round(
            review_round_df,
            historical_data_df,
            bet_value=10,
            home_advantage=config.home_advantage,
            k_factor=config.k_factor,
            season_carryover=config.season_carryover,
            form_lookback=config.form_lookback,
            injury_file=config.injury_file,
            odds_file=config.odds_file,
        )
        _, review_path = functions.generate_weekly_review_report(
            round_6_predictions,
            pd.read_csv("round_6_results.csv"),
            round_label="round_6",
            output_dir=config.review_dir,
        )
        print(f"Saved weekly review report to {review_path}")

    functions.average_stats(start_year=2019, variable_k_factor=False, years_prior=0)


if __name__ == "__main__":
    main()
