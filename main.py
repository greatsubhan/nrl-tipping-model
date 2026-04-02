import Functions.all_functions as functions


def main():
    # Walk-forward backtest using stronger Elo defaults found on the historical data.
    functions.walk_forward_backtest(
        start_year=2013,
        end_year=2019,
        config=functions.EloConfig(
            home_advantage=45,
            k_factor=24,
            season_carryover=0.5,
            bet_value=50,
            perc_diff_lower_threshold=-40,
            perc_diff_upper_threshold=-20,
        ),
    )

    try:
        round_data_df = functions.get_current_round_data()
        historical_data_df = functions.get_prior_season_data(year=2020, update_file=False, past_years=0)

        # Calculate elo rankings and predictions for current round
        current_round_predicted = functions.predict_current_round(
            round_data_df,
            historical_data_df,
            bet_value=10,
            home_advantage=45,
            k_factor=24,
            season_carryover=0.5,
        )

        functions.value_bets(current_round_predicted, exp_value_threshold=1.5)
    except Exception as exc:
        print(f"Skipping live round predictions because draw data could not be fetched: {exc}")

    # Calculate match point statistics over the period 2015-19
    functions.average_stats(start_year=2019, variable_k_factor=False, years_prior=0)


if __name__ == "__main__":
    main()
