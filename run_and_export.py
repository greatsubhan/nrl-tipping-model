import pandas as pd

def get_predictions_for_round(round_num: int) -> pd.DataFrame:
    # Read your manually created prediction CSV
    df = pd.read_csv(f"model_round_{round_num}.csv")
    # You can optionally add the round here, but we don’t filter on it
    df["round"] = round_num
    return df

if __name__ == "__main__":
    round_num = 3  # your iTipFooty round

    # Load fixtures (all rows are for this round)
    fixtures = pd.read_csv(f"fixtures_round_{round_num}.csv")

    # Read model predictions
    df_preds = get_predictions_for_round(round_num)

    # Merge by home/away team names
    merged = fixtures.merge(
        df_preds[["home_team", "away_team", "winner", "probability"]],
        on=["home_team", "away_team"],
        how="left"
    )

    # Add round column to the output for your tracking
    merged["round"] = round_num

    # Export CSV for your contest
    output = merged[["round", "home_team", "away_team", "winner", "probability"]]
    output.to_csv(f"round_{round_num}_model_tips.csv", index=False)

    print(f"Exported round {round_num} predictions to round_{round_num}_model_tips.csv for your contest tracking.")

