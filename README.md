# NRLPredictionModel

An Elo-based NRL prediction model with:

- walk-forward backtesting on historical results
- current-round tipping based on bookmaker odds versus model odds
- simple season-level data analysis
- bundled historical data refreshed from the public source through `2026-04-06`
- a hybrid ensemble using Elo, recent form, and injury availability

## Current model defaults

The code now uses a stronger historical Elo profile than the original baseline:

- `home_advantage = 45`
- `k_factor = 24`
- `season_carryover = 0.5`

On the bundled regular-season data from `2021-2025`, that configuration produced:

- `64.16%` tip accuracy
- `0.6274` log loss
- `11.21%` ROI using the existing value-bet thresholds of `-40` to `-20`

The previous settings (`home_advantage = 15`, `k_factor = 20`, no season reset) were worse on the same walk-forward test set.

## Hybrid model

Current predictions now combine three experts:

- `elo`: long-run team strength
- `form`: recent win rate, margin trend, and rest profile
- `injury`: team availability plus fatigue pressure

The final pick uses ensemble voting, so:

- `unanimous` means all `3/3` experts agree
- `2_of_3` means the majority agrees, but one expert disagrees

There is also an upset-risk cap to reduce overconfident probabilities when a favorite has:

- short rest
- a busy recent schedule
- worse recent form than the underdog
- a larger injury burden
- volatile recent results

## Injury file

Update `[injury_data.csv](/C:/Users/SubhanAbid/nrl_tips/NRLPredictionModel/injury_data.csv)` each round to feed the availability expert.

Columns:

- `date`
- `team`
- `injury_score`
- `notes`

Use a higher `injury_score` for bigger absences. If every team is `0`, the injury expert becomes neutral.

## Usage

Run the main script:

```bash
python main.py
```

That will:

1. run a walk-forward backtest
2. fetch the current NRL draw and generate round predictions using the full downloaded history
3. print value bets
4. show simple scoring statistics

## Credits

- Elo Rating System by Arpad Elo
- Glicko 2 rating system by Mark Glickman found at http://www.glicko.net/glicko.html
