# NRLPredictionModel

An Elo-based NRL prediction model with:

- walk-forward backtesting on historical results
- current-round tipping based on bookmaker odds versus model odds
- simple season-level data analysis
- bundled historical data refreshed from the public source through `2026-04-06`
- a hybrid ensemble using market odds, Elo, recent form, and injury availability

## Current model defaults

The current build uses a four-expert hybrid with:

- `home_advantage = 45`
- `k_factor = 24`
- `season_carryover = 0.5`

On the bundled regular-season data from `2021-2025`, the current configuration produced:

- `65.46%` tip accuracy
- `0.6118` log loss
- `0.2109` Brier score
- `0.0365` calibration error
- `11.52%` ROI using the existing value-bet thresholds of `-40` to `-20`

That is stronger than the earlier Elo-only setup and gives a more useful confidence calibration for tipping.

## Hybrid model

Current predictions now combine four experts:

- `market`: normalized implied probability from head-to-head odds
- `elo`: long-run team strength
- `form`: recent win rate, margin trend, and rest profile
- `injury`: team availability plus fatigue pressure

The final pick uses weighted ensemble probability, and the vote split is used as a confidence read:

- `unanimous_4_of_4`
- `strong_3_of_4`
- `split_2_of_4`

There is also an upset-risk cap to reduce overconfident probabilities when a favorite has:

- short rest
- a busy recent schedule
- lost `2` of the last `3` while the underdog is trending up
- a larger injury burden or multiple outs
- extra travel into difficult spots such as Perth or New Zealand

## Injury file

Update `injury_data.csv` each round to feed the availability expert.

Columns:

- `date`
- `team`
- `injury_score`
- `notes`

Use negative values for absences so the injury burden grows with severity:

- `-2.0`: major spine or multiple key outs
- `-1.0`: starting pack or outside-back absence
- `-0.5`: bench or depth hit

## Market file

Update `odds_data.csv` with round-level head-to-head prices.

Columns:

- `date`
- `round_number`
- `home_team`
- `away_team`
- `home_odds`
- `away_odds`
- `venue`
- `bookmaker`
- `source`

The market expert removes overround and turns those prices into calibrated probabilities for the ensemble.

## Weekly review

After a round, feed results into a `round_x_results.csv` style file and generate a review report.

Reports are written to `reports/`, including:

- per-match prediction audit
- which experts were right or wrong
- upset flags versus actual upsets
- favorite overconfidence misses

## Usage

Run the main script:

```bash
python main.py
```

That will:

1. run a walk-forward backtest
2. fetch the current NRL draw and generate round predictions using the full downloaded history
3. print value bets
4. generate a weekly review report when matching round results are available
5. show simple scoring statistics

## Files

- `Functions/all_functions.py`: hybrid model, tuning helpers, and review generation
- `injury_data.csv`: manual weekly availability scores
- `odds_data.csv`: round-level market odds used by the market expert
- `reports/`: generated weekly audits
- `round_6_results.csv`: example results input for review generation

## Credits

- Elo Rating System by Arpad Elo
- Glicko 2 rating system by Mark Glickman found at http://www.glicko.net/glicko.html
