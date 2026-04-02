# NRLPredictionModel

An Elo-based NRL prediction model with:

- walk-forward backtesting on historical results
- current-round tipping based on bookmaker odds versus model odds
- simple season-level data analysis

## Current model defaults

The code now uses a stronger historical Elo profile than the original baseline:

- `home_advantage = 45`
- `k_factor = 24`
- `season_carryover = 0.5`

On the bundled regular-season data from `2013-2019`, that configuration produced:

- `61.16%` tip accuracy
- `0.6535` log loss
- `4.27%` ROI using the existing value-bet thresholds of `-40` to `-20`

The previous settings (`home_advantage = 15`, `k_factor = 20`, no season reset) were worse on the same walk-forward test set.

## Usage

Run the main script:

```bash
python main.py
```

That will:

1. run a walk-forward backtest
2. fetch the current NRL draw and generate round predictions
3. print value bets
4. show simple scoring statistics

## Credits

- Elo Rating System by Arpad Elo
- Glicko 2 rating system by Mark Glickman found at http://www.glicko.net/glicko.html
