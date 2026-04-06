# Market Prediction Deep Learning Final Project

## Project Goal

The main goal of this project is to predict the movement of Bitcoin as accurately as possible. Sentiment analysis may help us observe how market sentiment behaves around price changes.

## Main Idea

The project will compare three models:

1. A data-based baseline that uses Bitcoin market data only.
2. A sentiment-based baseline that uses text-derived sentiment only.
3. A fused model that combines both branches.

The two single-source models are the baselines, and the fused model tests whether using both sources improves performance. Each branch should be tuned on its own before fusion, so the final comparison is fair.

## Prediction Task

The task is to predict Bitcoin movement over a fixed future window, rather than the exact future price.

Two versions are possible:

- Binary classification: `up` or `down`
- Ternary classification: `down`, `flat`, or `up`

The ternary version is more realistic because very small changes are often noise. A typical rule would be:

- `down` when future return is below a negative threshold
- `flat` when return stays inside a small neutral band
- `up` when future return is above a positive threshold

## Dataset Plan

The project needs two synchronized datasets:

- Bitcoin market data
- Bitcoin-related text data

Both datasets must cover the same time period and use timestamps that can be aligned cleanly.

### 1. Market Data

The market dataset provides the numeric input and defines the labels. It will include:

- open
- high
- low
- close
- volume

Possible sources:

- Coinbase public candles: <https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/public/get-public-product-candles>
- Kraken OHLC data: <https://docs.kraken.com/api/docs/rest-api/get-ohlc-data/>
- Alpha Vantage crypto time series: <https://www.alphavantage.co/documentation/>

Planned workflow:

1. Choose one source as the main source of truth.
2. Download BTC data for a fixed date range.
3. Save the raw data exactly as collected.
4. Convert it into one clean table with consistent timestamps and columns.
5. Build labels such as future movement direction from this cleaned table.

The market branch will use both raw OHLCV and engineered features such as returns, moving averages, volatility, and momentum.

### 2. Text Data

The text dataset provides the sentiment side of the model. The safest starting point is Bitcoin-related news because it is easier to filter and align. Reddit can be added later if time allows.

Possible sources for news:

- Alpha Vantage News and Sentiment API: <https://www.alphavantage.co/documentation/>
- Finnhub: <https://finnhubio.github.io/>
- Tiingo API: <https://api.tiingo.com/>
- GDELT news API: <https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/amp/>
- NewsAPI: <https://newsapi.org/docs>

Possible source for Reddit:

- Reddit API overview: <https://developers.reddit.com/docs/capabilities/server/reddit-api>

Planned collection workflow:

1. Query text sources using keywords such as `bitcoin`, `BTC`, and `crypto`.
2. Collect the headline, title, or post text.
3. Save the publish timestamp.
4. Save source metadata such as outlet name, subreddit name, score, or engagement when available.
5. Remove duplicates and obviously irrelevant items.
6. Store the raw text records before any sentiment processing.

For a first version, the text table should include:

- `id`
- `source`
- `published_at`
- `title`
- `body` or summary if available
- `url`
- optional metadata such as score, comment count, or source type

After collection, the text items will be grouped into fixed time windows and aligned with future market labels.

## How FinBERT Fits In

FinBERT will be used inside the sentiment pipeline, not as the final Bitcoin predictor.

Useful references:

- FinBERT paper: <https://huggingface.co/papers/1908.10063>
- FinBERT model: <https://huggingface.co/ProsusAI/finbert>

Planned process:

1. Collect raw headlines or posts.
2. Run each text item through FinBERT.
3. Get sentiment outputs such as positive, negative, and neutral probabilities.
4. Turn those outputs into structured sentiment features.
5. Feed those features into the sentiment model.

So FinBERT acts as the sentiment extractor. It helps convert raw language into signals the prediction model can use.

## Planned Model Architecture

The final system has three parts: a market branch, a sentiment branch, and a fusion module.

### 1. Market-Only Baseline

This model uses only market data.

Input features may include:

- OHLCV
- returns
- rolling means
- rolling volatility
- RSI
- MACD
- candle range and momentum features

Possible architectures:

- LSTM
- GRU
- Transformer encoder

This branch will be trained and tuned on its own first.

### 2. Sentiment-Only Baseline

This model uses only sentiment-related inputs built from timestamped text.

After FinBERT processing, each time window can produce features such as:

- average positive score
- average negative score
- average neutral score
- polarity score
- number of items in the window
- source diversity
- recency-weighted sentiment
- change in sentiment from one window to the next

Possible architectures:

- MLP on aggregated window features
- LSTM or GRU over a sequence of recent sentiment windows
- lightweight Transformer over recent sentiment windows

This branch should also be trained and tuned on its own before fusion.

### 3. Fused Model

After the two baselines are tuned separately, they will be combined into one final model. The goal is not just to stack two vectors together, but to learn how much to trust each branch depending on the situation.

Possible fusion ideas:

- concatenation plus prediction head
- gated fusion
- cross-attention between the two modalities
- gated cross-attention

The preferred direction is an adaptive fusion mechanism, where the model can learn that market history matters more in some cases and sentiment matters more in others.

## Training Strategy

Training should happen in stages.

### Stage 1: Market Branch Tuning

Train and tune the market-only model by testing different sequence lengths, features, learning rates, and architectures.

### Stage 2: Sentiment Branch Tuning

Train and tune the sentiment-only model by testing different time-window definitions, aggregation strategies, and architectures.

### Stage 3: Fusion

Once both branches are strong on their own, combine them into the fused model and train the fusion system. This makes the final comparison more meaningful.

## Planned Experiments

The project should tell a clear experimental story.

### Core Comparison

- Market-only baseline
- Sentiment-only baseline
- Fused model

### Tuning Before Fusion

For both baselines, tuning experiments may include:

- different model types
- different feature sets
- different sequence lengths
- different time windows

This ensures the final fusion result is fair.

### Fusion Experiments

After the baselines are tuned, we can compare a few fusion methods:

- simple fusion
- gated fusion
- attention-based fusion

### Data Source Experiments

If time allows, we can also compare:

- news only
- Reddit only
- news + Reddit

This would tell us whether one kind of text source is more useful than another.

## Evaluation

All models must be evaluated on the same time periods and the same label setup.

The evaluation will likely include:

- accuracy
- F1 score
- precision
- recall

Because this is a Bitcoin prediction project, we may also include a simple market-oriented evaluation such as:

- directional hit rate
- cumulative return under a simple strategy
- maximum drawdown

## What We May Learn Beyond Accuracy

Even though prediction is the main goal, the project may still reveal useful observations, such as:

- whether sentiment helps more during volatile periods
- whether sentiment reacts before price moves or only after them
- whether news and Reddit behave differently
- whether the model learns to rely more on one branch under certain market conditions

These are secondary findings. The central goal remains accurate Bitcoin movement prediction.

## Expected Project Structure

The final repository should be organized as a reproducible project rather than a loose collection of notebooks. A likely structure is:

- `src/data/` for data collection and cleaning
- `src/features/` for market features and sentiment aggregation
- `src/models/` for the market model, sentiment model, and fusion model
- `src/train.py` for training
- `src/evaluate.py` for evaluation
- `configs/` for experiment settings
- `tests/` for leakage checks, alignment checks, and smoke tests

The environment will be managed with `uv` and Docker so the project is reproducible and easy to run.

## New Plans
- Use Google API to get news articles, "top 15 articles on bitcoin on day i"
- Use Coinbase API to get the change in BTC price on day i
- Crunch data (influence score layer?)
- Try to predict price change today based on that
