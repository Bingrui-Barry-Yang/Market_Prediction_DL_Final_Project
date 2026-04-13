# Stage 1 API Setup

## Purpose

This document summarizes which APIs are needed for Stage 1, where to get the keys, how much data one key can realistically provide, and which provider combination is the best fit for the current project plan.

Stage 1 needs Bitcoin-related news data, especially historical coverage aligned with the project pipeline.

---

## Important Rule

Use your own personal API keys.

Do **not** use random API keys found on the internet.

Reasons:

- they may be stolen, revoked, or rate-limited
- they may be unsafe
- they may expose your data or usage
- they may stop working at any time

---

## Recommended API Stack

### Best match to the current project plan

Recommended combination:

- `NEWSAPI_API_KEY` from NewsAPI
- `GEMINI_API_KEY` from Gemini API
- `SERPAPI_API_KEY` from SerpApi if you want Google News results

This is the closest match to the original wording of the plan, which mentioned a Google News API plus NewsAPI.

### Simpler alternative

If implementation simplicity matters more than strict alignment with "Google News API", use:

- `NEWSAPI_API_KEY`
- `GNEWS_API_KEY`
- `GEMINI_API_KEY`

This is easier to integrate, but it is not literally Google News scraping through SerpApi.

---

## Historical Coverage Warning

As of April 13, 2026:

- NewsAPI documents access to the last 5 years of articles on paid business access
- GNews documents historical coverage from 2020 on paid tiers
- GDELT can search much older news archives, including back to January 1, 2017

Implication:

- NewsAPI and GNews are useful, but they may not be a perfect single-source solution for the full `2017-2021` historical range
- if the project truly requires complete historical coverage from `2017-01-01` to `2021-12-31`, consider adding GDELT as an archival supplement

Recommended practical strategy:

- primary structured sources: NewsAPI + SerpApi or GNews
- optional archival supplement for older coverage: GDELT

---

## API Volume and Rate-Limit Summary

These limits are the main reason one free API key is usually not enough for a serious historical backfill.

| Provider | Good for | Free / entry limit | Articles per request | Historical access | Main constraints |
| --- | --- | --- | --- | --- | --- |
| NewsAPI | structured search | 100 requests/day on Developer | up to 100 | free: 1 month, paid business: up to 5 years | daily quota, history window, free plan delayed by 24h |
| GNews | simpler news API | 100 requests/day on Free | up to 10 on Free | free: 30 days, paid tiers: from 2020 | daily quota, 429 on rate excess, 403 on daily limit |
| SerpApi | Google News-style results | 250 searches/month on Free | multiple results per search | depends on Google News results, not a clean archival database | quota measured in searches, not articles |
| Gemini API | filtering and scoring, not retrieval | varies by project and model | not applicable | not a news source | RPM/TPM/RPD limits and cost |

### Practical interpretation

#### NewsAPI

- Free Developer plan:
  - `100 requests/day`
  - `pageSize` up to `100`
  - delayed by `24 hours`
  - search only up to `1 month` old
- Business plan:
  - `250,000 requests/month`
  - real-time access
  - search up to `5 years` old

Theoretical free maximum:

- about `10,000` returned articles per day if every request returns 100 results

Reality:

- free plan is good for development and testing only
- free plan is not enough for a real `2017-2021` backfill

#### GNews

- Free:
  - `100 requests/day`
  - up to `10` articles per request
  - `12-hour` delay
  - `30 days` of history
- Essential:
  - `1,000 requests/day`
  - up to `25` articles per request
  - real-time
  - historical data from `2020`
- Business:
  - `5,000 requests/day`
  - up to `50` articles per request
  - real-time
  - historical data from `2020`
- Enterprise:
  - `25,000 requests/day`
  - up to `100` articles per request
  - real-time
  - historical data from `2020`

Theoretical free maximum:

- about `1,000` returned articles per day

Reality:

- simpler than SerpApi
- still not enough alone for a full `2017` start date

#### SerpApi

- Free:
  - `250 searches/month`
  - `50 throughput/hour`
- Starter:
  - `1,000 searches/month`
  - `200/hour`
- Developer:
  - `5,000 searches/month`
  - `1,000/hour`
- Production:
  - `15,000 searches/month`
  - `3,000/hour`

Important billing behavior:

- `1 successful search = 1 credit`
- quota is not directly measured in article count
- cached searches within the cache window are free
- cache expires after `1 hour`

Reality:

- useful if Google News discovery is important
- better as a complementary source than the only high-volume backfill source

#### Gemini API

- quota is measured by:
  - requests per minute
  - tokens per minute
  - requests per day
- limits vary by:
  - model
  - project
  - quota tier
- quotas are applied per project, not simply per key
- daily resets are based on Pacific time

Reality:

- not the article collection bottleneck
- the main concern is later filtering/scoring cost and throughput

---

## Recommended Choice

### If you want the project to stay closest to the current plan

Use:

```env
NEWSAPI_API_KEY=...
SERPAPI_API_KEY=...
GEMINI_API_KEY=...
```

### If you want a simpler first implementation

Use:

```env
NEWSAPI_API_KEY=...
GNEWS_API_KEY=...
GEMINI_API_KEY=...
```

### If historical completeness becomes a problem

Add:

- GDELT as an archival supplement for older news discovery

---

## API Key Setup Instructions

### 1. NewsAPI

Official site:

- <https://newsapi.org/docs/get-started>

Steps:

1. Create a NewsAPI account
2. Verify your email if requested
3. Open the dashboard
4. Copy your API key

Put it in `.env`:

```env
NEWSAPI_API_KEY=your_real_key
```

---

### 2. Gemini API

Official docs:

- <https://ai.google.dev/gemini-api/docs/api-key>

Steps:

1. Sign in to Google AI Studio
2. Create or select a project if needed
3. Open the API key page
4. Generate a Gemini API key
5. Copy the key

Put it in `.env`:

```env
GEMINI_API_KEY=your_real_key
```

---

### 3. SerpApi for Google News results

Official docs:

- <https://serpapi.com/google-news-api>
- <https://serpapi.com/users/sign_up>

Steps:

1. Create a SerpApi account
2. Open the account or dashboard page
3. Copy your private API key

Put it in `.env`:

```env
SERPAPI_API_KEY=your_real_key
```

Use this option if you want the closest practical replacement for a "Google News API".

---

### 4. GNews as a simpler alternative

Official docs:

- <https://docs.gnews.io/>

Steps:

1. Create a GNews account
2. Verify your email if needed
3. Open the dashboard
4. Copy your API key

Put it in `.env`:

```env
GNEWS_API_KEY=your_real_key
```

Use this option if you want a simpler implementation path than SerpApi.

---

## Current Project Note

The repository currently has placeholders in `.env` for:

- `GOOGLE_NEWS_API_KEY`
- `NEWSAPI_API_KEY`
- `GEMINI_API_KEY`

Because the project plan is more realistic with SerpApi or GNews than with a vague "Google News API" label, the environment design should likely be updated later to one of:

- `SERPAPI_API_KEY`
- `GNEWS_API_KEY`

instead of relying on `GOOGLE_NEWS_API_KEY` as a generic placeholder name.

---

## Sources

- <https://newsapi.org/pricing>
- <https://newsapi.org/docs/endpoints/everything>
- <https://gnews.io/pricing>
- <https://docs.gnews.io/endpoints/search-endpoint>
- <https://gnews.io/docs/v4>
- <https://serpapi.com/pricing>
- <https://serpapi.com/google-news-api>
- <https://ai.google.dev/gemini-api/docs/rate-limits>
- <https://ai.google.dev/gemini-api/docs/api-key>
- <https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/>
