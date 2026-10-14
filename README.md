# Reddit Signals · Europe (live)

Live extension of the r/europe inflation and unemployment indicators of Boss, Longo and Onorante, *Nowcasting the euro area with social media data* ([arXiv:2506.10546](https://arxiv.org/abs/2506.10546)), against euro area HICP inflation and unemployment. Companion of the US dashboard (`l-longo/ris-live`).

## Pipeline (`update_dashboard_eu.py`)

Two concepts, inflation and unemployment, each against its euro area target (HICP all items and core; unemployment rate).

1. **History.** The paper's daily series (`subreddits/aaa_series_reddit_new/df_{concept}_signal_subm_comm_0.5_{ma}.csv`, column `signal_comment_numeric`: first-level comments, temperature 0.5) end in December 2024. The files store moving averages; the daily sums are recovered exactly from the 30-day file and checked against every window at each run.
2. **Live.** r/europe submissions with the paper's keywords in the title (inflation: inflation, deflation, hyperinflation, price; unemployment: unemployment, employment, unemployed, job) and their first-level comments posted within a week, labelled UP / DOWN / NEUTRAL by `llama-3.3-70b-instruct` on GPT@JRC with the prompt of Figure 2.2, temperature 0.5.
3. Thread score L_i = (S_i + Σ_j C_ij)/(J+1); UP if L_i > τ, DOWN if L_i < −τ (τ = 0.3, `RIS_TAU`); daily sum by submission date.
4. **Overlap.** Posts from October 2024 are labelled too and compared with the paper's daily series for τ ∈ {0, 0.1, 0.3, 0.5} (`labels/{concept}_overlap_check.csv`).
5. **Outages** of the Reddit collection (no row in any subreddit file): Factiva headlines (titles only) for DE, FR, IT, ES (`subreddits/factiva_topics`), same prompt; daily mean label × average daily number of r/europe threads over the previous 90 days.
6. Moving averages 30–365 days, monthly means; line = best specification (60 days inflation, 90 days unemployment), band = 10–90% across windows; z-scores over 2012–2023. Macro data from the ECB Data Portal.

## Run

```bash
caffeinate -i /opt/anaconda3/bin/python update_dashboard_eu.py      # first run: posts since Oct 2024 (~16k calls)
PUSH=1 /opt/anaconda3/bin/python update_dashboard_eu.py             # publish
```
