# =============================================================================
# Reddit signals for Europe - live monthly update (GitHub Pages)
#
# Extends the r/europe inflation and unemployment indicators of
#   Boss, Longo, Onorante - "Nowcasting the euro area with social media data"
#   (arXiv:2506.10546)
# beyond the paper's data (which end in December 2024) and compares them with
# euro area HICP inflation (headline, core) and the unemployment rate.
#
# History: the paper's daily series (subreddits/aaa_series_reddit_new,
#   df_{concept}_signal_subm_comm_0.5_{ma}.csv, column signal_comment_numeric:
#   first-level comments, temperature 0.5). The files hold moving averages; the
#   underlying daily sums are recovered exactly from the 30-day file and checked
#   against every other window at each run.
# Live: same rules on newly collected posts
#   1. r/europe submissions whose title contains the paper's keywords, plus their
#      first-level comments posted within one week
#   2. UP / DOWN / NEUTRAL by llama-3.3-70b-instruct on GPT@JRC, prompt of
#      Figure 2.2, temperature 0.5
#   3. thread score L_i = (S_i + sum_j C_ij)/(J+1), UP if L_i > tau, DOWN if
#      L_i < -tau; daily sum by submission date
#   4. Reddit collection outages: Factiva headlines (DE, FR, IT, ES), same prompt,
#      daily mean label x average daily number of threads in the previous 90 days
# Overlap: posts from OVERLAP_START to the end of the paper data are labelled too
#   and compared with the paper's daily series for several tau (calibration).
#
# Run:  python update_dashboard_eu.py          label + build json
#       PUSH=1 python update_dashboard_eu.py   ... and commit/push
#       MOCK_LLM=1 ...                         offline test with keyword labels
# =============================================================================
import os, re, sys, json, time, glob, subprocess
from io import StringIO
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DESK = os.path.expanduser('~/Desktop')
DATA_DIR = os.environ.get('RIS_REDDIT_DIR', os.path.join(HOME_DESK, 'subreddits'))
PAPER_DIR = os.path.join(DATA_DIR, 'aaa_series_reddit_new')
FACTIVA_DIR = os.path.join(DATA_DIR, 'factiva_topics')
TOKEN_FILE = os.environ.get('GPTJRC_TOKEN_FILE', os.path.join(HOME_DESK, 'live_project', 'tokens', 'token-key.jwt'))
JRC_BASE_URL = 'https://api-gpt.jrc.ec.europa.eu/v1'
MODEL = 'llama-3.3-70b-instruct'
TEMPERATURE = 0.5                                                            # paper main analysis (and the files used)
TAU = float(os.environ.get('RIS_TAU', '0.3'))
TAU_GRID = [0.0, 0.1, 0.3, 0.5]
MOCK = os.environ.get('MOCK_LLM', '0') == '1'
PUSH = os.environ.get('PUSH', '0') == '1'
MAX_CALLS = int(os.environ.get('MAX_CALLS', '1000000'))
OVERLAP_START = pd.Timestamp(os.environ.get('RIS_OVERLAP_START', '2024-10-01'))
CHART_START = '2012-01-01'
ZSCORE_START, ZSCORE_END = '2012-01-01', '2023-12-31'                       # paper sample
MA_SEQ = [30, 60, 90, 120, 180, 365]                                         # windows of the paper files
COMMENT_WINDOW = pd.Timedelta(days=7)
FACTIVA_COUNTRIES = ['DE', 'FR', 'IT', 'ES']
SIGNAL_VALUE = {'UP': 1, 'DOWN': -1, 'NEUTRAL': 0}
MAX_CHARS = 6000
CHECKPOINT_EVERY = 50

CONCEPTS = {
    'inflation': {'kw': r'inflation|deflation|hyperinflation|price', 'concept': 'inflation rate', 'ma_main': 60},
    'unemployment': {'kw': r'unemployment|employment|unemployed|job', 'concept': 'unemployment rate', 'ma_main': 90},
}
PROMPT = ("You are a forecaster and want to predict the future {concept} in Europe from textual documents. "
          "The following document will report sentences potentially referring to Europe: {text}. "
          "You have to print a signal that can be UP (if the document is signalling {concept} going up in the short/long-term run), "
          "DOWN (if the sentence is signalling {concept} going down in the short/long-term run) or NEUTRAL "
          "(if the sentence is neutral or does not signal a particular direction on {concept}). "
          "Print only the results of the signal, do not summarize the sentence nor give any reason on your choice. "
          "Even if more sentences or paragraphs are provided to you, you only have to print one signal that can be UP, DOWN or NEUTRAL.")

LABEL_DIR = os.path.join(REPO_DIR, 'labels_mock' if MOCK else 'labels')
OUT_FILE = os.path.join(REPO_DIR, 'docs', 'data', 'dashboard_mock.json' if MOCK else 'dashboard.json')
MACRO_CACHE = os.path.join(REPO_DIR, 'data', 'macro_mock.csv' if MOCK else 'macro_ecb.csv')
for p in [LABEL_DIR, os.path.join(REPO_DIR, 'data'), os.path.dirname(OUT_FILE)]:
    os.makedirs(p, exist_ok=True)

t0 = time.time()
print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] RIS Europe update (mock={MOCK}, temperature={TEMPERATURE}, tau={TAU})')

calls, stop_all, client = 0, False, None
if not MOCK:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=open(TOKEN_FILE).read().strip(), base_url=JRC_BASE_URL, timeout=300.0, max_retries=0)
    except Exception as e:
        print(f'WARNING: GPT@JRC client not available ({type(e).__name__}: {e}) - no new labels this run')
        stop_all = True

# Reddit collection outages: days with no row at all in ANY subreddit file of the folder
raw_days = set()
for f in glob.glob(os.path.join(DATA_DIR, '*_submissions.csv')) + glob.glob(os.path.join(DATA_DIR, '*_comments.csv')):
    raw_days |= set(pd.to_datetime(pd.to_numeric(pd.read_csv(f, usecols=['created_utc'])['created_utc'], errors='coerce'), unit='s').dt.floor('D').dropna())

results, meta_c = {}, {}
for cname, cfg in CONCEPTS.items():
    print(f'\n######## {cname}')
    # ------------------------------------------------------------------------
    # 1. Paper history: recover the daily sums from the moving-average files
    # ------------------------------------------------------------------------
    R = {w: pd.read_csv(os.path.join(PAPER_DIR, f'df_{cname}_signal_subm_comm_0.5_{w}.csv'), index_col=0, parse_dates=True)['signal_comment_numeric'] * w
         for w in MA_SEQ}
    r30 = R[30]
    rv = r30.values
    xv = np.zeros(len(r30))
    xv[0] = rv[29]                                                           # the files start on the first submission day
    for t in range(30, len(r30)):
        xv[t] = xv[t - 30] + (rv[t] - rv[t - 1])
    hist = pd.Series(np.round(xv, 6), index=r30.index)
    chk = max(float(np.nanmax(np.abs((hist.rolling(w).sum() - R[w])[R[w].notna()]))) for w in MA_SEQ)
    paper_end = hist.index.max()
    print(f'paper daily series {hist.index.min().date()} - {paper_end.date()}, recovered exactly: max diff {chk:.1e}')
    if chk > 1e-6:
        sys.exit('paper series could not be recovered exactly')
    live_start = paper_end + pd.Timedelta(days=1)

    # ------------------------------------------------------------------------
    # 2. r/europe corpus from OVERLAP_START (overlap + live)
    # ------------------------------------------------------------------------
    sub = pd.read_csv(os.path.join(DATA_DIR, f'europe_{cname}_submissions.csv'), usecols=['id', 'created_utc', 'title'], dtype={'id': str})
    com = pd.read_csv(os.path.join(DATA_DIR, f'europe_{cname}_comments.csv'),
                      usecols=['id', 'created_utc', 'body', 'parent_id', 'submission_id'], dtype={'id': str, 'submission_id': str})
    sub['created'] = pd.to_datetime(pd.to_numeric(sub['created_utc'], errors='coerce'), unit='s')
    com['created'] = pd.to_datetime(pd.to_numeric(com['created_utc'], errors='coerce'), unit='s')
    data_through = max(sub['created'].max(), com['created'].max()).floor('D')
    sub = sub[(sub['created'] >= OVERLAP_START) & sub['title'].fillna('').str.contains(cfg['kw'], case=False, regex=True)].drop_duplicates('id')
    com = com[com['parent_id'].fillna('').str.startswith('t3_') & com['body'].notna() & ~com['body'].isin(['[deleted]', '[removed]'])]
    com = com.merge(sub[['id', 'created']].rename(columns={'id': 'submission_id', 'created': 'created_sub'}), on='submission_id', how='inner')
    com = com[(com['created'] >= com['created_sub']) & (com['created'] - com['created_sub'] <= COMMENT_WINDOW)].drop_duplicates('id')
    items = pd.concat([
        pd.DataFrame({'id': sub['id'], 'kind': 'submission', 'id_sub': sub['id'], 'created': sub['created'], 'text': sub['title'].astype(str)}),
        pd.DataFrame({'id': com['id'], 'kind': 'comment', 'id_sub': com['submission_id'], 'created': com['created'], 'text': com['body'].astype(str)}),
    ], ignore_index=True)
    print(f'r/europe since {OVERLAP_START.date()}: {len(sub)} submissions, {len(com)} first-level comments within 7 days')

    # ------------------------------------------------------------------------
    # 3. Labels (cached; live period first, then the overlap)
    # ------------------------------------------------------------------------
    sig_file = os.path.join(LABEL_DIR, f'{cname}_signal_llama70b.csv')
    sig = pd.read_csv(sig_file, dtype={'id': str}) if os.path.exists(sig_file) else pd.DataFrame(columns=['id', 'kind', 'label', 'raw', 'labeled_at'])
    todo = items[~(items['id'] + '|' + items['kind']).isin(set(sig['id'].astype(str) + '|' + sig['kind'].astype(str)))].copy()
    todo['_live'] = todo['created'] >= live_start
    todo = todo.sort_values(['_live', 'created'], ascending=[False, True])
    print(f'labels: {len(sig)} cached, {len(todo)} to label')
    new_rows = []
    for row in todo.itertuples(index=False):
        if calls >= MAX_CALLS or stop_all:
            break
        text = row.text[:MAX_CHARS]
        raw = None
        if MOCK:
            t = text.lower()
            u = len(re.findall(r'(rising|rise|surge|soar|higher|increase|accelerat|spike|jump|layoff)', t))
            d = len(re.findall(r'(fall|falling|cool|lower|decrease|decline|easing|slow|drop|hiring)', t))
            raw = 'UP' if u > d else ('DOWN' if d > u else 'NEUTRAL')
        else:
            rl = 0
            for attempt in range(8):
                try:
                    resp = client.chat.completions.create(model=MODEL, temperature=TEMPERATURE,
                                                          messages=[{'role': 'user', 'content': PROMPT.format(concept=cfg['concept'], text=text)}])
                    raw = resp.choices[0].message.content or ''
                    break
                except Exception as e:
                    if 'BadRequest' in type(e).__name__:
                        raw = 'BADREQUEST'; break
                    is_429 = '429' in str(e) or 'RateLimit' in type(e).__name__
                    rl = rl + 1 if is_429 else 0
                    wait = min(45 * 2 ** (rl - 1), 1800) if is_429 else 10
                    print(f'  {row.id}: {type(e).__name__} -> retry in {wait}s')
                    time.sleep(wait)
            calls += 1
            if raw is None:
                print('  GPT@JRC not responding - stopping (progress saved)'); stop_all = True; break
        m = re.search(r'\b(UP|DOWN|NEUTRAL)\b', raw.upper())
        new_rows.append({'id': row.id, 'kind': row.kind, 'label': m.group(1) if m else 'UNPARSED', 'raw': raw.strip()[:40],
                         'labeled_at': datetime.now(timezone.utc).isoformat(timespec='seconds')})
        if len(new_rows) % CHECKPOINT_EVERY == 0:
            pd.concat([sig, pd.DataFrame(new_rows)], ignore_index=True).to_csv(sig_file, index=False)
            print(f'  {cname} {len(new_rows)}/{len(todo)}  ({time.time() - t0:.0f}s)', flush=True)
    sig = pd.concat([sig, pd.DataFrame(new_rows)], ignore_index=True)
    sig.to_csv(sig_file, index=False)

    # ------------------------------------------------------------------------
    # 4. Thread scores and daily sums for each tau of the grid
    # ------------------------------------------------------------------------
    lab = items.merge(sig[['id', 'kind', 'label']], on=['id', 'kind'], how='left')
    lab['label'] = lab['label'].replace({'UNPARSED': 'NEUTRAL', 'BADREQUEST': 'NEUTRAL'})
    done = lab.groupby('id_sub')['label'].apply(lambda s: s.notna().all())
    ok_threads = set(done[done].index)
    n_pending_live = int(lab.loc[lab['created'] >= live_start, 'label'].isna().sum())
    lab = lab[lab['id_sub'].isin(ok_threads)].copy()
    lab['v'] = lab['label'].map(SIGNAL_VALUE).astype(float)
    th = lab.groupby('id_sub').agg(L=('v', 'mean'))                                          # (S_i + sum C_ij)/(J+1)
    th = th.join(sub.set_index('id')['created'].rename('date'))
    th['date'] = th['date'].dt.floor('D')
    incomplete_days = set(sub.loc[~sub['id'].isin(ok_threads), 'created'].dt.floor('D'))
    days = pd.date_range(OVERLAP_START, data_through, freq='D')

    daily_tau = {}
    for tau in sorted(set(TAU_GRID + [TAU])):
        S = np.where(th['L'] > tau, 1, np.where(th['L'] < -tau, -1, 0))
        d = pd.Series(S, index=th.index).groupby(th['date']).sum().reindex(days, fill_value=0).astype(float)
        d.loc[sorted(incomplete_days & set(days))] = np.nan
        daily_tau[tau] = d

    # overlap check against the paper's daily series
    ov = pd.date_range(OVERLAP_START, paper_end, freq='D')
    calib = []
    for tau, d in daily_tau.items():
        a, b = d.reindex(ov), hist.reindex(ov)
        mk = a.notna() & b.notna()
        if mk.sum() > 10:
            calib.append({'tau': tau, 'days': int(mk.sum()), 'corr_daily': round(float(a[mk].corr(b[mk])), 3),
                          'corr_ma30': round(float(a.rolling(30, min_periods=20).mean()[mk].corr(b.rolling(30).mean()[mk])), 3),
                          'mean_new': round(float(a[mk].mean()), 3), 'mean_paper': round(float(b[mk].mean()), 3)})
    if calib:
        print('overlap with the paper (' + str(OVERLAP_START.date()) + ' - ' + str(paper_end.date()) + '):')
        print(pd.DataFrame(calib).to_string(index=False))
        pd.DataFrame(calib).to_csv(os.path.join(LABEL_DIR, f'{cname}_overlap_check.csv'), index=False)

    # live daily series with the chosen tau
    daily = daily_tau[TAU].copy()
    outage = [d for d in days if d not in raw_days and d >= live_start]
    daily.loc[outage] = np.nan

    # ------------------------------------------------------------------------
    # 5. Outage days: Factiva headlines
    # ------------------------------------------------------------------------
    fac_used = []
    parts = []
    for k in FACTIVA_COUNTRIES:
        p = os.path.join(FACTIVA_DIR, f'{k}_{cname}.csv')
        if os.path.exists(p):
            f = pd.read_csv(p, usecols=['an', 'date', 'title'], dtype={'an': str}, low_memory=False)
            f['country'] = k
            parts.append(f)
    if outage and parts:
        fac = pd.concat(parts, ignore_index=True)
        fac['date'] = pd.to_datetime(fac['date'], errors='coerce').dt.floor('D')
        fac = fac.dropna(subset=['date', 'title'])
        fac['title'] = fac['title'].astype(str).str.strip()
        fac = fac.drop_duplicates(['date', 'title'])
        fac_days = [d for d in outage if d <= fac['date'].max()]
        fac = fac[fac['date'].isin(fac_days)]
        fsig_file = os.path.join(LABEL_DIR, f'{cname}_factiva_titles_signal_llama70b.csv')
        fsig = pd.read_csv(fsig_file, dtype={'an': str}) if os.path.exists(fsig_file) else pd.DataFrame(columns=['an', 'date', 'country', 'label', 'raw', 'labeled_at'])
        ftodo = fac[~fac['an'].isin(fsig['an'])]
        print(f'factiva: {len(fac_days)} outage days, {len(fac)} headlines, {len(ftodo)} to label')
        new_rows = []
        for row in ftodo.itertuples(index=False):
            if calls >= MAX_CALLS or stop_all:
                break
            raw = None
            if MOCK:
                raw = ['UP', 'NEUTRAL', 'DOWN'][len(row.title) % 3]
            else:
                rl = 0
                for attempt in range(8):
                    try:
                        resp = client.chat.completions.create(model=MODEL, temperature=TEMPERATURE,
                                                              messages=[{'role': 'user', 'content': PROMPT.format(concept=cfg['concept'], text=row.title)}])
                        raw = resp.choices[0].message.content or ''
                        break
                    except Exception as e:
                        if 'BadRequest' in type(e).__name__:
                            raw = 'BADREQUEST'; break
                        is_429 = '429' in str(e) or 'RateLimit' in type(e).__name__
                        rl = rl + 1 if is_429 else 0
                        wait = min(45 * 2 ** (rl - 1), 1800) if is_429 else 10
                        print(f'  factiva {row.an}: {type(e).__name__} -> retry in {wait}s')
                        time.sleep(wait)
                calls += 1
                if raw is None:
                    print('  GPT@JRC not responding - stopping (progress saved)'); stop_all = True; break
            m = re.search(r'\b(UP|DOWN|NEUTRAL)\b', raw.upper())
            new_rows.append({'an': row.an, 'date': row.date.strftime('%Y-%m-%d'), 'country': row.country, 'label': m.group(1) if m else 'UNPARSED',
                             'raw': raw.strip()[:40], 'labeled_at': datetime.now(timezone.utc).isoformat(timespec='seconds')})
            if len(new_rows) % CHECKPOINT_EVERY == 0:
                pd.concat([fsig, pd.DataFrame(new_rows)], ignore_index=True).to_csv(fsig_file, index=False)
        fsig = pd.concat([fsig, pd.DataFrame(new_rows)], ignore_index=True)
        fsig.to_csv(fsig_file, index=False)
        if fac['an'].isin(fsig['an']).all():
            fl = fac.merge(fsig[['an', 'label']], on='an')
            fl['v'] = fl['label'].replace({'UNPARSED': 'NEUTRAL', 'BADREQUEST': 'NEUTRAL'}).map(SIGNAL_VALUE)
            fmean = fl.groupby('date')['v'].mean()
            # threads per day on Reddit days: paper history + live, previous 90 days
            nthr_live = th.groupby('date').size().reindex(days, fill_value=0).astype(float)
            nthr_live.loc[outage] = np.nan
            scale = nthr_live.rolling(90, min_periods=30).mean().shift(1)
            for d in fac_days:
                daily.loc[d] = float(fmean.get(d, 0.0)) * (float(scale.get(d)) if pd.notna(scale.get(d)) else 0.0)
            fac_used = fac_days
    still_missing = [d for d in outage if d not in set(fac_used)]
    print(f'outage days since {live_start.date()}: {len(outage)}, Factiva {len(fac_used)}, still missing {len(still_missing)}')

    # ------------------------------------------------------------------------
    # 6. Paper history + live, moving averages, monthly means
    # ------------------------------------------------------------------------
    full = pd.concat([hist, daily[daily.index >= live_start]]).asfreq('D')
    full.to_csv(os.path.join(REPO_DIR, 'data', f'daily_{cname}{"_mock" if MOCK else ""}.csv'), header=['signal'], index_label='date')
    var = pd.DataFrame({f'ma{w}': full.rolling(w, min_periods=int(np.ceil(0.8 * w))).mean().resample('ME').mean() for w in MA_SEQ})
    # check: months fully inside the paper data reproduce the paper files
    pm = pd.DataFrame({f'ma{w}': (R[w] / w).resample('ME').mean() for w in MA_SEQ})
    cm = pm.index[(pm.index >= pd.Timestamp('2011-01-31')) & (pm.index < paper_end.to_period('M').to_timestamp())]   # after the first full year of windows
    print(f'monthly check vs paper files: max diff {float(np.nanmax(np.abs(var.loc[cm].values - pm.loc[cm].values))):.1e}')
    results[cname] = var
    meta_c[cname] = {'ma_main': cfg['ma_main'], 'paper_end': paper_end.strftime('%Y-%m-%d'), 'data_through': str(data_through.date()),
                     'n_pending': n_pending_live, 'factiva_days': len(fac_used),
                     'factiva_months': sorted({d.strftime('%Y-%m') for d in fac_used}),
                     'gap_months': sorted({d.strftime('%Y-%m') for d in still_missing}),
                     'n_live': {'submission': int(((lab['kind'] == 'submission') & (lab['created'] >= live_start)).sum()),
                                'comment': int(((lab['kind'] == 'comment') & (lab['created'] >= live_start)).sum())},
                     'overlap': calib}

# ----------------------------------------------------------------------------
# 7. Macro (ECB Data Portal): HICP all items, HICP ex energy/food/alcohol/tobacco,
#    unemployment rate; cache as fallback
# ----------------------------------------------------------------------------
SERIES = [('hicp', 'HICP', 'M.U2.N.000000.4D0.ANR'), ('core', 'HICP', 'M.U2.N.XEF000.4D0.ANR'),
          ('unemployment', 'LFSI', 'M.U2.S.UNEHRT.TOTAL0.15_74.T')]
macro, macro_source = {}, {}
for name, flow, key in SERIES:
    try:
        r = requests.get(f'https://data-api.ecb.europa.eu/service/data/{flow}/{key}',
                         params={'format': 'csvdata', 'startPeriod': '2008-01'}, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        d = pd.read_csv(StringIO(r.text))
        s = pd.Series(pd.to_numeric(d['OBS_VALUE'], errors='coerce').values, index=pd.to_datetime(d['TIME_PERIOD'])).dropna()
        s.index = s.index + pd.offsets.MonthEnd(0)
        macro[name] = s
        macro_source[name] = f'ECB {flow}.{key}'
    except Exception as e:
        print(f'ECB {name}: {type(e).__name__}')
if len(macro) == len(SERIES):
    pd.DataFrame(macro).to_csv(MACRO_CACHE)
elif os.path.exists(MACRO_CACHE):
    cached = pd.read_csv(MACRO_CACHE, index_col=0, parse_dates=True)
    for name, _, _ in SERIES:
        if name not in macro and name in cached:
            macro[name] = cached[name].dropna(); macro_source[name] = 'cache'
for name, _, _ in SERIES:
    if name not in macro:
        macro[name] = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        print(f'WARNING: no {name} data this run')
macro = pd.DataFrame(macro)

# ----------------------------------------------------------------------------
# 8. JSON for the page: z-scores over the paper sample 2012-2023
# ----------------------------------------------------------------------------
last = max([v.index.max() for v in results.values()] + ([macro.dropna(how='all').index.max()] if macro.notna().any().any() else []))
months = pd.date_range(pd.Timestamp(CHART_START) + pd.offsets.MonthEnd(0), last, freq='ME')
rd = lambda x: [None if pd.isna(v) else round(float(v), 3) for v in x]
zs = lambda df: (df - df.loc[ZSCORE_START:ZSCORE_END].mean()) / df.loc[ZSCORE_START:ZSCORE_END].std()
out = {'meta': {'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'mock': MOCK, 'model': MODEL,
                'temperature': TEMPERATURE, 'tau': TAU, 'macro_source': macro_source,
                'macro_last': {k: (macro[k].dropna().index.max().strftime('%Y-%m') if macro[k].notna().any() else None) for k in macro},
                'concepts': meta_c},
       'months': [d.strftime('%Y-%m') for d in months]}
for cname, var in results.items():
    Vz = zs(var).reindex(months)
    nv = Vz.notna().sum(axis=1)
    out[cname] = {'main': rd(Vz[f'ma{CONCEPTS[cname]["ma_main"]}']), 'p10': rd(Vz.quantile(0.10, axis=1).where(nv >= 3)),
                  'p90': rd(Vz.quantile(0.90, axis=1).where(nv >= 3))}
Mz = zs(macro).reindex(months) if macro.notna().any().any() else macro.reindex(months)
M = macro.reindex(months)
for k in macro:
    out[f'macro_{k}'] = {'z': rd(Mz[k]), 'level': rd(M[k])}
with open(OUT_FILE, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'\nwrote {OUT_FILE}: {len(months)} months, GPT@JRC calls {calls}')

# ----------------------------------------------------------------------------
# 9. Publish
# ----------------------------------------------------------------------------
if PUSH and not MOCK:
    subprocess.run(['git', '-C', REPO_DIR, 'add', 'docs/data/dashboard.json', 'data', 'labels'], check=True)
    if subprocess.run(['git', '-C', REPO_DIR, 'diff', '--cached', '--quiet']).returncode != 0:
        subprocess.run(['git', '-C', REPO_DIR, 'commit', '-m', f'RIS Europe update {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC'], check=True)
        subprocess.run(['git', '-C', REPO_DIR, 'push'], check=True)
        print('pushed')
    else:
        print('nothing new to push')
print(f'done in {time.time() - t0:.0f}s')
