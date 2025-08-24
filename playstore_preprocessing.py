
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATA_FILENAME = "googleplaystore (1).csv"

def parse_size(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if 'varies' in s: return np.nan
    if s.endswith('m'):
        try: return float(s[:-1])
        except: return np.nan
    if s.endswith('k'):
        try: return float(s[:-1]) / 1024.0
        except: return np.nan
    # try remove commas
    try:
        return float(s.replace(',',''))
    except:
        return np.nan

def parse_price(x):
    if pd.isna(x): return 0.0
    s = str(x).strip()
    if s.lower() in ('free','nan','0'): return 0.0
    s = s.replace('$','').replace(',','').strip()
    try:
        return float(s)
    except:
        return 0.0

def parse_installs(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace('+','').replace(',','').replace('free','0')
    try:
        return int(s)
    except:
        return np.nan

def parse_android(x):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    if 'varies' in s: return np.nan
    s = s.replace('and up','').strip()
    # extract leading number (e.g., '4.1' or '4')
    try:
        return float(s.split()[0])
    except:
        try:
            return float(s.split('-')[0])
        except:
            return np.nan

def cap_outliers_iqr(s):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return s.clip(lower=lower, upper=upper)

def main():
    p = Path(DATA_FILENAME)
    df = pd.read_csv(p)
    # Standardize column names (common)
    # Accept both 'App' and 'Application name'
    if 'App' not in df.columns and 'Application name' in df.columns:
        df.rename(columns={'Application name':'App'}, inplace=True)
    # 1) Rating
    df['Rating'] = pd.to_numeric(df.get('Rating', np.nan), errors='coerce')
    df['Rating'] = df['Rating'].clip(lower=0, upper=5)
    df['Rating'].fillna(df['Rating'].median(), inplace=True)
    # 2) Size -> Size_MB
    df['Size_MB'] = df.get('Size').apply(parse_size)
    df['Size_MB'] = df.groupby('Category')['Size_MB'].transform(lambda s: s.fillna(s.median()))
    df['Size_MB'].fillna(df['Size_MB'].median(), inplace=True)
    # 3) Price
    df['Price_USD'] = df.get('Price').apply(parse_price)
    # 4) Installs
    df['Installs_num'] = df.get('Installs').apply(parse_installs)
    df['Installs_num'].fillna(df['Installs_num'].median(), inplace=True)
    # 5) Reviews numeric
    df['Reviews_num'] = pd.to_numeric(df.get('Reviews',0), errors='coerce').fillna(0).astype(int)
    # 6) Category cleanup
    if 'Category' in df.columns:
        df['Category'] = df['Category'].astype(str).str.strip()
    # 7) Genres explode if needed
    if 'Genres' in df.columns:
        df['Genres'] = df['Genres'].astype(str).str.strip()
    else:
        df['Genres'] = df.get('Category','Unknown')
    # 8) Android version
    # find android column
    android_col = None
    for c in df.columns:
        if 'android' in c.lower():
            android_col = c; break
    if android_col:
        df['Android_min'] = df[android_col].apply(parse_android)
    else:
        df['Android_min'] = np.nan
    # 9) Type (Free/Paid)
    if 'Type' in df.columns:
        df['Type_norm'] = df['Type'].astype(str).str.strip()
    else:
        df['Type_norm'] = df['Price_USD'].apply(lambda x: 'Paid' if x>0 else 'Free')
    # 10) Outlier capping for Reviews, Installs, Price
    df['Reviews_cap'] = cap_outliers_iqr(df['Reviews_num'])
    df['Installs_cap'] = cap_outliers_iqr(df['Installs_num'])
    df['Price_cap'] = cap_outliers_iqr(df['Price_USD'])
    # Save cleaned data
    out = Path('outputs/processed_playstore.csv')
    df.to_csv(out, index=False)
    print('Saved cleaned CSV to', out)
    # ---- Analytical Questions ----
    # Q1 most expensive app
    most_exp = df.loc[df['Price_USD'].idxmax()][['App','Category','Price_USD','Rating','Installs_num']]
    print('\\nQ1 Most expensive app:\\n', most_exp.to_dict())
    # Q2 genre with highest number of apps (explode if needed)
    if df['Genres'].str.contains(';').any():
        gs = df.assign(Genre=df['Genres'].str.split(';')).explode('Genre')
        top_genre = gs['Genre'].str.strip().value_counts().idxmax()
        print('\\nQ2 Top genre:', top_genre)
    else:
        top_genre = df['Genres'].value_counts().idxmax()
        print('\\nQ2 Top genre:', top_genre)
    # Q3 avg size free vs paid
    avg_size = df.groupby('Type_norm')['Size_MB'].mean().to_dict()
    print('\\nQ3 Avg size Free vs Paid:', avg_size)
    # Q4 top5 expensive with rating 5
    top5_perfect = df[df['Rating']>=5.0].sort_values('Price_USD', ascending=False).head(5)[['App','Price_USD','Rating']]
    print('\\nQ4 Top5 perfect rating expensive:\\n', top5_perfect.to_dict(orient='records'))
    # Q5 apps with >50k reviews
    n_50k = int((df['Reviews_num']>50000).sum())
    print('\\nQ5 Apps with >50k reviews:', n_50k)
    # Q6 avg price by genre and installs bucket (sample)
    df_ex = df.assign(Genre=df['Genres'].str.split(';')).explode('Genre')
    bins = [0,1000,10000,100000,1000000,5000000,10000000,1e9]
    labels = ['<1k','1k-10k','10k-100k','100k-1M','1M-5M','5M-10M','10M+']
    df_ex['Installs_bucket'] = pd.cut(df_ex['Installs_num'], bins=bins, labels=labels, include_lowest=True)
    avg_price = df_ex.groupby(['Genre','Installs_bucket'])['Price_USD'].mean().dropna().reset_index().head(20)
    print('\\nQ6 Sample avg price by genre & installs (top 20):\\n', avg_price.to_dict(orient='records'))
    # Q7 rating >4.7 count & avg price
    hr = df[df['Rating']>4.7]
    print('\\nQ7 count rating>4.7:', len(hr), 'avg price:', round(hr['Price_USD'].mean(),2))
    # Q8 Google revenue from apps with >=5,000,000 installs (30% cut)
    apps_5m = df[(df['Installs_num']>=5000000) & (df['Price_USD']>0)].copy()
    apps_5m['Gross'] = apps_5m['Price_USD'] * apps_5m['Installs_num']
    total_gross = apps_5m['Gross'].sum()
    google_cut = total_gross * 0.30
    print('\\nQ8 total gross:', total_gross, 'google_cut(30%):', google_cut)
    # Q9 max/min sizes free vs paid
    sizes = df.groupby('Type_norm')['Size_MB'].agg(['min','max']).to_dict()
    print('\\nQ9 sizes min/max by type:', sizes)
    # Q10 correlation
    corr = df[['Rating','Reviews_num','Size_MB','Price_USD']].corr()
    print('\\nQ10 correlation matrix:\\n', corr.to_dict())
    # Q11 count by Type and Content Rating
    if 'Content Rating' in df.columns:
        cross = df.groupby(['Content Rating','Type_norm']).size().unstack(fill_value=0)
        print('\\nQ11 content rating x type sample:\\n', cross.head().to_dict())
    # Q12 Android 4.x compatibility
    n_android_4x = int(df[df['Android_min'] < 5].shape[0]) if 'Android_min' in df.columns else 0
    print('\\nQ12 Android 4.x compatible count:', n_android_4x)
    # Save a few plots
    try:
        plt.figure(figsize=(6,4))
        df['Category'].value_counts().head(10).plot(kind='bar', title='Top 10 Categories')
        plt.tight_layout()
        plt.savefig('outputs/top10_categories.png')
        plt.close()
    except Exception as e:
        print('Plot error:', e)

if __name__ == "__main__":
    main()
