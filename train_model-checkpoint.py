"""
=============================================================================
IPO Hype vs Reality — Model Training Script
=============================================================================
Run this script ONCE to train the ML model and save the .pkl files.
The Flask app (app.py) loads these .pkl files to make predictions.

Usage:
    python train_model.py

Output files generated:
    - best_model.pkl       : Trained Random Forest classifier
    - scaler.pkl           : Fitted StandardScaler
    - feature_names.pkl    : List of feature column names
=============================================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  IPO Hype vs Reality - Model Training Script")
print("=" * 60)

# ── STEP 1: Install / import required libraries ───────────────
print("\n[1/8] Loading libraries...")

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
except ImportError:
    print("  Installing nltk...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk", "-q"])
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
except ImportError:
    print("  Installing scikit-learn...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "-q"])
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report

print("  Libraries loaded successfully.")

# ── STEP 2: Load datasets ─────────────────────────────────────
print("\n[2/8] Loading datasets...")

ipo_df  = pd.read_csv('ipo_data.csv')
news_df = pd.read_csv('news_data.csv')

# Strip whitespace from column names
ipo_df.columns  = ipo_df.columns.str.strip()
news_df.columns = news_df.columns.str.strip()

# Drop empty rows
ipo_df  = ipo_df.dropna(subset=['company'])
news_df = news_df.dropna(subset=['company'])

print(f"  IPO data : {ipo_df.shape[0]} rows, {ipo_df.shape[1]} columns")
print(f"  News data: {news_df.shape[0]} rows, {news_df.shape[1]} columns")

# ── STEP 3: Preprocess IPO data ───────────────────────────────
print("\n[3/8] Preprocessing IPO data...")

# Convert date
ipo_df['ipo_date'] = pd.to_datetime(ipo_df['ipo_date'], errors='coerce')
ipo_df['ipo_year']  = ipo_df['ipo_date'].dt.year.fillna(2023).astype(int)
ipo_df['ipo_month'] = ipo_df['ipo_date'].dt.month.fillna(1).astype(int)

# Handle missing values
ipo_df['issue_price']     = pd.to_numeric(ipo_df['issue_price'],     errors='coerce').fillna(ipo_df['issue_price'].median())
ipo_df['listing_price']   = pd.to_numeric(ipo_df['listing_price'],   errors='coerce').fillna(ipo_df['listing_price'].median())
ipo_df['current_price']   = pd.to_numeric(ipo_df['current_price'],   errors='coerce').fillna(ipo_df['current_price'].median())
ipo_df['subscribed_times']= pd.to_numeric(ipo_df['subscribed_times'], errors='coerce').fillna(ipo_df['subscribed_times'].median())
ipo_df['sector']          = ipo_df['sector'].fillna('Unknown')

# Remove duplicates
ipo_df = ipo_df.drop_duplicates(subset=['company'])

# ── STEP 4: Create target variable ───────────────────────────
print("\n[4/8] Creating target variable (ipo_outcome)...")

ipo_df['listing_gain_pct']  = (ipo_df['listing_price']  - ipo_df['issue_price']) / ipo_df['issue_price'] * 100
ipo_df['longterm_gain_pct'] = (ipo_df['current_price']  - ipo_df['issue_price']) / ipo_df['issue_price'] * 100

def classify_ipo(row):
    lg  = row['listing_gain_pct']
    ltg = row['longterm_gain_pct']
    if lg > 0 and ltg > 20:
        return 'Successful IPO'
    elif lg > 20 and ltg < 0:
        return 'Overhyped IPO'
    else:
        return 'Underperforming IPO'

ipo_df['ipo_outcome'] = ipo_df.apply(classify_ipo, axis=1)

dist = ipo_df['ipo_outcome'].value_counts()
for label, count in dist.items():
    print(f"  {label}: {count} IPOs")

# ── STEP 5: Sentiment analysis on news ───────────────────────
print("\n[5/8] Running VADER sentiment analysis on news headlines...")

sia = SentimentIntensityAnalyzer()

def get_compound(text):
    try:
        return sia.polarity_scores(str(text))['compound']
    except:
        return 0.0

news_df['sentiment'] = news_df['headline'].apply(get_compound)

# Aggregate per company
news_agg = news_df.groupby('company').agg(
    avg_sentiment  = ('sentiment', 'mean'),
    news_frequency = ('sentiment', 'count')
).reset_index()

print(f"  Sentiment computed for {len(news_agg)} companies.")

# ── STEP 6: Merge & Feature Engineering ──────────────────────
print("\n[6/8] Merging datasets and engineering features...")

df = ipo_df.merge(news_agg, on='company', how='left')
df['avg_sentiment']  = df['avg_sentiment'].fillna(0.0)
df['news_frequency'] = df['news_frequency'].fillna(3).astype(int)

# sector_popularity: count of IPOs in same sector
sector_counts = df['sector'].value_counts().to_dict()
df['sector_popularity'] = df['sector'].map(sector_counts)

# hype_score: composite metric
df['hype_score'] = (
    (df['subscribed_times'] / df['subscribed_times'].max()) +
    ((df['avg_sentiment'] + 1) / 2) +
    (df['news_frequency'] / df['news_frequency'].max())
) / 3

# ipo_age_days: days since IPO date
today = pd.Timestamp.now()
df['ipo_age_days'] = (today - df['ipo_date']).dt.days.fillna(365).astype(int)

print(f"  Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── STEP 7: Encode & Scale ────────────────────────────────────
print("\n[7/8] Encoding and scaling features...")

# One-hot encode sector
df_encoded = pd.get_dummies(df, columns=['sector'], prefix='sector')

# Label encode target
le = LabelEncoder()
df_encoded['ipo_outcome_encoded'] = le.fit_transform(df_encoded['ipo_outcome'])

print(f"  Label classes: {list(le.classes_)}")
print(f"  Encoded as   : {list(range(len(le.classes_)))}")

# Define features (drop leaky and non-feature columns)
drop_cols = [
    'company', 'ipo_date', 'listing_price', 'current_price',
    'listing_gain_pct', 'longterm_gain_pct', 'ipo_outcome',
    'ipo_outcome_encoded', 'ipo_year', 'ipo_month'
]
feature_cols = [c for c in df_encoded.columns if c not in drop_cols]

X = df_encoded[feature_cols]
y = df_encoded['ipo_outcome_encoded']

print(f"  Features ({len(feature_cols)}): {feature_cols}")
print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (use all data if too small for stratify)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
except ValueError:
    # If stratify fails (class with 1 sample), split without stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

# ── STEP 8: Train & Save Best Model ──────────────────────────
print("\n[8/8] Training models and selecting best...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}

best_model  = None
best_name   = ''
best_acc    = 0.0
results     = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': round(acc * 100, 2)})
    print(f"  {name:25s} -> Accuracy: {acc*100:.1f}%")
    if acc > best_acc:
        best_acc   = acc
        best_model = model
        best_name  = name

print(f"\n  Best model: {best_name} ({best_acc*100:.1f}%)")

# Hyperparameter tuning on Random Forest
print("\n  Tuning Random Forest with GridSearchCV (cv=3)...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [3, 5, None],
    'min_samples_split': [2, 5],
}
rf_tuned = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=3, scoring='accuracy', n_jobs=-1
)
rf_tuned.fit(X_scaled, y)  # fit on full data for small dataset

tuned_acc = rf_tuned.best_score_
print(f"  Tuned RF best CV accuracy: {tuned_acc*100:.1f}%")
print(f"  Best params: {rf_tuned.best_params_}")

# Pick tuned RF if it's better
if tuned_acc >= best_acc:
    best_model = rf_tuned.best_estimator_
    best_name  = 'Tuned Random Forest'
    print(f"  Using tuned model as final model.")
else:
    print(f"  Keeping {best_name} as final model.")

# ── Save artifacts ────────────────────────────────────────────
print("\n  Saving model artifacts...")

joblib.dump(best_model,   'best_model.pkl')
joblib.dump(scaler,       'scaler.pkl')
joblib.dump(feature_cols, 'feature_names.pkl')

print("  best_model.pkl    -> saved")
print("  scaler.pkl        -> saved")
print("  feature_names.pkl -> saved")

# Verify files were created
import os
for fname in ['best_model.pkl', 'scaler.pkl', 'feature_names.pkl']:
    size = os.path.getsize(fname)
    print(f"  {fname:25s} -> {size:,} bytes")

print("\n" + "=" * 60)
print("  DONE! Model training complete.")
print("  Now run:  python app.py")
print("  Then open: http://localhost:5000")
print("=" * 60)
