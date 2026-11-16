"""
feature_engineering.py - שלב 3: הנדסת פיצ'רים
==============================================

מודול זה יוצר פיצ'רים חדשים מהנתונים הנקיים:
- פיצ'רים מבוססי זמן (temporal)
- פיצ'רים מבוססי רכב (vehicle)
- פיצ'רים מבוססי מיקום (location)
- פיצ'רים משולבים (interaction)
- ציון סיכון (risk score)

Usage:
    python src/feature_engineering.py
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


# ═══════════════════════════════════════════════════════════════════
# 3.1 פיצ'רים מבוססי זמן (Temporal Features)
# ═══════════════════════════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame, 
                          date_col: str = 'theft_date',
                          manufacture_year_col: str = 'vehicle_year') -> pd.DataFrame:
    """
    מוסיפה פיצ'רים מבוססי זמן.
    
    Features created:
        - theft_year, theft_month, theft_day_of_week, theft_quarter
        - theft_season (קיץ/חורף/אביב/סתיו)
        - vehicle_age (גיל הרכב בזמן הגניבה)
    
    Args:
        df: DataFrame נקי
        date_col: שם עמודת תאריך הגניבה
        manufacture_year_col: שם עמודת שנת ייצור
        
    Returns:
        pd.DataFrame: עם פיצ'רים חדשים
    """
    df = df.copy()
    
    # המרה ל-datetime (אם לא כבר)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        df['theft_year'] = df[date_col].dt.year
        df['theft_month'] = df[date_col].dt.month
        df['theft_day_of_week'] = df[date_col].dt.dayofweek  # 0=Monday
        df['theft_quarter'] = df[date_col].dt.quarter
        
        # עונה
        season_map = {
            12: 'חורף', 1: 'חורף', 2: 'חורף',
            3: 'אביב', 4: 'אביב', 5: 'אביב',
            6: 'קיץ', 7: 'קיץ', 8: 'קיץ',
            9: 'סתיו', 10: 'סתיו', 11: 'סתיו'
        }
        df['theft_season'] = df['theft_month'].map(season_map)
        
        print("   ✅ Added temporal features: year, month, day_of_week, quarter, season")
    
    # גיל הרכב
    if manufacture_year_col in df.columns and 'theft_year' in df.columns:
        df['vehicle_age'] = df['theft_year'] - df[manufacture_year_col]
        # סינון ערכים לא הגיוניים
        df.loc[df['vehicle_age'] < 0, 'vehicle_age'] = np.nan
        df.loc[df['vehicle_age'] > 50, 'vehicle_age'] = np.nan
        print("   ✅ Added vehicle_age")
    
    return df


# ═══════════════════════════════════════════════════════════════════
# 3.2 פיצ'רים מבוססי דגם (Vehicle Features)
# ═══════════════════════════════════════════════════════════════════

def add_vehicle_features(df: pd.DataFrame,
                         model_col: str = 'vehicle_model') -> pd.DataFrame:
    """
    מוסיפה פיצ'רים מבוססי דגם רכב.
    
    Features created:
        - model_theft_count: כמות גניבות לדגם
        - model_theft_rate: שיעור גניבה יחסי
        - model_rank: דירוג הדגם (מ-1 כלפי מטה)
        - model_risk_category: קטגוריית סיכון (Low/Medium/High/Very High)
        - model_trend: מגמת גניבה (rising/stable/declining)
    """
    df = df.copy()
    
    if model_col not in df.columns:
        print(f"   ⚠️  Column '{model_col}' not found!")
        return df
    
    # ─── כמות גניבות לפי דגם ───
    theft_counts = df[model_col].value_counts()
    df['model_theft_count'] = df[model_col].map(theft_counts)
    
    # ─── שיעור גניבה יחסי (% מסה"כ) ───
    total = len(df)
    df['model_theft_rate'] = df['model_theft_count'] / total
    
    # ─── דירוג ───
    rank_map = {model: rank for rank, model in enumerate(theft_counts.index, 1)}
    df['model_rank'] = df[model_col].map(rank_map)
    
    # ─── קטגוריית סיכון ───
    percentiles = df['model_theft_count'].quantile([0.5, 0.75, 0.9])
    def categorize_risk(count):
        if count >= percentiles[0.9]:
            return 'Very High'
        elif count >= percentiles[0.75]:
            return 'High'
        elif count >= percentiles[0.5]:
            return 'Medium'
        else:
            return 'Low'
    
    df['model_risk_category'] = df['model_theft_count'].apply(categorize_risk)
    
    # ─── מגמת גניבה לפי שנים ───
    if 'theft_year' in df.columns:
        yearly_counts = df.groupby([model_col, 'theft_year']).size().reset_index(name='count')
        
        trends = {}
        for model in df[model_col].unique():
            model_data = yearly_counts[yearly_counts[model_col] == model].sort_values('theft_year')
            if len(model_data) >= 3:
                slope, _, _, _, _ = stats.linregress(
                    model_data['theft_year'], model_data['count']
                )
                if slope > 1:
                    trends[model] = 'rising'
                elif slope < -1:
                    trends[model] = 'declining'
                else:
                    trends[model] = 'stable'
            else:
                trends[model] = 'insufficient_data'
        
        df['model_trend'] = df[model_col].map(trends)
        print("   ✅ Added model trend (rising/stable/declining)")
    
    print("   ✅ Added vehicle features: theft_count, theft_rate, rank, risk_category")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3.3 פיצ'רים מבוססי מיקום (Location Features)
# ═══════════════════════════════════════════════════════════════════

def add_location_features(df: pd.DataFrame,
                          city_col: str = 'city') -> pd.DataFrame:
    """
    מוסיפה פיצ'רים מבוססי מיקום.
    
    Features created:
        - city_theft_count: כמות גניבות ביישוב
        - city_theft_rate: שיעור גניבה יחסי
        - city_risk_cluster: קלאסטר סיכון
        - city_trend: מגמת גניבה ביישוב
    """
    df = df.copy()
    
    if city_col not in df.columns:
        print(f"   ⚠️  Column '{city_col}' not found!")
        return df
    
    # ─── כמות גניבות לפי יישוב ───
    city_counts = df[city_col].value_counts()
    df['city_theft_count'] = df[city_col].map(city_counts)
    
    # ─── שיעור גניבה יחסי ───
    total = len(df)
    df['city_theft_rate'] = df['city_theft_count'] / total
    
    # ─── קלאסטר סיכון ───
    percentiles = df.groupby(city_col).size().quantile([0.5, 0.75, 0.9])
    def city_risk(count):
        if count >= percentiles[0.9]:
            return 'Very High'
        elif count >= percentiles[0.75]:
            return 'High'
        elif count >= percentiles[0.5]:
            return 'Medium'
        else:
            return 'Low'
    
    city_risk_map = {city: city_risk(count) for city, count in city_counts.items()}
    df['city_risk_cluster'] = df[city_col].map(city_risk_map)
    
    # ─── מגמה ───
    if 'theft_year' in df.columns:
        yearly_city = df.groupby([city_col, 'theft_year']).size().reset_index(name='count')
        
        city_trends = {}
        for city in df[city_col].unique():
            city_data = yearly_city[yearly_city[city_col] == city].sort_values('theft_year')
            if len(city_data) >= 3:
                slope, _, _, _, _ = stats.linregress(
                    city_data['theft_year'], city_data['count']
                )
                if slope > 1:
                    city_trends[city] = 'rising'
                elif slope < -1:
                    city_trends[city] = 'declining'
                else:
                    city_trends[city] = 'stable'
            else:
                city_trends[city] = 'insufficient_data'
        
        df['city_trend'] = df[city_col].map(city_trends)
        print("   ✅ Added city trend")
    
    print("   ✅ Added location features: theft_count, theft_rate, risk_cluster")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3.4 פיצ'רים משולבים (Interaction Features)
# ═══════════════════════════════════════════════════════════════════

def add_interaction_features(df: pd.DataFrame,
                             model_col: str = 'vehicle_model',
                             city_col: str = 'city') -> pd.DataFrame:
    """
    מוסיפה פיצ'רים משולבים (model × location).
    
    Features created:
        - model_city_count: כמות גניבות לשילוב דגם+יישוב
        - model_city_rate: שיעור גניבה ספציפי
    """
    df = df.copy()
    
    if model_col in df.columns and city_col in df.columns:
        # שילוב דגם × יישוב
        combo_counts = df.groupby([model_col, city_col]).size()
        df['model_city_key'] = df[model_col] + '_' + df[city_col]
        key_counts = combo_counts.to_dict()
        df['model_city_count'] = df.apply(
            lambda row: key_counts.get((row[model_col], row[city_col]), 0), axis=1
        )
        df['model_city_rate'] = df['model_city_count'] / len(df)
        df.drop('model_city_key', axis=1, inplace=True)
        
        print("   ✅ Added interaction features: model_city_count, model_city_rate")
    
    return df


# ═══════════════════════════════════════════════════════════════════
# 3.5 ציון סיכון (Risk Score)
# ═══════════════════════════════════════════════════════════════════

def calculate_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    מחשבת ציון סיכון מורכב (0-100).
    
    הנוסחה:
        risk_score = w1 * model_risk + w2 * city_risk + w3 * trend_bonus
        
    כאשר:
        - model_risk: normalized model_theft_rate (0-1)
        - city_risk: normalized city_theft_rate (0-1)
        - trend_bonus: +0.1 if rising, 0 if stable, -0.1 if declining
    """
    df = df.copy()
    
    scaler = MinMaxScaler()
    
    # נרמול שיעורי גניבה
    components = []
    
    if 'model_theft_rate' in df.columns:
        df['model_risk_norm'] = scaler.fit_transform(df[['model_theft_rate']])
        components.append(('model_risk_norm', 0.4))
    
    if 'city_theft_rate' in df.columns:
        df['city_risk_norm'] = scaler.fit_transform(df[['city_theft_rate']])
        components.append(('city_risk_norm', 0.35))
    
    if 'model_city_count' in df.columns:
        df['combo_risk_norm'] = scaler.fit_transform(df[['model_city_count']])
        components.append(('combo_risk_norm', 0.15))
    
    # בונוס מגמה
    if 'model_trend' in df.columns:
        trend_map = {'rising': 0.1, 'stable': 0, 'declining': -0.05, 'insufficient_data': 0}
        df['trend_bonus'] = df['model_trend'].map(trend_map).fillna(0)
        components.append(('trend_bonus', 0.1))
    
    # חישוב ציון
    if components:
        df['risk_score'] = sum(df[col] * weight for col, weight in components)
        df['risk_score'] = (df['risk_score'] * 100).clip(0, 100).round(1)
        print("   ✅ Added risk_score (0-100)")
    
    # ניקוי עמודות זמניות
    for col in ['model_risk_norm', 'city_risk_norm', 'combo_risk_norm', 'trend_bonus']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return df


# ═══════════════════════════════════════════════════════════════════
# Pipeline ראשי
# ═══════════════════════════════════════════════════════════════════

def run_feature_engineering():
    """
    מריצה את pipeline הנדסת הפיצ'רים.
    
    ⚠️ יש להתאים את שמות העמודות!
    """
    print("🚀 Starting Feature Engineering Pipeline")
    print("=" * 60)
    
    # ─── Load cleaned data ───
    input_path = os.path.join(PROCESSED_DATA_DIR, 'merged_clean.csv')
    
    if not os.path.exists(input_path):
        print(f"⚠️  File not found: {input_path}")
        print("   Please run data_processing.py first!")
        return
    
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"📂 Loaded {len(df):,} rows from merged_clean.csv")
    
    # ─── TODO: Update column names to match your data ───
    DATE_COL = 'theft_date'        # עמודת תאריך גניבה
    YEAR_COL = 'vehicle_year'      # עמודת שנת ייצור
    MODEL_COL = 'vehicle_model'    # עמודת דגם
    CITY_COL = 'city'              # עמודת יישוב
    
    # ─── Apply features ───
    print("\n📐 Step 3.1: Temporal Features")
    df = add_temporal_features(df, DATE_COL, YEAR_COL)
    
    print("\n🚗 Step 3.2: Vehicle Features")
    df = add_vehicle_features(df, MODEL_COL)
    
    print("\n📍 Step 3.3: Location Features")
    df = add_location_features(df, CITY_COL)
    
    print("\n🔗 Step 3.4: Interaction Features")
    df = add_interaction_features(df, MODEL_COL, CITY_COL)
    
    print("\n📊 Step 3.5: Risk Score")
    df = calculate_risk_score(df)
    
    # ─── Save ───
    output_path = os.path.join(PROCESSED_DATA_DIR, 'features.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved features to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   New columns: {[c for c in df.columns if c not in pd.read_csv(input_path, nrows=0).columns]}")


if __name__ == '__main__':
    run_feature_engineering()
