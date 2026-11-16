"""
build_data_assets.py
====================
Processes raw police datasets and generates:
1. web/assets/data.json
2. data/processed/data.json
3. web/assets/embedded_data.js (for direct file:// browser support)
"""

import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
WEB_ASSETS_DIR = os.path.join(BASE_DIR, "web", "assets")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(WEB_ASSETS_DIR, exist_ok=True)

# 1. Process Models Dataset (2017-2023)
models_path = os.path.join(RAW_DIR, "stolen_cars_models_2017_2023.csv")
df_models_raw = pd.read_csv(models_path, encoding='utf-8')

# First row is Total summary, skip it for models
df_models = df_models_raw.iloc[1:].copy()

# Hebrew model names mapping for intuitive Hebrew search
HEBREW_MODEL_NAMES = {
    'COROLLA': 'קורולה',
    'TUCSON': 'טוסון',
    'SPORTAGE': "ספורטאז'",
    'PICANTO': 'פיקנטו',
    'OCTAVIA': 'אוקטביה',
    'IONIQ': 'איוניק',
    'IONIQ HYBRID': 'איוניק',
    'RAV 4': 'ראב 4',
    'LAND CRUISER': 'לנד קרוזר',
    'OUTLANDER': 'אאוטלנדר',
    'BERLINGO': 'ברלינגו',
    'C-HR': 'C-HR',
    'NIRO': 'נירו',
    'YARIS': 'יאריס',
    'AVENSIS': 'אוונסיס',
    'HILUX': 'היילקס',
    'IBIZA': 'איביזה',
    'LEON': 'לאון',
    'SUPERB': 'סופרב',
    'FABIA': 'פאביה',
    'KODIAQ': 'קודיאק',
    'RAPID': 'ראפיד',
    'GOLF': 'גולף',
    'POLO': 'פולו',
    'PASSAT': 'פאסאט',
    'JETTA': "ג'טה",
    'FOCUS': 'פוקוס',
    'FLUENCE': 'פלואנס',
    'MEGAN': 'מגאן',
    'CIVIC': 'סיוויק',
    'ACCORD': 'אקורד',
    'LANCER': 'לנסר',
    'PAJERO': "פאג'רו",
    'L200': 'L200 (האנטר)',
    'SANTA FE': 'סנטה פה',
    'KONA': 'קונה',
    'VENUE': 'וניו',
    'ELANTRA': 'אלנטרה',
    'SONATA': 'סונטה',
    'GETZ': 'גטס',
    'ACCENT': 'אקסנט',
    'SORENTO': 'סורנטו',
    'FORTE': 'פורטה',
    'RIO': 'ריו',
    'STONIC': 'סטוניק',
    'CEED': 'סיד',
    'CRUZE': 'קרוז',
    'MALIBU': 'מאליבו',
    'AVEO': 'אוואו',
    'ASTRA': 'אסטרה',
    'CORSA': 'קורסה',
    'IMPREZA': 'אימפרזה',
    'B4': 'B4 (לגאסי)',
    'QASHQAI': 'קשקאי',
    'JUKE': "ג'וק",
    'X-TRAIL': 'אקס טרייל',
    'PARTNER': 'פרטנר',
    '3008308': '3008 / 308',
    'PICK UP': 'פיק אפ (D-Max)',
    'IPPON': 'איפון',
    'LANTIS 323': 'לאנטיס 323',
    'CX': 'CX (CX-5 / CX-3)',
}

def clean_brand(maker_str):
    m = str(maker_str).strip()
    m = m.replace('-', ' ').replace('(', ' ').replace(')', ' ').replace('"', '').replace("'", '')
    if m.startswith('סאן יאנג'):
        return 'סאן יאנג'
    if m.startswith('קואנג יאנג'):
        return 'קואנג יאנג (קימקו)'
    if m.startswith('קי טי') or m.startswith('קי.טי') or m == 'קי':
        return 'KTM'
    parts = m.split()
    if not parts:
        return None
    first = parts[0]
    if first in ['Total', 'Fields', 'nan', 'סוג', 'שנת', 'מרכז', 'מרכבי']:
        return None
    brand_map = {
        'מזדה': 'מאזדה',
        'פיגו': "פיג'ו",
        'פיאגיו': "פיאג'ו",
    }
    return brand_map.get(first, first)

def clean_model_name(model_str):
    m = str(model_str).strip()
    if m in ['nan', '-', '']:
        return ''
    if m == '3008308':
        return '3008 / 308'
    if m == 'IONIQ HYBRID':
        return 'IONIQ'
    return m

models_dict = {}
years = ['2017', '2018', '2019', '2020', '2021', '2022', '2023']

for _, row in df_models.iterrows():
    raw_maker = row.iloc[0]
    brand = clean_brand(raw_maker)
    if not brand:
        continue
        
    model_name = clean_model_name(row.iloc[1])
    category = str(row.iloc[2]).strip()
    
    # Construct clean, unified display key
    if not model_name:
        if category in ['דו/תלת גלגלי']:
            full_name = f"{brand} (דו-גלגלי)"
            heb_name = ""
        elif category in ['משאית', 'מסחרי', 'טנדר', 'טרקטור']:
            full_name = f"{brand} ({category})"
            heb_name = ""
        else:
            full_name = f"{brand} (כללי)"
            heb_name = ""
    else:
        heb_name = HEBREW_MODEL_NAMES.get(model_name, "")
        if heb_name:
            full_name = f"{brand} {heb_name} ({model_name})"
        else:
            full_name = f"{brand} {model_name}"
    
    # Clean yearly counts
    yearly = {}
    for y in years:
        val = str(row.get(y, '0')).replace(',', '').strip()
        yearly[int(y)] = int(val) if val.isdigit() else 0
        
    total_val = str(row.get('Total', '0')).replace(',', '').strip()
    total_thefts = int(total_val) if total_val.isdigit() else sum(yearly.values())
    
    # Aggregate if model already exists
    if full_name in models_dict:
        for y in years:
            models_dict[full_name]['yearly_data'][int(y)] += yearly[int(y)]
        models_dict[full_name]['total_thefts'] += total_thefts
    else:
        models_dict[full_name] = {
            'maker': brand,
            'model_name': model_name if model_name else brand,
            'name_hebrew': heb_name,
            'display_name': full_name,
            'category': category,
            'total_thefts': total_thefts,
            'yearly_data': yearly
        }

# Post-compute growth rate & trend on aggregated yearly data
for m_name, m_data in models_dict.items():
    yd = m_data['yearly_data']
    recent_avg = (yd[2022] + yd[2023]) / 2.0
    early_avg = (yd[2017] + yd[2018]) / 2.0 if (yd[2017] + yd[2018]) > 0 else 1.0
    growth = recent_avg / early_avg if early_avg > 0 else 1.0
    
    if growth >= 1.4:
        trend = "rising"
    elif growth <= 0.75:
        trend = "declining"
    else:
        trend = "stable"
        
    m_data['trend'] = trend
    m_data['growth_rate'] = round(growth, 2)


# Rank models and compute normalized risk score (0-100)
sorted_models = sorted(models_dict.items(), key=lambda x: x[1]['total_thefts'], reverse=True)
max_thefts = sorted_models[0][1]['total_thefts'] if sorted_models else 1

for rank, (name, data) in enumerate(sorted_models, 1):
    # Risk score based on theft volume, recent trajectory, and growth
    theft_factor = (data['total_thefts'] / max_thefts) * 70.0
    growth_bonus = 20.0 if data['trend'] == 'rising' else (10.0 if data['trend'] == 'stable' else 0.0)
    recent_volume = (data['yearly_data'][2023] / (max_thefts / 7)) * 10.0
    
    score = min(99, max(15, round(theft_factor + growth_bonus + recent_volume)))
    
    if score >= 75:
        cat = "Very High"
    elif score >= 50:
        cat = "High"
    elif score >= 30:
        cat = "Medium"
    else:
        cat = "Low"
        
    data['rank'] = rank
    data['risk_score'] = score
    data['risk_category'] = cat
    data['theft_rate'] = round(data['total_thefts'] / 84997, 4)

models_processed = {k: v for k, v in sorted_models}

# 2. Process Cities Dataset (2018-2023)
area_path = os.path.join(RAW_DIR, "stolen_cars_by_area.csv")
df_area_raw = pd.read_csv(area_path, encoding='utf-8')

city_col = df_area_raw.columns[2]      # שם ישוב עבירה
year_col = df_area_raw.columns[1]      # שנה
cases_col = df_area_raw.columns[3]     # תיקים – נפתחו סה"כ
suspects_col = df_area_raw.columns[4]  # תיקים עם חשודים
indict_col = df_area_raw.columns[10]   # תיקים עם כתב אישום – סה"כ

# Clean columns
def clean_num(s):
    v = str(s).replace(',', '').strip()
    return int(v) if v.isdigit() else 0

df_area_raw['cases_clean'] = df_area_raw[cases_col].apply(clean_num)
df_area_raw['suspects_clean'] = df_area_raw[suspects_col].apply(clean_num)
df_area_raw['indict_clean'] = df_area_raw[indict_col].apply(clean_num)

# Filter valid cities
valid_mask = (
    df_area_raw[city_col].notna() & 
    ~df_area_raw[city_col].isin(['סה"כ', 'לא ידוע', 'כל הארץ']) &
    ~df_area_raw[year_col].isin(['סה"כ'])
)
df_valid_areas = df_area_raw[valid_mask].copy()

# Group by city
locations_dict = {}
city_groups = df_valid_areas.groupby(city_col)

for city_name, group in city_groups:
    tot_cases = group['cases_clean'].sum()
    if tot_cases < 5:  # skip extremely sparse hamlets
        continue
        
    tot_suspects = group['suspects_clean'].sum()
    tot_indict = group['indict_clean'].sum()
    
    detection_rate = round(tot_suspects / tot_cases, 3) if tot_cases > 0 else 0
    indictment_rate = round(tot_indict / tot_cases, 3) if tot_cases > 0 else 0
    
    # Yearly breakdown (2018-2023)
    yearly = {}
    for y in range(2018, 2024):
        y_group = group[group[year_col].astype(str).str.contains(str(y))]
        yearly[y] = int(y_group['cases_clean'].sum())
        
    y22_23 = (yearly.get(2022, 0) + yearly.get(2023, 0)) / 2.0
    y18_19 = (yearly.get(2018, 0) + yearly.get(2019, 0)) / 2.0 if (yearly.get(2018, 0) + yearly.get(2019, 0)) > 0 else 1.0
    city_growth = y22_23 / y18_19 if y18_19 > 0 else 1.0
    
    if city_growth >= 1.3:
        trend = "rising"
    elif city_growth <= 0.8:
        trend = "declining"
    else:
        trend = "stable"
        
    locations_dict[city_name] = {
        'total_thefts': int(tot_cases),
        'detection_rate': detection_rate,
        'indictment_rate': indictment_rate,
        'trend': trend,
        'growth_rate': round(city_growth, 2),
        'yearly_data': yearly
    }

# Rank cities
sorted_cities = sorted(locations_dict.items(), key=lambda x: x[1]['total_thefts'], reverse=True)
max_city_thefts = sorted_cities[0][1]['total_thefts'] if sorted_cities else 1

for rank, (c_name, c_data) in enumerate(sorted_cities, 1):
    c_data['rank'] = rank
    rel_vol = c_data['total_thefts'] / max_city_thefts
    if rel_vol >= 0.35:
        c_data['risk_cluster'] = "Very High"
        c_data['risk_score'] = min(98, round(75 + rel_vol * 23))
    elif rel_vol >= 0.15:
        c_data['risk_cluster'] = "High"
        c_data['risk_score'] = round(55 + rel_vol * 100)
    elif rel_vol >= 0.05:
        c_data['risk_cluster'] = "Medium"
        c_data['risk_score'] = round(35 + rel_vol * 120)
    else:
        c_data['risk_cluster'] = "Low"
        c_data['risk_score'] = max(10, round(15 + rel_vol * 200))

locations_processed = {k: v for k, v in sorted_cities}

# 3. Process 2023-2025 Police Offences Dataset
crimes_path = os.path.join(RAW_DIR, "stolen_cars_crimes_2023_2025.csv")
df_crimes_raw = pd.read_csv(crimes_path, encoding='utf-8')
crimes_processed = []

for _, row in df_crimes_raw.iloc[2:16].iterrows():
    c_name = str(row['תאור עבירה-מלא']).strip()
    tot = clean_num(row.get('Total', 0))
    y23 = clean_num(row.get('2023', 0))
    y24 = clean_num(row.get('2024', 0))
    y25 = clean_num(row.get('2025', 0))
    if tot > 0:
        crimes_processed.append({
            'offense_name': c_name,
            'total_2023_2025': tot,
            '2023': y23,
            '2024': y24,
            '2025': y25
        })

# 4. Process Tab 2 (Population / Suspect sectors)
tab2_path = os.path.join(RAW_DIR, "stolen_cars_by_area_tab2.csv")
df_tab2_raw = pd.read_csv(tab2_path, encoding='utf-8')
sector_col = df_tab2_raw.columns[3]  # ישראלי/פלסטיני/זרים
cases_col2 = df_tab2_raw.columns[4]

df_tab2_raw['cases_clean'] = df_tab2_raw[cases_col2].apply(clean_num)
valid_tab2 = df_tab2_raw[df_tab2_raw[sector_col].notna() & ~df_tab2_raw[sector_col].isin(['סה"כ'])].copy()
sector_stats = valid_tab2.groupby(sector_col)['cases_clean'].sum().to_dict()

# 5. Create final comprehensive payload
final_data = {
    'metadata': {
        'source': 'משטרת ישראל - אגף חקירות ומודיעין',
        'years_covered': '2017-2025',
        'models_count': len(models_processed),
        'cities_count': len(locations_processed),
        'total_recorded_thefts_2017_2023': 84997,
        'total_thefts_2023_2025': 59056,
        'generated_at': pd.Timestamp.now().isoformat()
    },
    'models': models_processed,
    'locations': locations_processed,
    'crimes_2023_2025': crimes_processed,
    'demographics_summary': sector_stats
}

# Export JSON
json_out1 = os.path.join(PROCESSED_DIR, "data.json")
json_out2 = os.path.join(WEB_ASSETS_DIR, "data.json")

with open(json_out1, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

with open(json_out2, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

# Also create embedded_data.js for standalone browser support (CORS-free)
js_out = os.path.join(WEB_ASSETS_DIR, "embedded_data.js")
with open(js_out, 'w', encoding='utf-8') as f:
    f.write("window.EMBEDDED_DATA = ")
    json.dump(final_data, f, ensure_ascii=False)
    f.write(";\n")

print(f"Generated {json_out1} ({os.path.getsize(json_out1):,} bytes)")
print(f"Generated {json_out2} ({os.path.getsize(json_out2):,} bytes)")
print(f"Generated {js_out} ({os.path.getsize(js_out):,} bytes)")
print(f"Processed {len(models_processed)} car models, {len(locations_processed)} cities, {len(crimes_processed)} crime types.")
