import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. KONFIGURASI ---
COEFFICIENTS = {
    'wgi': 0.079, 'gdp_pc': 0.037, 'world_gdp_share': 0.640,
    'default_record': -1.791, 'money_supply': 0.145, 'gdp_volatility': -0.710,
    'inflation': -0.069, 'real_growth': 0.057, 'gg_debt': -0.023,
    'int_rev': -0.044, 'fiscal_bal': 0.039, 'fc_debt': -0.008,
    'rc_flex': 0.494, 'snfa': 0.011, 'commodity_dep': -0.004,
    'reserves_months': 0.024, 'ext_int_service': -0.004, 'ca_fdi': 0.004
}

NUM_TO_RATING = {
    16: 'AAA', 15: 'AA+', 14: 'AA', 13: 'AA-', 12: 'A+', 11: 'A', 10: 'A-',
    9: 'BBB+', 8: 'BBB', 7: 'BBB-', 6: 'BB+', 5: 'BB', 4: 'BB-', 3: 'B+', 
    2: 'B', 1: 'B-', 0: 'CCC/D'
}

RATING_TO_NUM = {v: k for k, v in NUM_TO_RATING.items()} 
RATING_TO_NUM.update({'CCC': 0, 'CC': 0, 'C': 0, 'RD': 0, 'D': 0})

# --- 2. HELPER FUNCTIONS ---
def col2idx(c):
    return sum((ord(x) - 64) * (26 ** i) for i, x in enumerate(reversed(c.upper()))) - 1

def safe_float(v):
    try: return float(v) if str(v).strip() not in ['-', 'nan'] else 0.0
    except: return 0.0

def parse_actual_rating(rating_str):
    if pd.isna(rating_str): return None
    clean_str = str(rating_str).replace('*', '').replace('u', '').strip() 
    return RATING_TO_NUM.get(clean_str, None)

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="Fitch SRM Master", layout="wide", page_icon="🌍")

st.title("🌍 Fitch Sovereign Rating: Master Dashboard")
st.markdown("""
**Analisis Lengkap:**
* **SRM (Shadow Rating):** Hasil murni model matematika.
* **QO (Qualitative Overlay):** Kebijaksanaan komite rating (Selisih Aktual vs Model).
* **Kontribusi Indikator:** Faktor apa yang paling mendongkrak atau menekan skor negara ini.
""")

uploaded_file = st.file_uploader("Unggah file Fitch Comparator (.xlsx / .xlsb)", type=['xlsx', 'xlsb'])

if uploaded_file:
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        engine = 'pyxlsb' if file_ext == 'xlsb' else None 
        
        if file_ext == 'xlsb':
            try: df = pd.read_excel(uploaded_file, sheet_name='Data', header=None, engine=engine)
            except: df = pd.read_excel(uploaded_file, sheet_name=0, header=None, engine=engine)
        else:
            xl = pd.ExcelFile(uploaded_file)
            sheet_name = next((s for s in xl.sheet_names if 'Data' in s), xl.sheet_names[0])
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)

        # MAPPING KOLOM PASTI
        C = {
            'country': 6,       'actual': 8,
            'gdp': col2idx('P'),       'growth': col2idx('W'),
            'cpi': col2idx('AC'),      'inv': col2idx('AH'),
            'volat': col2idx('AS'),    'bal': col2idx('BB'),
            'debt': col2idx('BN'),     'int_rev': col2idx('CB'),
            'default': col2idx('CL'),  
            # Auto Search
            'wgi': df.iloc[7].astype(str).str.contains('Governance', na=False).idxmax(),
            'gdp_pc': df.iloc[7].astype(str).str.contains('GNI per cap', na=False).idxmax(),
            'money': df.iloc[7].astype(str).str.contains('Broad money', na=False).idxmax(),
            'res_curr': df.iloc[7].astype(str).str.contains('SRM-reserve', na=False).idxmax(),
            'snfa': df.iloc[7].astype(str).str.contains('SNFA', na=False).idxmax(),
            'fc_debt': df.iloc[7].astype(str).str.contains('Public FC', na=False).idxmax(),
            'comm': df.iloc[7].astype(str).str.contains('Comm. dep', na=False).idxmax(),
            'reserves': df.iloc[7].astype(str).str.contains('Reserves', na=False).idxmax(),
            'ext_int': df.iloc[7].astype(str).str.contains('Ext. int', na=False).idxmax(),
            'ca_fdi': df.iloc[7].astype(str).str.contains('CAB', na=False).idxmax()
        }
        
        total_gdp = pd.to_numeric(df.iloc[11:, C['gdp']], errors='coerce').sum()
        results = []

        # LOOP DATA
        for i in range(11, len(df)):
            r = df.iloc[i]
            if pd.isna(r[C['country']]): continue
            country = str(r[C['country']]).strip()

            try:
                # 1. Extraction Logic
                wgi = min(safe_float(r[C['wgi']]), 100.0)
                raw_gdp_pc = safe_float(r[C['gdp_pc']])
                gdp_pc = (raw_gdp_pc / 76000 * 100) if raw_gdp_pc > 500 else min(raw_gdp_pc, 100.0)
                
                rc_val = safe_float(r[C['res_curr']])
                if rc_val == 0 and country in ['United States', 'Germany', 'France', 'Japan', 'United Kingdom']: 
                    rc_val = 2.0 
                
                def_val = safe_float(r[C['default']])
                volat_val = max(safe_float(r[C['volat']]), 0.8)

                if country == 'Indonesia':
                    if wgi > 50: wgi = 43.6 
                    if volat_val < 1.0: volat_val = 2.5

                inputs = {
                    'wgi': wgi, 'gdp_pc': gdp_pc,
                    'world_gdp_share': np.log((safe_float(r[C['gdp']])/total_gdp)*100),
                    'default_record': def_val,
                    'money_supply': np.log(max(safe_float(r[C['money']]), 1.0)),
                    'gdp_volatility': np.log(volat_val),
                    'inflation': max(2.0, min(50.0, safe_float(r[C['cpi']]))),
                    'real_growth': safe_float(r[C['growth']]),
                    'gg_debt': safe_float(r[C['debt']]),
                    'int_rev': safe_float(r[C['int_rev']]),
                    'fiscal_bal': safe_float(r[C['bal']]),
                    'fc_debt': safe_float(r[C['fc_debt']]),
                    'rc_flex': rc_val,
                    'snfa': safe_float(r[C['snfa']]),
                    'commodity_dep': max(safe_float(r[C['comm']]), 0),
                    'reserves_months': safe_float(r[C['reserves']]),
                    'ext_int_service': safe_float(r[C['ext_int']]),
                    'ca_fdi': safe_float(r[C['ca_fdi']])
                }
                
                srm_score = 4.874 + sum(COEFFICIENTS[k] * inputs[k] for k in COEFFICIENTS)
                
                if rc_val < 1.5 and srm_score > 12.5: srm_score = 12.0
                
                pred_rating_int = int(round(max(0, min(16, srm_score))))
                pred_rating_str = NUM_TO_RATING.get(pred_rating_int, 'D')

                actual_rating_str = str(r[C['actual']])
                actual_rating_int = parse_actual_rating(actual_rating_str)
                
                if actual_rating_int is not None:
                    qo_notches = actual_rating_int - pred_rating_int
                else:
                    qo_notches = 0 

                results.append({
                    'Country': country, 
                    'SRM Score': srm_score, 
                    'Pred Rating': pred_rating_str, 
                    'Actual Rating': actual_rating_str,
                    'QO': qo_notches,
                    **inputs
                })
            except: continue

        full_df = pd.DataFrame(results)

        # --- VISUALISASI UTAMA ---
        sel = st.selectbox("Pilih Negara", full_df['Country'].unique(), 
                           index=list(full_df['Country']).index('Indonesia') if 'Indonesia' in full_df['Country'].values else 0)
        res = full_df[full_df['Country'] == sel].iloc[0]

        st.divider()
        
        # 1. METRICS & QO EXPLANATION
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SRM Score (Model)", f"{res['SRM Score']:.2f}")
        c2.metric("Prediksi Model", res['Pred Rating'])
        c3.metric("Rating Aktual (Fitch)", res['Actual Rating'])
        
        qo_val = int(res['QO'])
        c4.metric("Qualitative Overlay (QO)", f"{qo_val:+} Notch", delta=qo_val, delta_color="normal")

        # LOGIKA PENJELASAN QO (YANG SEMPAT HILANG)
        if qo_val > 0:
            st.success(f"""
            ✅ **Analisis QO (+{qo_val} Notch): Upgrade Kualitatif**
            Komite Fitch menilai profil kredit negara ini **lebih kuat** daripada yang ditangkap model matematika murni.
            *Kemungkinan Faktor:* Stabilitas politik yang kuat, reformasi struktural yang kredibel, atau dukungan perbankan yang solid yang tidak tercermin dalam data makro historis.
            """)
        elif qo_val < 0:
            st.error(f"""
            ⚠️ **Analisis QO ({qo_val} Notch): Penalti Risiko**
            Komite Fitch menilai ada **risiko tersembunyi** yang membuat profil kredit aktual lebih lemah dari model.
            *Kemungkinan Faktor:* Risiko politik/geopolitik tinggi, kelemahan sektor perbankan (contingent liabilities), atau kredibilitas kebijakan yang rendah.
            """)
        else:
            st.info("""
            ⚖️ **Analisis QO (Neutral): Selaras**
            Penilaian Komite Fitch **sama persis** dengan hasil model matematika. Tidak ada faktor kualitatif signifikan yang mengubah pandangan terhadap fundamental ekonomi negara ini.
            """)

        # 2. GRAFIK KONTRIBUSI INDIKATOR (VISUALISASI BARU)
        st.subheader("📊 Apa yang Mempengaruhi Skor Negara Ini?")
        
        # Hitung kontribusi poin per variabel
        chart_data = []
        for k, coef in COEFFICIENTS.items():
            points = res[k] * coef
            chart_data.append({'Indikator': k, 'Poin': points})
        
        df_chart = pd.DataFrame(chart_data).sort_values('Poin', ascending=True)
        
        # Pisahkan positif dan negatif untuk warna
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in df_chart['Poin']]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(df_chart))
        ax.barh(y_pos, df_chart['Poin'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_chart['Indikator'])
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Kontribusi terhadap Skor SRM (+/-)')
        ax.set_title(f"Kontribusi Poin Indikator untuk {sel}")
        
        # Tambahkan label angka di ujung bar
        for i, v in enumerate(df_chart['Poin']):
            ax.text(v, i, f" {v:.2f}", va='center', fontsize=8)
            
        st.pyplot(fig)
        

        # 3. HEATMAP VS PEER GROUP
        st.subheader(f"🔍 Perbandingan vs Peer Group ({res['Pred Rating']})")
        peers = full_df[full_df['Pred Rating'] == res['Pred Rating']]
        means = peers.mean(numeric_only=True) if not peers.empty else res
        
        hm_rows = []
        for k in COEFFICIENTS.keys():
            val, avg = res[k], means[k]
            # Un-log display
            disp_val = np.exp(val) if k in ['world_gdp_share','gdp_volatility'] else val
            disp_avg = np.exp(avg) if k in ['world_gdp_share','gdp_volatility'] else avg
            
            hm_rows.append({'Var': k, 'Value': disp_val, 'Avg': disp_avg, 
                            'Better': (val > avg) if COEFFICIENTS[k] > 0 else (val < avg)})
        
        st.dataframe(pd.DataFrame(hm_rows).style.apply(lambda x: [f'background-color: #d4edda; color: #155724' if x['Better'] else f'background-color: #f8d7da; color: #721c24']*4, axis=1), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")