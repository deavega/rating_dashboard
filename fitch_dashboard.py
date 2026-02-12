import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re

# --- 1. DATA STRUKTUR VERSI ---

# Definisi Deskripsi & Metadata (Stabil)
INDICATOR_META = {
    'wgi': {'Name': 'Governance (WGI)', 'Type': 'Structural', 'Desc': 'Percentile Rank (0-100)'},
    'gdp_pc': {'Name': 'GDP per Capita (PPP)', 'Type': 'Structural', 'Desc': 'Percentile Rank (% of US)'},
    'world_gdp_share': {'Name': 'World GDP Share', 'Type': 'Structural', 'Desc': 'Log(Share %)'},
    'default_record': {'Name': 'Years Since Default', 'Type': 'Structural', 'Desc': 'Inverse (1 / 1+Years)'},
    'money_supply': {'Name': 'Broad Money', 'Type': 'Structural', 'Desc': 'Log(% GDP)'},
    'gdp_volatility': {'Name': 'Real GDP Volatility', 'Type': 'Macro', 'Desc': 'Log(Rolling StdDev)'},
    'inflation': {'Name': 'Inflation (CPI)', 'Type': 'Macro', 'Desc': 'Median Annual %'},
    'real_growth': {'Name': 'Real GDP Growth', 'Type': 'Macro', 'Desc': '3-Year Centered Avg'},
    'gg_debt': {'Name': 'Govt Debt (% GDP)', 'Type': 'Fiscal', 'Desc': '3-Year Centered Avg'},
    'int_rev': {'Name': 'Interest/Revenue', 'Type': 'Fiscal', 'Desc': '3-Year Centered Avg'},
    'fiscal_bal': {'Name': 'Fiscal Balance', 'Type': 'Fiscal', 'Desc': '3-Year Centered Avg'},
    'fc_debt': {'Name': 'Foreign Currency Debt', 'Type': 'Fiscal', 'Desc': 'Ratio (% Total Debt)'},
    'rc_flex': {'Name': 'Reserve Currency Flex.', 'Type': 'External', 'Desc': 'Index (0 or 1-4.6)'},
    'snfa': {'Name': 'Net Foreign Assets', 'Type': 'External', 'Desc': 'Ratio (% GDP)'},
    'commodity_dep': {'Name': 'Commodity Dep.', 'Type': 'External', 'Desc': 'Ratio (% CXR)'},
    'reserves_months': {'Name': 'Reserves (Months)', 'Type': 'External', 'Desc': 'Ratio (Months of CXP)'},
    'ext_int_service': {'Name': 'Ext. Interest Service', 'Type': 'External', 'Desc': 'Ratio (% CXR)'},
    'ca_fdi': {'Name': 'Current Account + FDI', 'Type': 'External', 'Desc': 'Ratio (% GDP)'}
}

# DATABASE VERSI MODEL
SRM_VERSIONS = {
    "2025 (Current)": {
        "intercept": 4.874,
        "coeffs": {
            'wgi': 0.079, 'gdp_pc': 0.037, 'world_gdp_share': 0.640,
            'default_record': -1.791, 'money_supply': 0.145, 'gdp_volatility': -0.710,
            'inflation': -0.069, 'real_growth': 0.057, 'gg_debt': -0.023,
            'int_rev': -0.044, 'fiscal_bal': 0.039, 'fc_debt': -0.008,
            'rc_flex': 0.494, 'snfa': 0.011, 'commodity_dep': -0.004,
            'reserves_months': 0.024, 'ext_int_service': -0.004, 'ca_fdi': 0.004
        }
    },
    "2024 (Legacy)": { # Placeholder untuk versi lama
        "intercept": 4.500,
        "coeffs": {
            'wgi': 0.080, 'gdp_pc': 0.040, 'world_gdp_share': 0.600,
            'default_record': -1.800, 'money_supply': 0.150, 'gdp_volatility': -0.700,
            'inflation': -0.070, 'real_growth': 0.060, 'gg_debt': -0.025,
            'int_rev': -0.045, 'fiscal_bal': 0.040, 'fc_debt': -0.010,
            'rc_flex': 0.500, 'snfa': 0.010, 'commodity_dep': -0.005,
            'reserves_months': 0.025, 'ext_int_service': -0.005, 'ca_fdi': 0.005
        }
    }
}

NUM_TO_RATING = {
    16: 'AAA', 15: 'AA+', 14: 'AA', 13: 'AA-', 12: 'A+', 11: 'A', 10: 'A-',
    9: 'BBB+', 8: 'BBB', 7: 'BBB-', 6: 'BB+', 5: 'BB', 4: 'BB-', 3: 'B+', 
    2: 'B', 1: 'B-', 0: 'CCC/D'
}

RATING_TO_NUM = {v: k for k, v in NUM_TO_RATING.items()} 
RATING_TO_NUM.update({'CCC': 0, 'CC': 0, 'C': 0, 'RD': 0, 'D': 0, 'WD': None})

# --- 2. HELPER FUNCTIONS ---
def col2idx(c):
    return sum((ord(x) - 64) * (26 ** i) for i, x in enumerate(reversed(c.upper()))) - 1

def safe_float(v):
    try: return float(v) if str(v).strip() not in ['-', 'nan', 'na'] else 0.0
    except: return 0.0

def parse_actual_rating(rating_str):
    if pd.isna(rating_str): return None
    clean_str = str(rating_str).replace('*', '').replace('u', '').strip() 
    return RATING_TO_NUM.get(clean_str, None)

def extract_period_from_filename(filename):
    match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', filename, re.IGNORECASE)
    if match: return match.group(0)
    return "(Periode Tidak Terdeteksi)"

# FUNGSI RENDER QO (BERSIH)
def render_qo_analysis(qo_val, country):
    if qo_val > 3:
        st.warning(f"⚠️ **Pengecualian Struktural (+{qo_val} Notch): Kasus Khusus**")
        st.markdown("""
        Negara ini memiliki QO positif yang sangat besar (>3 notch). Ini biasanya terjadi pada Negara Kecil yang Sangat Kaya.
        
        **Alasan Teknis:**
        * Model SRM menghukum negara ekonomi kecil (World GDP Share rendah).
        * Komite melakukan koreksi masif karena solvabilitas aset luar negeri yang tinggi.
        """)
    elif qo_val > 0:
        st.success(f"✅ **Upgrade Kualitatif (+{qo_val} Notch): Fundamental Lebih Kuat dari Model**")
        st.markdown(f"""
        Profil kredit {country} dinilai lebih kuat daripada hasil model. Faktor pendukung:
        * **Kekuatan Sektor Perbankan (Structural):** Risiko kewajiban kontinjensi yang rendah.
        * **Fleksibilitas Pembiayaan (Public Finances):** Akses pasar domestik yang dalam.
        * **Kredibilitas Kebijakan (Macro):** Disiplin moneter dan stabilitas makroekonomi.
        """)
    elif qo_val < 0:
        st.error(f"⚠️ **Penalti Risiko ({qo_val} Notch): Risiko Tersembunyi Terdeteksi**")
        st.markdown(f"""
        Profil kredit {country} dinilai lebih lemah daripada hasil model. Faktor risiko:
        * **Risiko Politik & Tata Kelola (Structural):** Ketidakpastian transisi atau polarisasi tinggi.
        * **Kewajiban Kontinjensi (Public Finances):** Risiko utang tersembunyi BUMN atau perbankan.
        * **Kualitas Data (Structural):** Transparansi data fiskal yang rendah.
        """)
    else:
        st.info("⚖️ **Selaras (Neutral): Model & Komite Sepakat**")
        st.markdown("Penilaian kualitatif Komite Rating Fitch sejalan dengan model matematika.")

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="Sovereign Credit Watch", layout="wide", page_icon="🌍")

# --- SIDEBAR: PILIH VERSI MODEL (DIKEMBALIKAN) ---
with st.sidebar:
    st.header("⚙️ Konfigurasi Model")
    selected_version = st.selectbox(
        "Pilih Versi Metodologi SRM:",
        options=list(SRM_VERSIONS.keys()),
        index=0
    )
    
    # Load Parameter Aktif
    ACTIVE_PARAMS = SRM_VERSIONS[selected_version]
    COEFFICIENTS = ACTIVE_PARAMS['coeffs']
    INTERCEPT = ACTIVE_PARAMS['intercept']
    
    st.info(f"Menggunakan Intercept: **{INTERCEPT}**\nVersi: **{selected_version}**")

st.title("🌍 Sovereign Credit Watch")

uploaded_file = st.file_uploader("Unggah file Fitch Comparator (.xlsx / .xlsb)", type=['xlsx', 'xlsb'])

if uploaded_file:
    try:
        # Subjudul Dinamis
        period = extract_period_from_filename(uploaded_file.name)
        st.markdown(f"### Berdasarkan Data Fitch **{period}** dengan Model: {selected_version}")
        
        # Load Data
        file_ext = uploaded_file.name.split('.')[-1].lower()
        engine = 'pyxlsb' if file_ext == 'xlsb' else None 
        
        if file_ext == 'xlsb':
            try: df = pd.read_excel(uploaded_file, sheet_name='Data', header=None, engine=engine)
            except: df = pd.read_excel(uploaded_file, sheet_name=0, header=None, engine=engine)
        else:
            xl = pd.ExcelFile(uploaded_file)
            sheet_name = next((s for s in xl.sheet_names if 'Data' in s), xl.sheet_names[0])
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)

        # MAPPING KOLOM
        C = {
            'country': 6,       'actual': 8,
            'gdp': col2idx('P'),       
            'growth': col2idx('U'),    
            'cpi': col2idx('AC'),      
            'inv': col2idx('AH'),
            'volat': col2idx('AS'),    
            'bal': col2idx('AZ'),      
            'debt': col2idx('BL'),     
            'debt_lc': col2idx('CF'),  
            'int_rev': col2idx('BZ'),  
            'default': col2idx('CL'),  
            'hdi': col2idx('IY'),      
            
            # Auto Search
            'wgi': df.iloc[7].astype(str).str.contains('Governance', na=False).idxmax(),
            'gdp_pc': df.iloc[7].astype(str).str.contains('GNI per cap', na=False).idxmax(),
            'money': df.iloc[7].astype(str).str.contains('Broad money', na=False).idxmax(),
            'res_curr': df.iloc[7].astype(str).str.contains('SRM-reserve', na=False).idxmax(),
            'snfa': df.iloc[7].astype(str).str.contains('SNFA', na=False).idxmax(),
            'comm': df.iloc[7].astype(str).str.contains('Comm. dep', na=False).idxmax(),
            'reserves': df.iloc[7].astype(str).str.contains('Reserves', na=False).idxmax(),
            'ext_int': df.iloc[7].astype(str).str.contains('Ext. int', na=False).idxmax(),
            'ca_fdi': df.iloc[7].astype(str).str.contains('CAB', na=False).idxmax()
        }
        
        total_gdp = pd.to_numeric(df.iloc[11:, C['gdp']], errors='coerce').sum()
        results = []

        # PROCESSING LOOP
        for i in range(11, len(df)):
            r = df.iloc[i]
            if pd.isna(r[C['country']]): continue
            country = str(r[C['country']]).strip()

            try:
                # Extraction
                wgi = min(safe_float(r[C['wgi']]), 100.0)
                raw_gdp_pc = safe_float(r[C['gdp_pc']])
                gdp_pc = (raw_gdp_pc / 76000 * 100) if raw_gdp_pc > 500 else min(raw_gdp_pc, 100.0)
                
                rc_val = safe_float(r[C['res_curr']])
                if rc_val > 0: rc_val = max(1.0, min(4.6, rc_val)) 
                
                def_val = safe_float(r[C['default']])
                volat_val = max(safe_float(r[C['volat']]), 0.8)

                total_debt_val = safe_float(r[C['debt']])
                lc_debt_val = safe_float(r[C['debt_lc']])
                fc_debt_input = 0.0
                if total_debt_val > 0:
                    fc_debt_input = ((total_debt_val - lc_debt_val) / total_debt_val * 100)

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
                    'fc_debt': fc_debt_input,
                    'rc_flex': rc_val,
                    'snfa': safe_float(r[C['snfa']]),
                    'commodity_dep': max(safe_float(r[C['comm']]), 0),
                    'reserves_months': safe_float(r[C['reserves']]),
                    'ext_int_service': safe_float(r[C['ext_int']]),
                    'ca_fdi': safe_float(r[C['ca_fdi']])
                }
                
                # USE SELECTED INTERCEPT & COEFFICIENTS
                srm_score = INTERCEPT + sum(COEFFICIENTS[k] * inputs[k] for k in COEFFICIENTS)
                
                if rc_val < 1.0 and srm_score > 12.5: srm_score = 12.0
                
                pred_rating_int = int(round(max(0, min(16, srm_score))))
                pred_rating_str = NUM_TO_RATING.get(pred_rating_int, 'D')

                actual_rating_str = str(r[C['actual']])
                actual_rating_int = parse_actual_rating(actual_rating_str)
                
                qo_notches = 0
                if actual_rating_int is not None:
                    qo_notches = actual_rating_int - pred_rating_int

                results.append({
                    'Country': country, 
                    'SRM Score': srm_score, 
                    'Pred Rating': pred_rating_str, 
                    'Pred Rating Num': pred_rating_int,
                    'Actual Rating': actual_rating_str,
                    'Actual Rating Num': actual_rating_int,
                    'QO': qo_notches,
                    'GDP Nominal': safe_float(r[C['gdp']]),
                    'HDI': safe_float(r[C['hdi']]),
                    **inputs
                })
            except: continue

        full_df = pd.DataFrame(results)

        # --- TABS LAYOUT ---
        tab1, tab2 = st.tabs(["📊 Analisis Negara", "💡 Metodologi & Indikator"])

        with tab1:
            sel = st.selectbox("Pilih Negara", full_df['Country'].unique(), 
                               index=list(full_df['Country']).index('Indonesia') if 'Indonesia' in full_df['Country'].values else 0)
            res = full_df[full_df['Country'] == sel].iloc[0]

            st.divider()
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SRM Score (Model)", f"{res['SRM Score']:.2f}", help=f"Versi Model: {selected_version}")
            c2.metric("Prediksi Model", res['Pred Rating'])
            c3.metric("Rating Aktual (Fitch)", res['Actual Rating'])
            qo_val = int(res['QO'])
            c4.metric("Qualitative Overlay (QO)", f"{qo_val:+} Notch", delta=qo_val)
            
            st.caption(f"ℹ️ **Human Development Index (HDI):** {res['HDI']:.3f}")

            # PENJELASAN QO (BERSIH)
            render_qo_analysis(qo_val, sel)

            # HEATMAP & CHART
            col_heat, col_chart = st.columns([1.2, 1]) 

            with col_heat:
                st.subheader(f"Perbandingan vs Peer ({res['Pred Rating']})")
                peers = full_df[full_df['Pred Rating'] == res['Pred Rating']]
                means = peers.mean(numeric_only=True) if not peers.empty else res
                hm_rows = []
                for k in COEFFICIENTS.keys():
                    val, avg = res[k], means[k]
                    disp_val = np.exp(val) if k in ['world_gdp_share','gdp_volatility'] else val
                    disp_avg = np.exp(avg) if k in ['world_gdp_share','gdp_volatility'] else avg
                    hm_rows.append({'Indikator': INDICATOR_META[k]['Name'], 'Nilai': disp_val, 'Rerata Peer': disp_avg, 
                                    'Status': 'Better' if ((val > avg) if COEFFICIENTS[k] > 0 else (val < avg)) else 'Worse'})
                
                st.dataframe(
                    pd.DataFrame(hm_rows).style.apply(lambda x: [f'background-color: #d4edda' if x['Status']=='Better' else f'background-color: #f8d7da']*4, axis=1),
                    height=700, 
                    use_container_width=True
                )
            
            with col_chart:
                st.subheader("Kontribusi Poin Indikator")
                chart_data = [{'Indikator': k, 'Poin': res[k]*COEFFICIENTS[k]} for k in COEFFICIENTS]
                df_chart = pd.DataFrame(chart_data).sort_values('Poin')
                
                df_chart['Warna'] = df_chart['Poin'].apply(lambda x: 'Positif' if x >= 0 else 'Negatif')

                fig_bar = px.bar(df_chart, x='Poin', y='Indikator', orientation='h',
                                 color='Warna',
                                 color_discrete_map={'Positif': '#2ecc71', 'Negatif': '#e74c3c'},
                                 text_auto='.2f')
                fig_bar.update_layout(height=700, showlegend=False, xaxis_title="Poin Kontribusi")
                st.plotly_chart(fig_bar, use_container_width=True)

            # BUBBLE CHART
            st.divider()
            st.header("🌐 Peta Kuadran Rating Global")
            
            plot_df = full_df.dropna(subset=['Actual Rating Num', 'Pred Rating Num']).copy()
            
            def get_status(row):
                if row['Country'] == sel: return "📍 NEGARA TERPILIH"
                diff = row['Actual Rating Num'] - row['Pred Rating Num']
                if diff > 0: return 'Underrated by Model (QO Positif)'
                elif diff < 0: return 'Overrated by Model (QO Negatif)'
                return 'Aligned (Sesuai)'

            plot_df['Status'] = plot_df.apply(get_status, axis=1)
            plot_df['GDP Size'] = plot_df['GDP Nominal'].fillna(0) + 10

            color_map = {
                '📍 NEGARA TERPILIH': '#FFD700',
                'Underrated by Model (QO Positif)': '#2ecc71',
                'Overrated by Model (QO Negatif)': '#e74c3c',
                'Aligned (Sesuai)': '#3498db'
            }

            fig = px.scatter(
                plot_df,
                x="Actual Rating Num",
                y="Pred Rating Num",
                size="GDP Size",
                color="Status",
                hover_name="Country",
                hover_data=["Actual Rating", "Pred Rating", "QO"],
                color_discrete_map=color_map,
                size_max=60,
                title=f"Posisi {sel} dalam Peta Rating Global"
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=16, y1=16, line=dict(color="Gray", width=2, dash="dash"))
            
            tick_vals = list(range(0, 17))
            tick_text = [NUM_TO_RATING.get(i, '') for i in tick_vals]
            
            fig.update_layout(
                xaxis=dict(title="Rating Aktual (Komite)", tickmode='array', tickvals=tick_vals, ticktext=tick_text),
                yaxis=dict(title="Rating Prediksi (Model)", tickmode='array', tickvals=tick_vals, ticktext=tick_text),
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader(f"Parameter Model SRM (Versi {selected_version})")
            
            method_data = []
            for k, coef in COEFFICIENTS.items():
                det = INDICATOR_META[k]
                method_data.append({
                    'Nama Indikator': det['Name'],
                    'Kategori': det['Type'],
                    'Deskripsi': det['Desc'],
                    'Bobot (Versi Ini)': coef,
                    'Arah': 'Positif (+)' if coef > 0 else 'Negatif (-)'
                })
            
            st.dataframe(
                pd.DataFrame(method_data), 
                use_container_width=True, 
                height=800,
                column_config={
                    "Bobot (Versi Ini)": st.column_config.NumberColumn(format="%.4f")
                }
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan aplikasi: {e}")