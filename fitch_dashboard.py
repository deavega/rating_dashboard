import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from io import BytesIO

# Coba import pypdf
try:
    from pypdf import PdfReader
except ImportError:
    st.error("Library 'pypdf' belum terinstall. Mohon jalankan: pip install pypdf")
    PdfReader = None

# --- 1. KONFIGURASI & METADATA ---

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
    }
}

INDICATOR_META = {
    'wgi': {'Name': 'Governance (WGI)', 'Type': 'Structural', 'Unit': 'Index 0-100', 'Trans': 'Linear', 'Keywords': ['governance', 'structural', 'political']},
    'gdp_pc': {'Name': 'GDP per Capita (PPP)', 'Type': 'Structural', 'Unit': '% of US', 'Trans': 'Linear', 'Keywords': ['gdp per capita', 'income']},
    'world_gdp_share': {'Name': 'World GDP Share', 'Type': 'Structural', 'Unit': '% Share', 'Trans': 'Log', 'Keywords': ['size', 'share']},
    'default_record': {'Name': 'Years Since Default', 'Type': 'Structural', 'Unit': 'Years', 'Trans': 'Inverse', 'Keywords': ['default', 'history']},
    'money_supply': {'Name': 'Broad Money', 'Type': 'Structural', 'Unit': '% GDP', 'Trans': 'Log', 'Keywords': ['intermediation', 'banking']},
    
    'gdp_volatility': {'Name': 'Real GDP Volatility', 'Type': 'Macro', 'Unit': 'Std Dev', 'Trans': 'Log_Floor', 'Keywords': ['volatility']},
    'inflation': {'Name': 'Inflation (CPI)', 'Type': 'Macro', 'Unit': '%', 'Trans': 'Cap_50', 'Keywords': ['inflation', 'cpi']},
    'real_growth': {'Name': 'Real GDP Growth', 'Type': 'Macro', 'Unit': '%', 'Trans': 'Linear', 'Keywords': ['growth', 'gdp growth']},
    
    'gg_debt': {'Name': 'Govt Debt', 'Type': 'Fiscal', 'Unit': '% GDP', 'Trans': 'Linear', 'Keywords': ['debt', 'public finance', 'fiscal']},
    'int_rev': {'Name': 'Interest/Revenue', 'Type': 'Fiscal', 'Unit': '% Rev', 'Trans': 'Linear', 'Keywords': ['interest', 'revenue', 'affordability']},
    'fiscal_bal': {'Name': 'Fiscal Balance', 'Type': 'Fiscal', 'Unit': '% GDP', 'Trans': 'Linear', 'Keywords': ['deficit', 'fiscal balance']},
    'fc_debt': {'Name': 'Foreign Ccy Debt', 'Type': 'Fiscal', 'Unit': '% Total Debt', 'Trans': 'Linear', 'Keywords': ['foreign currency', 'fx debt']},
    
    'rc_flex': {'Name': 'Reserve Currency Flex.', 'Type': 'External', 'Unit': 'Index 0-4.6', 'Trans': 'Linear', 'Keywords': ['reserve currency']},
    'snfa': {'Name': 'Net Foreign Assets', 'Type': 'External', 'Unit': '% GDP', 'Trans': 'Linear', 'Keywords': ['net foreign assets', 'nfa']},
    'commodity_dep': {'Name': 'Commodity Dep.', 'Type': 'External', 'Unit': '% CXR', 'Trans': 'Linear', 'Keywords': ['commodity']},
    'reserves_months': {'Name': 'Reserves', 'Type': 'External', 'Unit': 'Months CXP', 'Trans': 'Linear', 'Keywords': ['reserves', 'fx reserve', 'external liquidity']},
    'ext_int_service': {'Name': 'Ext. Interest Service', 'Type': 'External', 'Unit': '% CXR', 'Trans': 'Linear', 'Keywords': ['external interest']},
    'ca_fdi': {'Name': 'Current Account + FDI', 'Type': 'External', 'Unit': '% GDP', 'Trans': 'Linear', 'Keywords': ['current account', 'cad', 'fdi']}
}

NUM_TO_RATING = {
    16: 'AAA', 15: 'AA+', 14: 'AA', 13: 'AA-', 12: 'A+', 11: 'A', 10: 'A-',
    9: 'BBB+', 8: 'BBB', 7: 'BBB-', 6: 'BB+', 5: 'BB', 4: 'BB-', 3: 'B+', 
    2: 'B', 1: 'B-', 0: 'CCC/D'
}
RATING_TO_NUM = {v: k for k, v in NUM_TO_RATING.items()} 
RATING_TO_NUM.update({'CCC': 0, 'CC': 0, 'C': 0, 'RD': 0, 'D': 0, 'WD': None})

# --- 2. HELPER FUNCTIONS ---
def col2idx(c): return sum((ord(x) - 64) * (26 ** i) for i, x in enumerate(reversed(c.upper()))) - 1
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

# --- 3. LOGIC PARSING PDF ---
def parse_fitch_report(pdf_file):
    if PdfReader is None: return {}
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    sensitivities = {'Positive': [], 'Negative': []}
    try:
        clean_text = re.sub(r'\s+', ' ', text)
        
        # Regex Pattern
        neg_match = re.search(r'Lead to Negative Rating Action/Downgrade(.*?)Factors that Could', clean_text, re.IGNORECASE)
        if not neg_match: neg_match = re.search(r'Lead to Negative Rating Action/Downgrade(.*?)SOVEREIGN RATING MODEL', clean_text, re.IGNORECASE)
        
        if neg_match:
            points = re.split(r' - | – ', neg_match.group(1))
            sensitivities['Negative'] = [p.strip() for p in points if len(p) > 10]

        pos_match = re.search(r'Lead to Positive Rating Action/Upgrade(.*?)Factors that Could', clean_text, re.IGNORECASE)
        if not pos_match: pos_match = re.search(r'Lead to Positive Rating Action/Upgrade(.*?)SOVEREIGN RATING MODEL', clean_text, re.IGNORECASE)

        if pos_match:
            points = re.split(r' - | – ', pos_match.group(1))
            sensitivities['Positive'] = [p.strip() for p in points if len(p) > 10]
            
    except Exception as e:
        st.error(f"Gagal memparsing PDF: {e}")
    return sensitivities

def map_sensitivities_to_indicators(sensitivities):
    active_constraints = {} 
    for direction, points in sensitivities.items():
        for point in points:
            p_lower = point.lower()
            for code, meta in INDICATOR_META.items():
                for keyword in meta.get('Keywords', []):
                    if keyword in p_lower:
                        if code not in active_constraints:
                            active_constraints[code] = {'direction': [], 'text': []}
                        active_constraints[code]['direction'].append(direction)
                        active_constraints[code]['text'].append(point[:150]+"...")
    return active_constraints

# --- FUNGSI RENDER QO (BERSIH & FORMAL) ---
def render_qo_analysis(qo_val, country):
    """Menampilkan analisis QO yang mendalam tanpa ikon"""
    
    if qo_val > 3:
        st.warning(f"⚠️ **Pengecualian Struktural (+{qo_val} Notch): Kasus Khusus**")
        st.markdown("""
        Negara ini memiliki QO positif yang sangat besar (>3 notch). Ini biasanya terjadi pada **Negara Kecil yang Sangat Kaya** (seperti Singapura, Denmark, Norwegia).
        
        **Alasan Teknis:**
        * Model SRM memiliki variabel **'World GDP Share'** yang menghukum negara dengan ekonomi kecil secara matematis.
        * Komite Fitch harus mengoreksi bias ini karena negara tersebut memiliki aset luar negeri (Sovereign Wealth Funds) yang masif dan solvabilitas sangat tinggi yang tidak tertangkap sepenuhnya oleh model GDP flow.
        """)

    elif qo_val > 0:
        st.success(f"✅ **Upgrade Kualitatif (+{qo_val} Notch): Fundamental Lebih Kuat dari Model**")
        st.markdown(f"""
        Komite Fitch menilai profil kredit {country} lebih kuat daripada hasil perhitungan mesin. Faktor pendukung biasanya meliputi:
        
        * **Kekuatan Sektor Perbankan (Structural):**
            Sektor perbankan yang sangat kuat, termodalisasi dengan baik, dan risiko kewajiban kontinjensi (*Contingent Liabilities*) yang sangat rendah bagi pemerintah.
        * **Fleksibilitas Pembiayaan (Public Finances):**
            Akses pasar utang yang dalam (deep domestic bond market) sehingga negara tidak bergantung pada utang valas, mengurangi risiko volatilitas nilai tukar.
        * **Kredibilitas Kebijakan (Macro):**
            Rekam jejak bank sentral yang disiplin dalam menjaga inflasi dan stabilitas makroekonomi jangka panjang.
        """)

    elif qo_val < 0:
        st.error(f"⚠️ **Penalti Risiko ({qo_val} Notch): Risiko Tersembunyi Terdeteksi**")
        st.markdown(f"""
        Komite Fitch menilai profil kredit {country} lebih lemah daripada hasil perhitungan mesin. Ada risiko kualitatif yang membebani rating:
        
        * **Risiko Politik & Tata Kelola (Structural):**
            Ketidakpastian transisi kepemimpinan, polarisasi politik tinggi, atau risiko geopolitik yang dapat menghambat pembayaran utang.
        * **Kewajiban Kontinjensi (Public Finances):**
            Risiko utang tersembunyi dari BUMN (*State-Owned Enterprises*) atau sektor perbankan yang lemah yang sewaktu-waktu bisa menjadi beban anggaran negara (bailout risk).
        * **Kualitas Data (Structural):**
            Transparansi data fiskal atau cadangan devisa yang rendah dapat menyebabkan penalti rating karena ketidakpastian.
        """)
        
    else:
        st.info("⚖️ **Selaras (Neutral): Model & Komite Sepakat**")
        st.markdown("Penilaian kualitatif Komite Rating Fitch sejalan dengan model matematika. Tidak ada distorsi struktural atau risiko tersembunyi yang signifikan.")

    # Expander Edukasi
    with st.expander("Tentang Qualitative Overlay (QO)"):
        st.markdown("""
        **QO (Qualitative Overlay)** adalah penyesuaian tahap akhir yang dilakukan oleh Komite Rating Fitch.
        
        Model SRM hanyalah angka awal. Komite dapat menambah atau mengurangi rating berdasarkan faktor yang sulit dikuantifikasi:
        1.  **Structural Features:** Kualitas data, risiko politik, kedalaman sektor keuangan.
        2.  **Macroeconomic Policy:** Kredibilitas target inflasi dan kebijakan fiskal.
        3.  **Public Finances:** Struktur utang (valas vs lokal) dan risiko BUMN.
        4.  **External Finances:** Dukungan donor bilateral/multilateral.
        """)

# --- 4. ENGINE CALCULATOR ---
def calculate_single_score(raw_values, intercept, coeffs):
    score = intercept
    breakdown = {}
    for k, v in raw_values.items():
        trans_rule = INDICATOR_META[k]['Trans']
        final_val = v
        if trans_rule == 'Log': final_val = np.log(max(v, 0.001))
        elif trans_rule == 'Log_Floor': final_val = np.log(max(v, 0.8))
        elif trans_rule == 'Inverse': final_val = 1 / (1 + v)
        elif trans_rule == 'Cap_50': final_val = min(v, 50.0)
        
        contribution = final_val * coeffs[k]
        score += contribution
        breakdown[k] = contribution
    return score, breakdown

# --- 5. UI DASHBOARD ---
st.set_page_config(page_title="Sovereign Credit Watch", layout="wide", page_icon="🌍")

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    selected_version = st.selectbox("Versi Metodologi:", list(SRM_VERSIONS.keys()))
    ACTIVE_PARAMS = SRM_VERSIONS[selected_version]
    COEFFICIENTS = ACTIVE_PARAMS['coeffs']
    INTERCEPT = ACTIVE_PARAMS['intercept']

st.title("🌍 Sovereign Credit Watch")

# --- TAMBAHAN DESKRIPSI ---
st.markdown("""
> **Metodologi:** Dashboard ini mengadopsi logika **Fitch Sovereign Rating Model (SRM)**.
> Skor dihitung berdasarkan 18 indikator kuantitatif (Struktural, Makroekonomi, Fiskal, dan Eksternal) untuk memproyeksikan kelayakan kredit negara (Sovereign Rating) sesuai kriteria Fitch Ratings.
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

        # Mapping Columns
        C = {
            'country': 6, 'actual': 8, 'gdp': col2idx('P'), 'growth': col2idx('U'),    
            'cpi': col2idx('AC'), 'inv': col2idx('AH'), 'volat': col2idx('AS'),    
            'bal': col2idx('AZ'), 'debt': col2idx('BL'), 'debt_lc': col2idx('CF'),  
            'int_rev': col2idx('BZ'), 'default': col2idx('CL'), 'hdi': col2idx('IY'),      
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

        for i in range(11, len(df)):
            r = df.iloc[i]
            if pd.isna(r[C['country']]): continue
            country = str(r[C['country']]).strip()
            try:
                # Raw Extraction
                raw = {}
                raw['wgi'] = min(safe_float(r[C['wgi']]), 100.0)
                raw_gni = safe_float(r[C['gdp_pc']])
                raw['gdp_pc'] = (raw_gni / 76000 * 100) if raw_gni > 500 else min(raw_gni, 100.0)
                raw_gdp_val = safe_float(r[C['gdp']])
                raw['world_gdp_share'] = (raw_gdp_val / total_gdp * 100) if raw_gdp_val > 0 else 0.001
                inv_def = safe_float(r[C['default']])
                raw['default_record'] = (1 / inv_def - 1) if inv_def > 0 else 0
                raw['money_supply'] = max(safe_float(r[C['money']]), 1.0)
                raw['gdp_volatility'] = max(safe_float(r[C['volat']]), 0.8)
                raw['inflation'] = safe_float(r[C['cpi']])
                raw['real_growth'] = safe_float(r[C['growth']])
                raw['gg_debt'] = safe_float(r[C['debt']])
                raw['int_rev'] = safe_float(r[C['int_rev']])
                raw['fiscal_bal'] = safe_float(r[C['bal']])
                total_debt_val = safe_float(r[C['debt']])
                lc_debt_val = safe_float(r[C['debt_lc']])
                raw['fc_debt'] = ((total_debt_val - lc_debt_val) / total_debt_val * 100) if total_debt_val > 0 else 0.0
                rc = safe_float(r[C['res_curr']])
                raw['rc_flex'] = max(1.0, min(4.6, rc)) if rc > 0 else 0.0
                raw['snfa'] = safe_float(r[C['snfa']])
                raw['commodity_dep'] = max(safe_float(r[C['comm']]), 0)
                raw['reserves_months'] = safe_float(r[C['reserves']])
                raw['ext_int_service'] = safe_float(r[C['ext_int']])
                raw['ca_fdi'] = safe_float(r[C['ca_fdi']])

                if country == 'Indonesia':
                    if raw['wgi'] > 50: raw['wgi'] = 43.6 
                    if raw['gdp_volatility'] < 1.0: raw['gdp_volatility'] = 2.5

                score, breakdown = calculate_single_score(raw, INTERCEPT, COEFFICIENTS)
                if raw['rc_flex'] < 1.0 and score > 12.5: score = 12.0
                rating_int = int(round(max(0, min(16, score))))
                rating_str = NUM_TO_RATING.get(rating_int, 'D')
                actual_str = str(r[C['actual']])
                actual_int = parse_actual_rating(actual_str)
                qo = (actual_int - rating_int) if actual_int is not None else 0

                results.append({
                    'Country': country, 'SRM Score': score, 'Pred Rating': rating_str, 'Actual Rating': actual_str,
                    'Pred Rating Num': rating_int, 'Actual Rating Num': actual_int, 'QO': qo, 
                    'HDI': safe_float(r[C['hdi']]), 
                    'GDP Nominal': raw_gdp_val,
                    'Raw': raw
                })
            except: continue

        full_df = pd.DataFrame(results)
        tab1, tab2, tab3 = st.tabs(["📊 Analisis Negara", "💡 Metodologi", "🧪 Simulasi Kebijakan"])

        # --- TAB 1: ANALISIS NEGARA ---
        with tab1:
            sel = st.selectbox("Pilih Negara", full_df['Country'].unique(), index=list(full_df['Country']).index('Indonesia') if 'Indonesia' in full_df['Country'].values else 0)
            res = full_df[full_df['Country'] == sel].iloc[0]

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SRM Score", f"{res['SRM Score']:.2f}")
            c2.metric("Prediksi Model", res['Pred Rating'])
            c3.metric("Rating Aktual (Fitch)", res['Actual Rating'])
            c4.metric("QO (Kualitatif)", f"{int(res['QO']):+} Notch", delta=int(res['QO']))
            
            st.caption(f"ℹ️ **Human Development Index (HDI):** {res['HDI']:.3f}")

            render_qo_analysis(int(res['QO']), sel)

            col_heat, col_chart = st.columns([1.2, 1]) 
            with col_heat:
                st.subheader(f"Perbandingan vs Peer ({res['Pred Rating']})")
                peers = full_df[full_df['Pred Rating'] == res['Pred Rating']]
                means = peers.mean(numeric_only=True) if not peers.empty else res['Raw']
                hm_rows = []
                for k in COEFFICIENTS.keys():
                    val = res['Raw'][k]
                    peer_vals = [p['Raw'][k] for i, p in peers.iterrows()]
                    avg = np.mean(peer_vals) if peer_vals else val
                    status = 'Better' if ((val > avg) if COEFFICIENTS[k] > 0 else (val < avg)) else 'Worse'
                    hm_rows.append({'Indikator': INDICATOR_META[k]['Name'], 'Nilai': val, 'Rerata Peer': avg, 'Status': status})
                st.dataframe(pd.DataFrame(hm_rows).style.apply(lambda x: [f'background-color: #d4edda' if x['Status']=='Better' else f'background-color: #f8d7da']*4, axis=1), height=700, use_container_width=True)
            
            with col_chart:
                st.subheader("Kontribusi Poin Indikator")
                _, breakdown = calculate_single_score(res['Raw'], INTERCEPT, COEFFICIENTS)
                chart_data = [{'Indikator': INDICATOR_META[k]['Name'], 'Poin': v} for k, v in breakdown.items()]
                df_chart = pd.DataFrame(chart_data).sort_values('Poin')
                df_chart['Warna'] = df_chart['Poin'].apply(lambda x: 'Positif' if x >= 0 else 'Negatif')
                fig_bar = px.bar(df_chart, x='Poin', y='Indikator', orientation='h', color='Warna',
                                 color_discrete_map={'Positif': '#2ecc71', 'Negatif': '#e74c3c'}, text_auto='.2f')
                fig_bar.update_layout(height=700, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()
            st.header("🌐 Peta Kuadran Rating Global")
            
            plot_df = full_df.dropna(subset=['Actual Rating Num', 'Pred Rating Num']).copy()
            def get_status(row):
                if row['Country'] == sel: return "📍 NEGARA TERPILIH"
                diff = row['Actual Rating Num'] - row['Pred Rating Num']
                if diff > 0: return 'Underrated (QO+)'
                elif diff < 0: return 'Overrated (QO-)'
                return 'Aligned'
                
            plot_df['Status'] = plot_df.apply(get_status, axis=1)
            plot_df['GDP Size'] = plot_df['GDP Nominal'].fillna(0) + 10
            
            fig_bubble = px.scatter(plot_df, x="Actual Rating Num", y="Pred Rating Num", size="GDP Size", color="Status",
                             hover_name="Country", color_discrete_map={'📍 NEGARA TERPILIH': '#FFD700', 'Underrated (QO+)': '#2ecc71', 'Overrated (QO-)': '#e74c3c', 'Aligned': '#3498db'}, size_max=60)
            fig_bubble.add_shape(type="line", x0=0, y0=0, x1=16, y1=16, line=dict(color="Gray", width=2, dash="dash"))
            
            tick_vals = list(range(0, 17))
            tick_text = [NUM_TO_RATING.get(i, '') for i in tick_vals]
            fig_bubble.update_layout(height=700, xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, title="Rating Aktual"),
                                     yaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, title="Prediksi Model"))
            st.plotly_chart(fig_bubble, use_container_width=True)

        # --- TAB 2: METODOLOGI ---
        with tab2:
            st.subheader("Parameter Indikator")
            method_data = []
            for k, coef in COEFFICIENTS.items():
                det = INDICATOR_META[k]
                method_data.append({
                    'Nama Indikator': det['Name'],
                    'Bobot (Coef)': coef,
                    'Satuan': det['Unit'],
                    'Kategori': det['Type'],
                    'Transformasi': det['Trans']
                })
            st.dataframe(pd.DataFrame(method_data), use_container_width=True, height=700)

        # --- TAB 3: SIMULASI KEBIJAKAN ---
        with tab3:
            st.header(f"🧪 Simulasi Kebijakan: {sel}")
            
            # A. PDF READER
            st.markdown("### 1. Deteksi Sensitivitas (Upload Fitch RAC PDF)")
            uploaded_pdf = st.file_uploader("Upload Report Fitch (PDF)", type="pdf")
            active_constraints = {}
            
            if uploaded_pdf:
                with st.spinner("Menganalisis Dokumen..."):
                    sensitivities_text = parse_fitch_report(uploaded_pdf)
                    active_constraints = map_sensitivities_to_indicators(sensitivities_text)
                
                if active_constraints:
                    st.success(f"Ditemukan {len(active_constraints)} Indikator Sensitif dari dokumen!")
                    with st.expander("Lihat Detail Sensitivitas"):
                        for code, data in active_constraints.items():
                            direction = ", ".join(set(data['direction']))
                            st.markdown(f"**{INDICATOR_META[code]['Name']}** ({direction}):")
                            for txt in data['text']: st.caption(f"- \"{txt}\"")
                else:
                    st.warning("Tidak ditemukan bagian 'Rating Sensitivities' yang valid.")

            st.divider()
            
            # B. SLIDERS
            st.markdown("### 2. Atur Parameter Ekonomi")
            base_raw = res['Raw'].copy()
            sim_raw = {}
            
            def get_label(code):
                base_label = f"{INDICATOR_META[code]['Name']} (Bobot: {COEFFICIENTS[code]})"
                if code in active_constraints:
                    dirs = active_constraints[code]['direction']
                    if 'Negative' in dirs: return f"⚠️ {base_label} [SENSITIVE]"
                    if 'Positive' in dirs: return f"🌟 {base_label} [OPPORTUNITY]"
                return base_label

            c_sim1, c_sim2 = st.columns(2)
            with c_sim1:
                st.subheader("Fiskal & Makro")
                sim_raw['gg_debt'] = st.slider(get_label('gg_debt'), 0.0, 150.0, base_raw['gg_debt'], 0.5)
                sim_raw['fiscal_bal'] = st.slider(get_label('fiscal_bal'), -15.0, 5.0, base_raw['fiscal_bal'], 0.1)
                sim_raw['int_rev'] = st.slider(get_label('int_rev'), 0.0, 60.0, base_raw['int_rev'], 0.5)
                sim_raw['inflation'] = st.slider(get_label('inflation'), 0.0, 30.0, base_raw['inflation'], 0.1)
                sim_raw['real_growth'] = st.slider(get_label('real_growth'), -5.0, 10.0, base_raw['real_growth'], 0.1)
                sim_raw['gdp_volatility'] = st.slider(get_label('gdp_volatility'), 0.8, 10.0, base_raw['gdp_volatility'], 0.1)
                
            with c_sim2:
                st.subheader("Eksternal & Struktural")
                sim_raw['reserves_months'] = st.slider(get_label('reserves_months'), 0.0, 20.0, base_raw['reserves_months'], 0.1)
                sim_raw['fc_debt'] = st.slider(get_label('fc_debt'), 0.0, 100.0, base_raw['fc_debt'], 1.0)
                sim_raw['ca_fdi'] = st.slider(get_label('ca_fdi'), -10.0, 10.0, base_raw['ca_fdi'], 0.1)
                sim_raw['ext_int_service'] = st.slider(get_label('ext_int_service'), 0.0, 20.0, base_raw['ext_int_service'], 0.1)
                sim_raw['wgi'] = st.slider(get_label('wgi'), 0.0, 100.0, base_raw['wgi'], 0.5)
                
                # Hidden sliders
                for k in base_raw:
                    if k not in sim_raw: sim_raw[k] = base_raw[k]

            # C. WARNING SYSTEM & CALCULATION
            alerts = []
            if 'gg_debt' in active_constraints and 'Negative' in active_constraints['gg_debt']['direction']:
                if sim_raw['gg_debt'] > base_raw['gg_debt'] + 2.0:
                    alerts.append(f"⚠️ **PERINGATAN:** Fitch memperingatkan kenaikan Utang Pemerintah.")
            
            if 'reserves_months' in active_constraints and 'Negative' in active_constraints['reserves_months']['direction']:
                 if sim_raw['reserves_months'] < base_raw['reserves_months'] - 1.0:
                    alerts.append(f"⚠️ **PERINGATAN:** Fitch memperingatkan penurunan Cadangan Devisa.")

            for alert in alerts: st.error(alert)

            new_score, _ = calculate_single_score(sim_raw, INTERCEPT, COEFFICIENTS)
            if sim_raw['rc_flex'] < 1.0 and new_score > 12.5: new_score = 12.0
            
            delta = new_score - res['SRM Score']
            new_rating_int = int(round(max(0, min(16, new_score))))
            new_rating_str = NUM_TO_RATING.get(new_rating_int, 'D')

            st.divider()
            c_res1, c_res2 = st.columns([1, 2])
            with c_res1:
                st.metric("Skor Simulasi", f"{new_score:.2f}", f"{delta:+.2f}")
                st.metric("Estimasi Rating", new_rating_str)
            
            with c_res2:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta", value = new_score,
                    delta = {'reference': res['SRM Score']},
                    gauge = {'axis': {'range': [0, 16]}, 'bar': {'color': "darkblue"},
                             'steps': [{'range': [0, 9], 'color': '#ffcccb'}, {'range': [9, 16], 'color': '#90ee90'}]}
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan aplikasi: {e}")