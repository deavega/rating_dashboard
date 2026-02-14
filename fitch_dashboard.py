import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import gdown
import json
from io import BytesIO


# Coba import pypdf
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# --- SECURITY CONFIG ---
def verify_passcode(input_code):
    """Verifikasi passcode dari Streamlit Secrets"""
    try:
        correct_code = st.secrets["passwords"]["database_passcode"]
        return input_code == correct_code
    except:
        st.error("⚠️ Konfigurasi keamanan tidak ditemukan!")
        return False

def load_database():
    """Load database dari Google Drive dan proses seperti upload manual"""
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        return None, None
    
    try:
        import gdown
        import os
        
        # Ambil File ID dari secrets
        file_id = st.secrets["database"]["file_id"]
        
        # Download file temporary
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = "temp_database.xlsx"
        
        with st.spinner("📥 Mengunduh database dari server..."):
            gdown.download(url, output_path, quiet=True)
        
        # Detect extension dan load
        if output_path.endswith('.xlsb'):
            try: 
                df = pd.read_excel(output_path, sheet_name='Data', header=None, engine='pyxlsb')
            except: 
                df = pd.read_excel(output_path, sheet_name=0, header=None, engine='pyxlsb')
        else:
            xl = pd.ExcelFile(output_path)
            sheet_name = next((s for s in xl.sheet_names if 'Data' in s), xl.sheet_names[0])
            df = pd.read_excel(output_path, sheet_name=sheet_name, header=None)
        
        # Extract period dari nama file di Drive (opsional)
        period = "Database Terkini (December 2025)"
        
        # Hapus file temporary
        if os.path.exists(output_path):
            os.remove(output_path)
        
        return df, period
        
    except Exception as e:
        st.error(f"⚠️ Gagal memuat database: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

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
    },
    "2024 (Legacy)": { 
        "intercept": 4.874, 
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

INDICATOR_META = {
    'wgi': {'Name': 'Governance (WGI)', 'Type': 'Structural', 'Unit': 'Index 0-100', 'Trans': 'Linear', 'Keywords': ['governance', 'political stability', 'rule of law']},
    'gdp_pc': {'Name': 'GDP per Capita (PPP)', 'Type': 'Structural', 'Unit': '% of US', 'Trans': 'Linear', 'Keywords': ['income', 'wealth', 'gdp per capita']},
    'world_gdp_share': {'Name': 'World GDP Share', 'Type': 'Structural', 'Unit': '% Share', 'Trans': 'Log', 'Keywords': ['economy size']},
    'default_record': {'Name': 'Years Since Default', 'Type': 'Structural', 'Unit': 'Years', 'Trans': 'Inverse', 'Keywords': ['default history', 'restructuring']},
    'money_supply': {'Name': 'Broad Money', 'Type': 'Structural', 'Unit': '% GDP', 'Trans': 'Log', 'Keywords': ['banking sector', 'financial depth']},
    'gdp_volatility': {'Name': 'Real GDP Volatility', 'Type': 'Macro', 'Unit': 'Std Dev', 'Trans': 'Log_Floor', 'Keywords': ['volatility', 'shock']},
    'inflation': {'Name': 'Inflation (CPI)', 'Type': 'Macro', 'Unit': '%', 'Trans': 'Cap_50', 'Keywords': ['inflation', 'price stability']},
    'real_growth': {'Name': 'Real GDP Growth', 'Type': 'Macro', 'Unit': '%', 'Trans': 'Linear', 'Keywords': ['growth', 'economic activity']},
    'gg_debt': {'Name': 'Govt Debt (% GDP)', 'Type': 'Fiscal', 'Unit': '% GDP', 'Trans': 'Linear', 'Keywords': ['government debt', 'public debt', 'debt burden', 'fiscal consolidation']},
    'int_rev': {'Name': 'Interest/Revenue', 'Type': 'Fiscal', 'Unit': '% Rev', 'Trans': 'Linear', 'Keywords': ['interest payment', 'affordability']},
    'fiscal_bal': {'Name': 'Fiscal Balance', 'Type': 'Fiscal', 'Unit': '% GDP', 'Trans': 'Linear', 'Keywords': ['fiscal deficit', 'budget balance']},
    'fc_debt': {'Name': 'Foreign Currency Debt', 'Type': 'Fiscal', 'Unit': '% Total Debt', 'Trans': 'Linear', 'Keywords': ['foreign currency debt', 'fx exposure']},
    'rc_flex': {'Name': 'Reserve Currency Flex.', 'Type': 'External', 'Unit': 'Index 0-4.6', 'Trans': 'Linear', 'Keywords': ['reserve currency']},
    'snfa': {'Name': 'Net Foreign Assets', 'Type': 'External', 'Unit': '% GDP', 'Trans': 'Linear', 'Keywords': ['net foreign assets', 'sovereign wealth fund']},
    'commodity_dep': {'Name': 'Commodity Dep.', 'Type': 'External', 'Unit': '% CXR', 'Trans': 'Linear', 'Keywords': ['commodity', 'oil', 'resource']},
    'reserves_months': {'Name': 'Reserves (Months)', 'Type': 'External', 'Unit': 'Months CXP', 'Trans': 'Linear', 'Keywords': ['foreign reserves', 'fx reserves', 'liquidity']},
    'ext_int_service': {'Name': 'Ext. Interest Service', 'Type': 'External', 'Unit': '% CXR', 'Trans': 'Linear', 'Keywords': ['external debt service']},
    'ca_fdi': {'Name': 'Current Account + FDI', 'Type': 'External', 'Unit': '% GDP', 'Trans': 'Linear', 'Keywords': ['current account', 'fdi', 'balance of payments']}
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

def parse_fitch_report(pdf_file):
    """Membaca PDF Fitch dan mencari faktor sensitivitas"""
    if PdfReader is None: return {}
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        sensitivities = {'Positive': [], 'Negative': []}
        clean_text = re.sub(r'\s+', ' ', text)
        
        # Regex mencari section "Rating Sensitivities"
        # Pola umum: "Factors that Could, Individually or Collectively, Lead to Negative Rating Action/Downgrade"
        neg_match = re.search(r'Lead to Negative Rating Action/Downgrade:?(.*?)Factors that Could', clean_text, re.IGNORECASE)
        if not neg_match: neg_match = re.search(r'negative rating action(.*?)positive rating action', clean_text, re.IGNORECASE)
        
        if neg_match:
            points = re.split(r' - | – | • ', neg_match.group(1))
            sensitivities['Negative'] = [p.strip() for p in points if len(p) > 10]

        pos_match = re.search(r'Lead to Positive Rating Action/Upgrade:?(.*?)Factors that Could', clean_text, re.IGNORECASE)
        if not pos_match: pos_match = re.search(r'positive rating action(.*?)SOVEREIGN RATING MODEL', clean_text, re.IGNORECASE)

        if pos_match:
            points = re.split(r' - | – | • ', pos_match.group(1))
            sensitivities['Positive'] = [p.strip() for p in points if len(p) > 10]
            
        return sensitivities
    except Exception as e:
        st.error(f"Gagal memparsing PDF: {e}")
        return {}

def map_sensitivities_to_indicators(sensitivities):
    """Mapping teks PDF ke kode indikator"""
    active_constraints = {} 
    for direction, points in sensitivities.items():
        for point in points:
            p_lower = point.lower()
            for code, meta in INDICATOR_META.items():
                for keyword in meta.get('Keywords', []):
                    if keyword in p_lower:
                        if code not in active_constraints:
                            active_constraints[code] = {'direction': [], 'text': []}
                        if direction not in active_constraints[code]['direction']:
                            active_constraints[code]['direction'].append(direction)
                            active_constraints[code]['text'].append(point)
    return active_constraints

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

# --- FUNGSI RENDER QO (DI DALAM KOTAK RAPI) ---
def render_qo_analysis(qo_val, country):
    """Menampilkan analisis QO di dalam kotak berdesain khusus"""
    
    # Tentukan warna dan teks berdasarkan nilai QO
    if qo_val > 0:
        bg_color = "#d4edda"  # Hijau lembut
        border_color = "#28a745"
        text_color = "#155724"
        title_text = f"Upgrade Kualitatif (+{qo_val} Notch): Fundamental Lebih Kuat dari Model"
        content = f"""
        Komite Rating Fitch menilai profil kredit <b>{country}</b> sebenarnya lebih kuat daripada yang dihasilkan oleh perhitungan model (SRM). Faktor pendukung utama biasanya meliputi:
        <ul>
            <li><b>Kekuatan Sektor Perbankan (Structural Features):</b> Sektor perbankan yang memiliki kapitalisasi kuat dan risiko rendah, sehingga meminimalkan risiko kewajiban kontinjensi bagi pemerintah.</li>
            <li><b>Fleksibilitas Pembiayaan (Public Finances):</b> Keberadaan pasar obligasi domestik yang dalam dan likuid. Hal ini mengurangi ketergantungan negara pada utang valuta asing dan melindungi anggaran dari volatilitas nilai tukar.</li>
            <li><b>Kredibilitas Kebijakan (Macroeconomic Policy):</b> Rekam jejak bank sentral yang disiplin dalam menjaga inflasi rendah dan stabilitas makroekonomi jangka panjang.</li>
        </ul>
        """
    elif qo_val < 0:
        bg_color = "#f8d7da"  # Merah lembut
        border_color = "#dc3545"
        text_color = "#721c24"
        title_text = f"Penalti Risiko ({qo_val} Notch): Risiko Tersembunyi Terdeteksi"
        content = f"""
        Komite Rating Fitch menilai profil kredit <b>{country}</b> sebenarnya lebih lemah daripada hasil perhitungan model. Terdapat risiko kualitatif yang membebani rating:
        <ul>
            <li><b>Risiko Politik & Tata Kelola (Structural Features):</b> Ketidakpastian transisi kepemimpinan atau risiko geopolitik yang dapat menghambat kelancaran pembayaran utang.</li>
            <li><b>Kewajiban Kontinjensi (Public Finances):</b> Risiko utang tersembunyi dari BUMN atau kelemahan sektor perbankan yang sewaktu-waktu dapat menjadi beban anggaran negara (risiko bailout).</li>
            <li><b>Kualitas Data (Structural Features):</b> Transparansi data fiskal atau cadangan devisa yang rendah menyebabkan ketidakpastian dalam analisis.</li>
        </ul>
        """
    else:
        bg_color = "#e2e3e5"  # Abu-abu lembut
        border_color = "#6c757d"
        text_color = "#383d41"
        title_text = "Selaras (Neutral): Model & Komite Sepakat"
        content = f"Penilaian kualitatif Komite Rating Fitch untuk <b>{country}</b> sejalan dengan hasil model matematika SRM. Tidak ditemukan distorsi struktural yang memerlukan penyesuaian manual."

    # Render Kotak menggunakan HTML
    st.markdown(f"""
        <div style="
            background-color: {bg_color};
            border: 1px solid {border_color};
            border-left: 8px solid {border_color};
            padding: 20px;
            border-radius: 8px;
            color: {text_color};
            margin-bottom: 25px;
            line-height: 1.6;
        ">
            <h4 style="margin-top: 0; color: {text_color}; font-weight: bold;">{title_text}</h4>
            {content}
        </div>
    """, unsafe_allow_html=True)

    # Expander Edukasi tetap di bawah kotak
    with st.expander("Tentang Qualitative Overlay (QO)"):
        st.markdown("""
        **QO (Qualitative Overlay)** adalah penyesuaian tahap akhir yang dilakukan oleh Komite Rating Fitch terhadap hasil model matematis. 
        Komite dapat menambah atau mengurangi rating berdasarkan faktor kualitatif seperti stabilitas politik, kualitas data, dan fleksibilitas pembiayaan yang tidak tertangkap sepenuhnya oleh rumus mesin.
        """)

# --- 4. UI DASHBOARD ---
st.set_page_config(page_title="Sovereign Credit Watch", layout="wide", page_icon="🌍")

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    selected_version = st.selectbox("Versi Metodologi:", list(SRM_VERSIONS.keys()))
    ACTIVE_PARAMS = SRM_VERSIONS[selected_version]
    COEFFICIENTS = ACTIVE_PARAMS['coeffs']
    INTERCEPT = ACTIVE_PARAMS['intercept']

# Judul dengan Class CSS khusus untuk kontrol jarak
st.markdown('<div class="main-title"><h1>🌍 Sovereign Credit Watch</h1></div>', unsafe_allow_html=True)

# Deskripsi Dashboard
st.markdown("""
<div class="description-box">
    Platform analitik berbasis data yang merujuk pada <b>Fitch Sovereign Rating Model (SRM)</b>. 
    Gunakan dashboard ini untuk memantau indikator makro-fiskal, menganalisis Qualitative Overlay (QO), 
    dan melakukan simulasi dampak kebijakan terhadap rating kredit negara secara presisi.
</div>
""", unsafe_allow_html=True)

st.divider()

# --- CSS KHUSUS (TABS & FILE UPLOADER) ---
# --- FULL CSS CONFIGURATION ---
st.markdown("""
<style>
    /* 1. Spasi di bawah Judul Utama */
    .main-title {
        margin-bottom: 40px !important;
        padding-bottom: 4px;
    }

    /* 2. Memperbesar Font Judul Tabs & Efek Highlight pada Tab Aktif */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px; /* Jarak antar tab diperlebar */
        background-color: #f0f2f6; 
        padding: 10px;
        border-radius: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        /* Hapus height statis agar padding bisa bekerja */
        height: auto !important; 
        padding: 6px 20px !important; /* Jarak atas-bawah dan kiri-kanan */
        border-radius: 12px !important;
        background-color: transparent;
        transition: all 0.3s ease;
        border-bottom: 4px solid transparent !important; /* Border transparan agar layout tidak loncat saat aktif */
    }

    /* Efek untuk Tab yang Sedang Dipilih (Aktif) */
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1) !important; 
        border-bottom: 4px solid #1E3A8A !important; /* Garis bawah lebih tegas */
        transform: translateY(-1px); /* Sedikit lebih tinggi */
    }

    /* Memastikan teks memiliki jarak yang cukup dari border */
    .stTabs [data-baseweb="tab"] p {
        font-size: 1.2rem !important;
        font-weight: bold !important;
        color: #1E3A8A !important;
        margin: 0 !important; /* Hapus margin bawaan p */
        padding-bottom: 4px; /* Jarak tambahan antara teks dan garis bawah */
    }
    /* 3. Memperbesar Label File Uploader */
    /* Memperbesar dan mengubah warna label File Uploader menjadi Hitam */
    [data-testid="stFileUploader"] label p {
        font-size: 1.2rem !important; /* Ukuran disesuaikan agar tidak terlalu raksasa */
        font-weight: 800 !important;
        color: #000000 !important;   /* Mengubah warna menjadi HITAM PEKAT */
        padding-bottom: 5px !important;
    }

    /* Mengubah warna teks instruksi kecil di bawah tombol browse */
    [data-testid="stFileUploader"] section div div {
        color: #333333 !important;
    }
            

    /* 4. Memperbesar Label Pilih Negara */
    [data-testid="stSelectbox"] label p {
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 10px !important;
    }

    /* 5. Custom styling untuk box deskripsi dashboard */
    .description-box {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 20px;
        border-left: 8px solid #1E3A8A;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* 6. Font Tabel Peer */
    [data-testid="stDataFrame"] {
        font-size: 0.9rem !important;
    }

    /* 7. Penyesuaian Subheader */
    h2, h3 {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #333 !important;
    }
            
    /* Style untuk Running Text */
    .rating-marquee {
        background-color: #1E3A8A;
        color: white;
        padding: 10px 0;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- OPSI AKSES DATA ---
st.markdown("### 📂 Pilih Sumber Data")

# ✅ Buat layout 2 kolom: Radio di kiri, Logout di kanan
col_radio, col_logout = st.columns([4, 1])

with col_radio:
    data_source = st.radio(
        "Pilih metode akses data:",
        ["Upload File Manual", "🔒 Akses Database Terkini (Perlu Passcode)"],
        horizontal=True,
        label_visibility="collapsed"
    )

with col_logout:
    # Tampilkan tombol logout hanya jika sudah terautentikasi
    if st.session_state.get('authenticated', False):
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

# ✅ Inisialisasi
uploaded_file = None
df = None

if data_source == "Upload File Manual":
    uploaded_file = st.file_uploader(
        "Unggah file Fitch Comparator (.xlsx / .xlsb)", 
        type=['xlsx', 'xlsb']
    )
    
elif data_source == "🔒 Akses Database Terkini (Perlu Passcode)":
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.form("passcode_form"):
            st.warning("🔐 Akses Terbatas: Database hanya untuk pengguna terotorisasi")
            passcode_input = st.text_input(
                "Masukkan Passcode:", 
                type="password",
                placeholder="Contoh: Dspp2026#"
            )
            submit = st.form_submit_button("🔓 Verifikasi & Akses Database")
            
            if submit:
                if verify_passcode(passcode_input):
                    st.session_state.authenticated = True
                    st.success("✅ Autentikasi berhasil! Memuat database...")
                    st.rerun()
                else:
                    st.error("❌ Passcode salah! Akses ditolak.")
    else:
        # ✅ HAPUS tombol logout dari sini (sudah dipindah ke atas)
        st.success("✅ Terautentikasi | Database Terkini Aktif")
        
        df, _ = load_database()
        if df is None:
            st.stop()

# --- PROSES DATA (Gabungkan logika upload & database) ---
if uploaded_file is not None or df is not None:  # ✅ Tambahkan pengecekan is not None
    try:
        # Jika dari upload, load seperti biasa
        if uploaded_file is not None:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            engine = 'pyxlsb' if file_ext == 'xlsb' else None 
            if file_ext == 'xlsb':
                try: 
                    df = pd.read_excel(uploaded_file, sheet_name='Data', header=None, engine=engine)
                except: 
                    df = pd.read_excel(uploaded_file, sheet_name=0, header=None, engine=engine)
            else:
                xl = pd.ExcelFile(uploaded_file)
                sheet_name = next((s for s in xl.sheet_names if 'Data' in s), xl.sheet_names[0])
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
            period = extract_period_from_filename(uploaded_file.name)

        elif df is not None:  # ← Dari database
            period = "Database Terkini (December 2025)"
            
        else:
            # Jika tidak ada file dan tidak ada database
            st.stop()

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

        # --- DATA LOOP (DIPERBAIKI) ---
        for i in range(11, len(df)):
            r = df.iloc[i]
            if pd.isna(r[C['country']]): continue
            country = str(r[C['country']]).strip()
            
            # -----------------------------------------------------------
            # FILTER BARU: Skip jika nama negara mengandung "[Median]"
            # -----------------------------------------------------------
            if "[Median]" in country: 
                continue 

            try:
                # Raw Extraction
                raw = {}
                
                # --- FIX: LOGIKA YEARS SINCE DEFAULT ---
                inv_def = safe_float(r[C['default']]) # Ambil nilai inverse dari Excel (Kolom CL)
                
                # Jika nilai > 0, hitung tahunnya: (1 / nilai) - 1
                if inv_def > 0.001:
                    raw['default_record'] = (1 / inv_def) - 1
                else:
                    # Jika 0 atau kosong, artinya TIDAK PERNAH DEFAULT (Sangat Aman).
                    # Kita set ke angka tinggi (misal 100 tahun) agar efeknya mendekati 0 di rumus inverse nanti.
                    raw['default_record'] = 100.0 

                # Sisanya tetap sama...
                raw['wgi'] = min(safe_float(r[C['wgi']]), 100.0)
                raw_gni = safe_float(r[C['gdp_pc']])
                raw['gdp_pc'] = (raw_gni / 76000 * 100) if raw_gni > 500 else min(raw_gni, 100.0)
                
                raw_gdp_val = safe_float(r[C['gdp']])
                raw['world_gdp_share'] = (raw_gdp_val / total_gdp * 100) if raw_gdp_val > 0 else 0.001
                
                raw['money_supply'] = max(safe_float(r[C['money']]), 1.0)
                raw['gdp_volatility'] = max(safe_float(r[C['volat']]), 0.8)
                raw['inflation'] = safe_float(r[C['cpi']])
                raw['real_growth'] = safe_float(r[C['growth']])
                raw['gg_debt'] = safe_float(r[C['debt']])
                raw['int_rev'] = safe_float(r[C['int_rev']])
                raw['fiscal_bal'] = safe_float(r[C['bal']])
                
                total_debt_val = safe_float(r[C['debt']])
                lc_debt_val = safe_float(r[C['debt_lc']])
                # Pastikan tidak negatif jika data LC debt sedikit error
                raw['fc_debt'] = max(0.0, ((total_debt_val - lc_debt_val) / total_debt_val * 100)) if total_debt_val > 0 else 0.0
                
                rc = safe_float(r[C['res_curr']])
                raw['rc_flex'] = max(1.0, min(4.6, rc)) if rc > 0 else 0.0
                
                raw['snfa'] = safe_float(r[C['snfa']])
                raw['commodity_dep'] = max(safe_float(r[C['comm']]), 0)
                raw['reserves_months'] = safe_float(r[C['reserves']])
                raw['ext_int_service'] = safe_float(r[C['ext_int']])
                raw['ca_fdi'] = safe_float(r[C['ca_fdi']])

               

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
                    'HDI': safe_float(r[C['hdi']]), 'GDP Nominal': raw_gdp_val, 'Raw': raw
                })
                
            except: continue

        full_df = pd.DataFrame(results)

        full_df = pd.DataFrame(results)
        
        st.markdown(f"### Berdasarkan Data Fitch per **{period}**")

        # --- KODE RUNNING TEXT (MARQUEE) ---
        if 'full_df' in locals() or 'full_df' in globals():
            # Mengambil daftar negara dan rating aktual
            # Pastikan nama kolom sesuai dengan dataframe Anda ('Country' dan 'Actual Rating')
            marquee_data = full_df[['Country', 'Actual Rating']].dropna()
            
            # Menyusun teks: "INDONESIA (BBB)  |  MALAYSIA (BBB+)  |  ..."
            marquee_content = " &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; ".join(
                [f"{row['Country'].upper()}: {row['Actual Rating']}" for _, row in marquee_data.iterrows()]
            )

            # Render Marquee
            st.markdown(f"""
                <div class="rating-marquee">
                    <marquee behavior="scroll" direction="left" scrollamount="6">
                        {marquee_content}
                    </marquee>
                </div>
            """, unsafe_allow_html=True)

        # --- TAB INIT ---
        tab1, tab2, tab3, tab4 = st.tabs(["Analisis Negara", "Metodologi", "Simulasi Kebijakan", "Komparasi Indikator"])

        # --- GLOBAL SELECTOR (Tab 1 Only) ---
        with tab1:
            sel = st.selectbox("Pilih Negara:", full_df['Country'].unique(), index=list(full_df['Country']).index('Indonesia') if 'Indonesia' in full_df['Country'].values else 0)
            res = full_df[full_df['Country'] == sel].iloc[0]

            
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
                hm_rows = []
                
                # --- LOGIKA ARAH PERBAIKAN (IMPROVEMENT DIRECTION) ---
                # 1 = Lebih Tinggi Lebih Baik
                # -1 = Lebih Rendah Lebih Baik
                DIRECTION_MAP = {
                    'wgi': 1, 'gdp_pc': 1, 'world_gdp_share': 1, 'default_record': 1, # FIX: Years harusnya 1 (Makin lama makin baik)
                    'money_supply': 1, 'gdp_volatility': -1, 'inflation': -1, 'real_growth': 1, 
                    'gg_debt': -1, 'int_rev': -1, 'fiscal_bal': 1, 'fc_debt': -1, 
                    'rc_flex': 1, 'snfa': 1, 'commodity_dep': -1, 'reserves_months': 1, 
                    'ext_int_service': -1, 'ca_fdi': 1
                }

                for k in COEFFICIENTS.keys():
                    val = res['Raw'][k]
                    peer_vals = [p['Raw'][k] for i, p in peers.iterrows()]
                    avg = np.mean(peer_vals) if peer_vals else val
                    
                    # Logika Penentuan Status Baru
                    direction = DIRECTION_MAP.get(k, 1)
                    
                    if direction == 1:
                        status = 'Better' if val > avg else 'Worse'
                    else:
                        status = 'Better' if val < avg else 'Worse'

                    hm_rows.append({
                        'Indikator': INDICATOR_META[k]['Name'], 
                        'Nilai': val, 
                        'Rerata Peer': avg, 
                        'Status': status
                    })
                
                st.dataframe(
                    pd.DataFrame(hm_rows).style.apply(lambda x: [f'background-color: #d4edda' if x['Status']=='Better' else f'background-color: #f8d7da']*4, axis=1), 
                    height=700, 
                    use_container_width=True
                )
            
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
            st.subheader("🌐 Peta Kuadran Rating Global")
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
                meta = INDICATOR_META[k]
                method_data.append({
                    'Kode': k, 'Indikator': meta['Name'], 'Kategori': meta['Type'], 'Satuan': meta['Unit'], 'Koefisien': coef, 'Transformasi': meta['Trans']
                })
            st.dataframe(pd.DataFrame(method_data), use_container_width=True, height=600)

        # --- TAB 3: SIMULASI KEBIJAKAN (FIXED) ---
        # --- TAB 3: SIMULASI KEBIJAKAN (FIXED & DYNAMIC) ---
        with tab3:
            st.subheader(f"Simulasi Kebijakan untuk Negara: {sel}")
            st.caption(f"Data awal indikator disesuaikan otomatis berdasarkan profil terkini: **{sel}**.")
            
            # 1. UPLOAD PDF FITCH
            pdf_file = st.file_uploader("Upload Fitch Report (PDF) untuk deteksi sensitivitas", type=['pdf'])
            
            active_constraints = {}
            if pdf_file and 'PdfReader' in locals():
                sensitivities = parse_fitch_report(pdf_file)
                active_constraints = map_sensitivities_to_indicators(sensitivities)
                if active_constraints:
                    st.success(f"PDF Terbaca! Ditemukan {len(active_constraints)} indikator sensitif.")
            
            # 2. INPUT PARAMETER EKONOMI
            # base_raw mengambil data 'res' yang sudah terfilter dinamis di Tab 1
            base_raw = res['Raw']
            custom_values = {} # Dictionary untuk menyimpan nilai simulasi baru
            sim_cols = st.columns(3)
            
            for idx, var_code in enumerate(COEFFICIENTS.keys()):
                meta = INDICATOR_META[var_code]
                default_val = safe_float(base_raw.get(var_code, 0))
                
                # Cek Warning/Opportunity dari PDF
                label_extra = ""
                help_text = ""
                if var_code in active_constraints:
                    dirs = active_constraints[var_code]['direction']
                    if 'Negative' in dirs: 
                        label_extra += " 📉 (Risk)"
                        help_text += f"⚠️ RISIKO: {active_constraints[var_code]['text'][0][:150]}..."
                    if 'Positive' in dirs: 
                        label_extra += " 📈 (Upside)"
                        help_text += f"🌟 PELUANG: {active_constraints[var_code]['text'][0][:150]}..."

                step = 0.1
                with sim_cols[idx % 3]:
                    # FIX: Menambahkan {sel} pada key agar otomatis reset saat ganti negara
                    new_val = st.number_input(
                        f"{meta['Name']} ({meta['Unit']}){label_extra}", 
                        value=float(default_val), 
                        step=step, 
                        key=f"sim_{sel}_{var_code}",
                        help=help_text if help_text else f"Bobot Model: {COEFFICIENTS[var_code]}"
                    )
                    custom_values[var_code] = new_val # Simpan ke custom_values
            
            # 3. ANALISIS RISIKO (MENGGUNAKAN custom_values)
            if active_constraints:
                st.divider()
                st.subheader("🛡️ Monitor Sensitivitas Rating")
                
                # Mapping Arah Risiko: 1 = Bahaya jika NAIK, -1 = Bahaya jika TURUN
                RISK_DIRECTION = {
                    'wgi': -1, 'gdp_pc': -1, 'world_gdp_share': -1, 'default_record': -1, 
                    'money_supply': -1, 'gdp_volatility': 1, 'inflation': 1, 'real_growth': -1, 
                    'gg_debt': 1, 'int_rev': 1, 'fiscal_bal': -1, 'fc_debt': 1, 
                    'rc_flex': -1, 'snfa': -1, 'commodity_dep': 1, 'reserves_months': -1, 
                    'ext_int_service': 1, 'ca_fdi': -1
                }

                found_risk = False
                
                for code, data in active_constraints.items():
                    if 'Negative' in data['direction']:
                        current_sim_val = float(custom_values.get(code, 0))
                        base_val = float(base_raw.get(code, 0))
                        meta = INDICATOR_META[code]
                        direction_rule = RISK_DIRECTION.get(code, 0)
                        
                        is_worsening = False
                        
                        # Logika Deteksi
                        if direction_rule == 1 and current_sim_val > base_val: is_worsening = True
                        elif direction_rule == -1 and current_sim_val < base_val: is_worsening = True
                        
                        if is_worsening:
                            found_risk = True
                            delta = current_sim_val - base_val
                            arrow = "⬆️ Naik" if delta > 0 else "⬇️ Turun"
                            
                            st.error(
                                f"🚨 **PERINGATAN RISIKO:** Simulasi Anda membuat **{meta['Name']}** {arrow} "
                                f"({base_val:.2f} → {current_sim_val:.2f})."
                            )
                            with st.chat_message("user", avatar="📄"):
                                st.write(f"**Kutipan Laporan Fitch:**")
                                for t in data['text']: st.caption(f"\"...{t}...\"")
                
                # Cek apakah ada perubahan nilai sama sekali
                has_changes = any(custom_values[k] != base_raw[k] for k in custom_values)

                if not found_risk:
                    if has_changes:
                        # Jika sudah ada perubahan tapi aman
                        st.success("✅ **Simulasi Aman:** Perubahan yang Anda lakukan tidak melanggar batas sensitivitas negatif yang terdeteksi di dokumen Fitch.")
                    else:
                        # Jika belum ada perubahan sama sekali (Default State)
                        st.info("ℹ️ **Menunggu Simulasi:** Silakan ubah angka pada input di atas. Sistem akan otomatis memperingatkan jika perubahan Anda berlawanan dengan *guidance* Fitch.")

            # 4. HASIL KALKULASI AKHIR
            st.divider()
            
            # Hitung skor baru
            final_vals = custom_values if custom_values else base_raw
            new_score, _ = calculate_single_score(final_vals, INTERCEPT, COEFFICIENTS)
            
            # Logic Cap EM
            if final_vals.get('rc_flex', 0) < 1.0 and new_score > 12.5: 
                new_score = 12.0
                
            new_rating_int = int(round(max(0, min(16, new_score))))
            new_rating_str = NUM_TO_RATING.get(new_rating_int, 'D')
            
            # Layout Kolom: Metrics di kiri, Gauge di kanan
            c_sim1, c_sim2, c_sim3 = st.columns([1, 1, 1.3])
            
            with c_sim1:
                st.markdown("##### Skor Baru")
                st.metric(
                    label="SRM Score", 
                    value=f"{new_score:.2f}", 
                    delta=f"{new_score - res['SRM Score']:.2f}",
                    label_visibility="collapsed"
                )
            
            with c_sim2:
                st.markdown("##### Rating Baru")
                st.metric(
                    label="Rating", 
                    value=new_rating_str, 
                    delta=f"{new_rating_int - res['Pred Rating Num']} Notch",
                    label_visibility="collapsed"
                )
            
            with c_sim3:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta", 
                    value = new_score,
                    delta = {'reference': res['SRM Score'], 'position': "bottom", 'relative': False},
                    number = {'font': {'size': 24, 'color': "black", 'family': "Arial"}}, 
                    gauge = {
                        'axis': {'range': [0, 16], 'tickwidth': 1, 'tickcolor': "gray"},
                        'bar': {'color': "#2E86C1", 'thickness': 0.8}, 
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "#cccccc",
                        'steps': [
                            {'range': [0, 9], 'color': '#ffcccb'}, 
                            {'range': [9, 16], 'color': '#d4edda'} 
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.8,
                            'value': new_score
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    height=170, 
                    margin=dict(l=35, r=35, t=40, b=10), 
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': "Arial", 'color': "black"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

        with tab4:
            st.subheader("Komparasi Indikator: Benchmarking")
            
            # 1. Filter Negara Pembanding
            available_peers = [p for p in full_df['Country'].unique() if p != sel]
            
            selected_peers = st.multiselect(
                f"Bandingkan {sel} dengan (Pilih hingga 5 negara):", 
                options=sorted(available_peers), 
                max_selections=5,
                key="peer_comp_universal_key"
            )

            if selected_peers:
                all_selected = [sel] + selected_peers
                comp_rows = []
                
                # 2. Ambil data dari full_df yang sudah diproses
                for code in INDICATOR_META.keys():
                    meta = INDICATOR_META[code]
                    row_data = {"Indikator": meta['Name']}
                    
                    for country in all_selected:
                        # Ambil nilai dari kolom 'Raw' di full_df
                        country_row = full_df[full_df['Country'] == country]
                        if not country_row.empty:
                            raw_dict = country_row.iloc[0]['Raw']
                            row_data[country] = safe_float(raw_dict.get(code, 0))
                        else:
                            row_data[country] = 0.0
                            
                    comp_rows.append(row_data)
                
                df_comp = pd.DataFrame(comp_rows)

                # 3. Logika Warna (Sama seperti sebelumnya)
                IMPROVEMENT_DIRECTION = {
                    'wgi': 1, 'gdp_pc': 1, 'world_gdp_share': 1, 'default_record': 1, 
                    'money_supply': 1, 'gdp_volatility': -1, 'inflation': -1, 'real_growth': 1, 
                    'gg_debt': -1, 'int_rev': -1, 'fiscal_bal': 1, 'fc_debt': -1, 
                    'rc_flex': 1, 'snfa': 1, 'commodity_dep': -1, 'reserves_months': 1, 
                    'ext_int_service': -1, 'ca_fdi': 1
                }

                def style_comparison(row):
                    styles = [''] * len(row)
                    if sel not in row.index: return styles
                    
                    anchor_val = row[sel]
                    code = list(INDICATOR_META.keys())[row.name]
                    direction = IMPROVEMENT_DIRECTION.get(code, 1)

                    for i, col_name in enumerate(row.index):
                        if col_name in ['Indikator', sel]: continue
                        peer_val = row[col_name]
                        
                        if direction == 1:  # High is better
                            if anchor_val > peer_val: 
                                styles[i] = 'background-color: #d4edda; color: #155724; font-weight: bold;'
                            elif anchor_val < peer_val: 
                                styles[i] = 'background-color: #f8d7da; color: #721c24;'
                        else:  # Low is better
                            if anchor_val < peer_val: 
                                styles[i] = 'background-color: #d4edda; color: #155724; font-weight: bold;'
                            elif anchor_val > peer_val: 
                                styles[i] = 'background-color: #f8d7da; color: #721c24;'
                    return styles

                # 4. Render Tabel
                st.write(f"### Tabel Komparasi: {sel} vs Peers")
                st.dataframe(
                    df_comp.style.apply(style_comparison, axis=1).format(precision=2, subset=all_selected),
                    use_container_width=True, 
                    height=600
                )
                
                st.info(f"💡 **Hijau** = {sel} lebih baik | **Merah** = {sel} lebih buruk")
            else:
                st.info("💡 Pilih negara pembanding dari daftar di atas.")

    except Exception as e:
        st.error(f"❌ Error saat memproses data: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("👋 Silakan unggah file Excel untuk memulai.")