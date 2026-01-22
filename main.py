import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="기온 비교", layout="wide")


# -----------------------------
# UI 스타일 (제목/부제/metric 카드 폰트 & 잘림 방지)
# -----------------------------
st.markdown(
    """
<style>
/* 전체 폭/여백 */
.block-container {
    padding-top: 1.2rem;
    max-width: 1400px;
}

/* 큰 제목 */
.app-title {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 0.2rem 0;
    word-break: keep-all;
}

/* 부제 */
.app-subtitle {
    font-size: 0.95rem;
    color: rgba(0,0,0,0.62);
    margin: 0 0 1.2rem 0;
    word-break: keep-all;
}

/* metric 카드 */
.metric-box {
    background: #fafafa;
    padding: 0.85rem 1rem;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #eee;
    overflow: hidden;
}

.metric-label {
    font-size: 0.85rem;
    color: rgba(0,0,0,0.55);
    margin-bottom: 0.2rem;
    white-space: nowrap;
}

.metric-value {
    font-size: 2.0rem;
    font-weight: 800;
    line-height: 1.15;
    word-break: keep-all;
    overflow-wrap: anywhere;
}

.metric-delta {
    font-size: 0.9rem;
    margin-top: 0.35rem;
    color: #d62728;
    white-space: nowrap;
}

/* 작은 화면에서 글자 자동 축소 */
@media (max-width: 1100px) {
    .metric-value { font-size: 1.6rem; }
}
@media (max-width: 700px) {
    .app-title { font-size: 1.7rem; }
    .metric-value { font-size: 1.3rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">📈 기온 비교 웹앱</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Streamlit + Plotly (업로드 CSV 자동 병합 · 같은 월-일 기준 비교)</div>',
    unsafe_allow_html=True,
)


# -----------------------------
# Plotly 세로선 안전 추가 (환경/버전 TypeError 방지)
# -----------------------------
def add_vline_safe(fig, x, annotation_text=None):
    # pandas.Timestamp -> python datetime 변환
    if hasattr(x, "to_pydatetime"):
        x = x.to_pydatetime()

    try:
        fig.add_vline(x=x, line_dash="dash")
        if annotation_text:
            fig.add_annotation(x=x, y=1, yref="paper", text=annotation_text, showarrow=False)
    except Exception:
        # add_vline이 실패하면 add_shape로 fallback
        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(dash="dash"),
        )
        if annotation_text:
            fig.add_annotation(x=x, y=1, yref="paper", text=annotation_text, showarrow=False)


# -----------------------------
# Parsing helpers (KMA-style CSV export)
# -----------------------------
def _find_header_row(raw: pd.DataFrame) -> int:
    """첫 컬럼에 '날짜'가 등장하는 행을 헤더로 간주."""
    for i in range(min(len(raw), 400)):
        v = raw.iloc[i, 0]
        if isinstance(v, str) and v.strip() == "날짜":
            return i
    raise ValueError("헤더 행(예: '날짜')을 찾지 못했습니다. 업로드 파일 형식을 확인해 주세요.")


def parse_kma_like_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Expected columns (Korean):
      날짜, 지점, 평균기온(℃), 최저기온(℃), 최고기온(℃)

    Returns standardized:
      date, station, tavg, tmin, tmax
    """
    raw = pd.read_csv(io.BytesIO(file_bytes), dtype=str, header=0, encoding="utf-8", engine="python")
    hdr_idx = _find_header_row(raw)

    header = raw.iloc[hdr_idx].tolist()
    df = raw.iloc[hdr_idx + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all")

    # Normalize header spaces
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    required = ["날짜", "지점", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}. 업로드 파일이 샘플과 같은 형식인지 확인해 주세요.")

    # Clean date column
    df["날짜"] = df["날짜"].astype(str).str.replace("\t", "", regex=False).str.strip()
    df["date"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df[df["date"].notna()]

    # Station
    df["station"] = pd.to_numeric(df["지점"], errors="coerce").astype("Int64")

    # Temperatures
    for src, dst in [("평균기온(℃)", "tavg"), ("최저기온(℃)", "tmin"), ("최고기온(℃)", "tmax")]:
        df[dst] = pd.to_numeric(df[src].astype(str).str.strip(), errors="coerce")

    out = df[["date", "station", "tavg", "tmin", "tmax"]].copy()
    out = out.sort_values("date")
    out = out.drop_duplicates(subset=["date", "station"], keep="last").reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def load_base_dataset(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return parse_kma_like_csv(f.read())


def merge_datasets(base: pd.DataFrame, extra_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not extra_frames:
        return base
    merged = pd.concat([base] + extra_frames, ignore_index=True)
    merged = merged.sort_values("date")
    merged = merged.drop_duplicates(subset=["date", "station"], keep="last").reset_index(drop=True)
    return merged


def day_of_year_stats(df: pd.DataFrame, target_dt: pd.Timestamp, metric: str) -> dict:
    """
    Compare target date's metric to distribution of same month-day across all years.
    """
    month = int(target_dt.month)
    day = int(target_dt.day)

    same_md = df[(df["date"].dt.month == month) & (df["date"].dt.day == day)][metric].dropna()
    target_val = df.loc[df["date"] == target_dt, metric].dropna()

    if target_val.empty:
        return {"ok": False, "reason": "선택한 날짜에 값이 없습니다."}
    val = float(target_val.iloc[-1])

    if same_md.empty:
        return {"ok": False, "reason": "같은 월-일의 과거 분포가 없습니다."}

    mean = float(same_md.mean())
    median = float(same_md.median())
    std = float(same_md.std(ddof=0)) if same_md.size > 1 else float("nan")
    diff = val - mean
    z = (diff / std) if std and not np.isnan(std) and std != 0 else float("nan")
    pct_rank = float((same_md < val).mean() * 100.0)

    return {
        "ok": True,
        "val": val,
        "mean": mean,
        "median": median,
        "std": std,
        "diff": diff,
        "z": z,
        "pct_rank": pct_rank,
        "n": int(same_md.size),
    }


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("데이터")
    st.caption("기본 데이터는 저장소 루트의 temp.csv를 사용합니다. 같은 형식 CSV를 업로드하면 자동 병합됩니다.")
    uploaded = st.file_uploader("추가 CSV 업로드 (여러 개 가능)", type=["csv"], accept_multiple_files=True)

    st.divider()
    st.header("비교 날짜")
    use_latest = st.checkbox("최근 데이터 사용", value=True)
    pick = st.date_input("날짜 선택", value=date.today())

    st.divider()
    st.header("비교 지표")
    metric_label = st.selectbox("기온 지표", ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"])
    metric_map = {"평균기온(℃)": "tavg", "최저기온(℃)": "tmin", "최고기온(℃)": "tmax"}
    metric = metric_map[metric_label]


# -----------------------------
# Load + merge
# -----------------------------
BASE_PATH = "temp.csv"  # ✅ 루트에 temp.csv

try:
    base = load_base_dataset(BASE_PATH)
except FileNotFoundError:
    st.error("기본 데이터 파일(temp.csv)을 찾지 못했습니다. 저장소 루트에 temp.csv를 포함해 주세요.")
    st.stop()
except Exception as e:
    st.error(f"기본 데이터 로드/파싱 실패: {e}")
    st.stop()

extras = []
if uploaded:
    for f in uploaded:
        try:
            extras.append(parse_kma_like_csv(f.getvalue()))
        except Exception as e:
            st.warning(f"업로드 파일 '{f.name}' 파싱 실패: {e}")

df = merge_datasets(base, extras)
if df.empty:
    st.error("데이터가 비어 있습니다.")
    st.stop()

# 지점 선택 (여러 지점이면 드롭다운)
stations = df["station"].dropna().unique()
stations = sorted([int(x) for x in stations]) if len(stations) else []
station = None
if stations:
    station = st.sidebar.selectbox("지점 선택", options=stations, index=0)

dff = df[df["station"] == station].copy() if station is not None else df.copy()

# -----------------------------
# Determine target date
# -----------------------------
last_dt = dff["date"].max()
if use_latest:
    target_dt = pd.Timestamp(last_dt.date())
else:
    target_dt = pd.Timestamp(pick)

# 선택한 날짜 데이터가 없으면, 가장 가까운 이전 날짜로 보정
if (dff["date"] == target_dt).sum() == 0:
    prev = dff[dff["date"] <= target_dt]["date"]
    target_dt = prev.max() if not prev.empty else dff["date"].min()

# -----------------------------
# Summary (커스텀 metric 카드)
# -----------------------------
stats = day_of_year_stats(dff, target_dt, metric)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">선택 날짜</div>
            <div class="metric-value">{target_dt.strftime('%Y-%m-%d')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">지점</div>
            <div class="metric-value">{station if station is not None else "N/A"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if stats.get("ok"):
    diff = stats["diff"]
    z = stats["z"]
    z_txt = "—" if np.isnan(z) else f"{z:+.2f}σ"

    with c3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">{metric_label}</div>
                <div class="metric-value">{stats['val']:.1f}℃</div>
                <div class="metric-delta">{diff:+.1f}℃ (평균 대비)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">과거 동일 월-일 대비</div>
                <div class="metric-value">{stats['pct_rank']:.1f}퍼센타일</div>
                <div class="metric-delta">{z_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    with c3:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-label">기온</div>
                <div class="metric-value">N/A</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">비교</div>
                <div class="metric-value">N/A</div>
                <div class="metric-delta">{stats.get("reason","")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption("비교 기준: 선택한 날짜와 같은 **월-일(MM-DD)**의 과거(모든 연도) 분포와 비교합니다.")


# -----------------------------
# Charts
# -----------------------------
left, right = st.columns([1.1, 1.0])

with left:
    st.subheader("① 같은 월-일 과거 분포 vs 선택 날짜")
    month = target_dt.month
    day = target_dt.day
    same_md = dff[(dff["date"].dt.month == month) & (dff["date"].dt.day == day)][["date", metric]].dropna()

    if same_md.empty:
        st.info("해당 월-일의 과거 데이터가 부족합니다.")
    else:
        same_md = same_md.assign(year=same_md["date"].dt.year)
        fig = px.histogram(
            same_md,
            x=metric,
            nbins=40,
            hover_data=["year"],
            title=f"{month:02d}-{day:02d} ({metric_label}) 과거 분포",
        )
        sel_val = float(dff.loc[dff["date"] == target_dt, metric].dropna().iloc[-1])
        add_vline_safe(fig, sel_val, annotation_text=f"선택: {sel_val:.1f}℃")
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("② 최근 30일 추이 (선택 날짜 기준)")
    window = 30
    start = target_dt - pd.Timedelta(days=window)
    end = target_dt + pd.Timedelta(days=1)

    recent = dff[(dff["date"] >= start) & (dff["date"] <= end)][["date", "tavg", "tmin", "tmax"]].copy()

    if recent.empty:
        st.info("최근 구간 데이터가 없습니다.")
    else:
        long = recent.melt(id_vars="date", value_vars=["tavg", "tmin", "tmax"], var_name="metric", value_name="temp")
        label_map = {"tavg": "평균", "tmin": "최저", "tmax": "최고"}
        long["metric"] = long["metric"].map(label_map)

        fig2 = px.line(long, x="date", y="temp", color="metric", markers=True, title="최근 30일 기온 추이")
        add_vline_safe(fig2, target_dt, annotation_text="선택 날짜")
        fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10), legend_title_text="지표")
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("③ 선택 날짜 vs 과거 동일 월-일(연도별) 비교")
same_md2 = dff[(dff["date"].dt.month == target_dt.month) & (dff["date"].dt.day == target_dt.day)][["date", metric]].dropna()

if same_md2.empty:
    st.info("연도별 비교를 위한 데이터가 부족합니다.")
else:
    same_md2 = same_md2.assign(year=same_md2["date"].dt.year).sort_values("year")
    fig3 = px.bar(same_md2, x="year", y=metric, title=f"{target_dt.month:02d}-{target_dt.day:02d} 연도별 {metric_label}")
    add_vline_safe(fig3, int(target_dt.year), annotation_text=f"선택 연도({target_dt.year})")
    fig3.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("데이터 미리보기"):
    st.dataframe(dff.tail(50), use_container_width=True)
