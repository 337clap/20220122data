import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="기온 비교", layout="wide")


# -----------------------------
# Parsing helpers (KMA-style CSV export)
# -----------------------------
def _find_header_row(raw: pd.DataFrame) -> int:
    """Find the row index that contains the real header (e.g., first column == '날짜')."""
    for i in range(min(len(raw), 300)):
        v = raw.iloc[i, 0]
        if isinstance(v, str) and v.strip() == "날짜":
            return i
    raise ValueError("헤더 행(예: '날짜')을 찾지 못했습니다. 업로드 파일 형식을 확인해 주세요.")


def parse_kma_like_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Parses the provided CSV bytes (same format as your sample) into a clean DataFrame:
    columns: ['date', 'station', 'tavg', 'tmin', 'tmax']
    """
    raw = pd.read_csv(io.BytesIO(file_bytes), dtype=str, header=0, encoding="utf-8", engine="python")
    hdr_idx = _find_header_row(raw)

    header = raw.iloc[hdr_idx].tolist()
    df = raw.iloc[hdr_idx + 1:].copy()
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

    # Remove duplicates (keep last)
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
    Compare target date's metric to the distribution of the same month-day across all years.
    Returns dict with value, mean, median, std, diff, z, pct_rank, n.
    """
    month = int(target_dt.month)
    day = int(target_dt.day)

    same_md = df[(df["date"].dt.month == month) & (df["date"].dt.day == day)][metric].dropna()
    target_val = df.loc[df["date"] == target_dt, metric].dropna()

    if target_val.empty:
        return {"ok": False, "reason": "선택한 날짜에 값이 없습니다."}
    val = float(target_val.iloc[-1])

    if same_md.empty:
        return {"ok": False, "reason": "같은 월-일(예: 01-22)에 대한 과거 분포가 없습니다."}

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
# UI
# -----------------------------
st.title("📈 기온 비교 웹앱 (Streamlit + Plotly)")

with st.sidebar:
    st.header("데이터")
    st.caption("기본 데이터(data/base.csv)는 앱에 포함됩니다. 같은 형식 CSV를 업로드하면 자동 병합됩니다.")
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

# Load base dataset
BASE_PATH = "data/base.csv"
try:
    base = load_base_dataset(BASE_PATH)
except FileNotFoundError:
    st.error("기본 데이터 파일(data/base.csv)을 찾지 못했습니다. 저장소에 포함해 주세요.")
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

# Station selection (if multiple)
stations = df["station"].dropna().unique()
stations = sorted([int(x) for x in stations]) if len(stations) else []
station = stations[0] if stations else None

if station is not None:
    dff = df[df["station"] == station].copy()
else:
    dff = df.copy()

# Determine target date
last_dt = dff["date"].max()
if use_latest:
    target_dt = pd.Timestamp(last_dt.date())
else:
    target_dt = pd.Timestamp(pick)

# If chosen date not exists, fallback to nearest previous date
if (dff["date"] == target_dt).sum() == 0:
    prev = dff[dff["date"] <= target_dt]["date"]
    target_dt = prev.max() if not prev.empty else dff["date"].min()

# Summary cards
stats = day_of_year_stats(dff, target_dt, metric)

c1, c2, c3, c4 = st.columns(4)
c1.metric("선택 날짜", target_dt.strftime("%Y-%m-%d"))
c2.metric("지점", str(station) if station is not None else "N/A")

if stats.get("ok"):
    c3.metric(f"{metric_label}", f"{stats['val']:.1f}℃", delta=f"{stats['diff']:+.1f}℃ (평균 대비)")
    z_txt = "—" if np.isnan(stats["z"]) else f"{stats['z']:+.2f}σ"
    c4.metric("과거 동일 월-일 대비", f"{stats['pct_rank']:.1f}퍼센타일", delta=z_txt)
else:
    c3.metric(f"{metric_label}", "N/A")
    c4.metric("비교", stats.get("reason", "N/A"))

st.caption("비교 기준: 선택한 날짜와 같은 **월-일(MM-DD)**의 과거(모든 연도) 분포와 비교합니다.")

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
        fig.add_vline(x=sel_val, line_dash="dash", annotation_text=f"선택: {sel_val:.1f}℃", annotation_position="top")
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
        fig2.add_vline(x=target_dt, line_dash="dash", annotation_text="선택 날짜", annotation_position="top")
        fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10), legend_title_text="지표")
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("③ 선택 날짜 vs 과거 동일 월-일(연도별) 비교")
same_md2 = dff[(dff["date"].dt.month == target_dt.month) & (dff["date"].dt.day == target_dt.day)][["date", metric]].dropna()

if same_md2.empty:
    st.info("연도별 비교를 위한 데이터가 부족합니다.")
else:
    same_md2 = same_md2.assign(year=same_md2["date"].dt.year).sort_values("year")
    fig3 = px.bar(same_md2, x="year", y=metric, title=f"{target_dt.month:02d}-{target_dt.day:02d} 연도별 {metric_label}")
    fig3.add_vline(x=target_dt.year, line_dash="dash", annotation_text=f"선택 연도({target_dt.year})", annotation_position="top")
    fig3.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("데이터 미리보기"):
    st.dataframe(dff.tail(50), use_container_width=True)
