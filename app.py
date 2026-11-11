"""NIRF Ranking Insight Dashboard - Python/Streamlit edition.

Run: streamlit run app.py
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="NIRF Insight Dashboard", page_icon="📊", layout="wide")

DATA_FILE = Path(__file__).with_name("nirf_raw.csv")
PARAMETERS = ["TLR", "RPC", "GO", "OI", "PR"]
METRICS = {"Overall score": "overall_score", "Rank": "rank", **{p: p for p in PARAMETERS}}

QUESTIONS = {
    "Time series and trend analysis": [
        "Institution rank trend", "Greatest rank improvement", "Largest rank decline",
        "Average score over time", "Consistent top-10 institutions",
    ],
    "Geospatial and regional analysis": [
        "States with most top-100 institutions", "Top-100 representation by state over time",
        "States improving top-100 representation", "Compare two states by rank", "New institutions by state",
    ],
    "Univariate analysis": [
        "Score distribution", "Rank distribution", "Score summary statistics", "Score quartiles", "Score spread by category",
    ],
    "Bivariate analysis and correlation": [
        "RPC versus overall score", "PR versus overall score", "Strongest parameter correlation",
        "TLR versus RPC", "RPC/overall correlation trend",
    ],
    "Multivariate analysis": [
        "Top five parameter heatmap", "High research / low teaching", "High teaching / low perception",
        "Improved score despite parameter decline", "Largest parameter gap",
    ],
    "Distribution and statistical comparison": [
        "Top-10 versus remaining distribution", "Score dispersion by year", "Most variable parameter",
        "Median and IQR over time", "State score spread",
    ],
    "Ranking transitions and mobility": [
        "51-100 to top-50 transitions", "Top-10 entries and exits", "Rank-band retention",
        "Largest rank movements", "Moved up/down/unchanged",
    ],
    "Outlier detection and performance gaps": [
        "High research relative to overall", "Research versus teaching outliers", "Large score change",
        "Strong parameter with poor rank", "Ranking volatility",
    ],
}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load, standardize, validate, and deduplicate supplied NIRF records."""
    raw = pd.read_csv(path)
    aliases = {
        "category_original": "category", "institute": "institution", "institute_name": "institution",
        "institution_name": "institution", "score": "overall_score", "overall": "overall_score",
        "rank_raw": "rank", "ranking_year": "year", "ranking_category": "category",
    }
    df = raw.rename(columns=aliases).copy()
    for col in ("year", "rank", "overall_score", *PARAMETERS):
        if col in df:
            df[col] = pd.to_numeric(df[col].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")
    for col in ("category", "institution", "state", "city"):
        df[col] = df.get(col, "Unknown").fillna("Unknown").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["category"] = df["category"].replace({"S T A T E P U B L I C U N I V E R S I T Y": "State Public University"})
    df = df.dropna(subset=["year", "institution"]).drop_duplicates(["year", "category", "institution"], keep="first")
    df["year"] = df["year"].astype(int)
    return df


def rank_band(rank: float) -> str:
    if rank <= 10:
        return "Top 10"
    if rank <= 50:
        return "11-50"
    if rank <= 100:
        return "51-100"
    return "100+"


def paired_years(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    # An institution can appear once per category in a given year. Aggregate
    # first so the comparison always has exactly one record per institution.
    yearly = (
        df.groupby(["year", "institution"], as_index=False)
        .agg(rank=("rank", "min"), overall_score=("overall_score", "mean"))
    )
    left = yearly[yearly.year.eq(start)].set_index("institution")
    right = yearly[yearly.year.eq(end)].set_index("institution")
    common = left.index.intersection(right.index)
    result = pd.DataFrame({
        "institution": common,
        "rank_start": left.loc[common, "rank"].to_numpy(), "rank_end": right.loc[common, "rank"].to_numpy(),
        "score_start": left.loc[common, "overall_score"].to_numpy(), "score_end": right.loc[common, "overall_score"].to_numpy(),
    })
    return result.dropna(subset=["rank_start", "rank_end"])


def render_question(question: str, selected: pd.DataFrame, base: pd.DataFrame, analysis_year: int, start: int, end: int, institution: str, state_b: str):
    """Render one of the 40 Annexure A analyses."""
    year_rows = selected[selected.year.eq(analysis_year)].copy()
    pair = paired_years(selected, start, end)
    st.subheader(question)

    if question == "Institution rank trend":
        chart = base[base.institution.eq(institution)].sort_values("year")
        fig = px.line(chart, x="year", y="rank", markers=True, title=f"Rank trend: {institution}")
        fig.update_yaxes(autorange="reversed", title="Rank (lower is better)")
        st.plotly_chart(fig, use_container_width=True)
    elif question in {"Greatest rank improvement", "Largest rank decline", "Largest rank movements"}:
        pair["rank_change"] = pair.rank_start - pair.rank_end
        values = pair.sort_values("rank_change", ascending=question == "Largest rank decline")
        if question == "Largest rank movements": values = pair.reindex(pair.rank_change.abs().sort_values(ascending=False).index)
        values = values.head(15)
        st.plotly_chart(px.bar(values, x="rank_change", y="institution", orientation="h", title=f"Rank change, {start}-{end}"), use_container_width=True)
    elif question == "Average score over time":
        values = selected.groupby("year", as_index=False).overall_score.mean()
        st.plotly_chart(px.line(values, x="year", y="overall_score", markers=True, title="Average overall score"), use_container_width=True)
    elif question == "Consistent top-10 institutions":
        values = selected[selected.rank.le(10)].groupby("institution", as_index=False).size().sort_values("size", ascending=False).head(20)
        st.plotly_chart(px.bar(values, x="size", y="institution", orientation="h", title="Years in the top 10"), use_container_width=True)
    elif question in {"States with most top-100 institutions", "States improving top-100 representation", "New institutions by state"}:
        if question == "States with most top-100 institutions":
            values = year_rows[year_rows.rank.le(100)].groupby("state", as_index=False).institution.nunique().sort_values("institution", ascending=False).head(20)
            title, x = f"Top-100 institutions by state ({analysis_year})", "institution"
        elif question == "States improving top-100 representation":
            a = selected[(selected.year == start) & (selected.rank <= 100)].groupby("state").institution.nunique()
            b = selected[(selected.year == end) & (selected.rank <= 100)].groupby("state").institution.nunique()
            values = b.sub(a, fill_value=0).rename("change").reset_index().sort_values("change", ascending=False); title, x = f"Change in top-100 representation ({start}-{end})", "change"
        else:
            before = set(selected[selected.year.eq(start)].institution)
            values = selected[(selected.year.eq(end)) & ~selected.institution.isin(before)].groupby("state", as_index=False).institution.nunique().sort_values("institution", ascending=False); title, x = f"New institutions by state ({start}-{end})", "institution"
        st.plotly_chart(px.bar(values.head(20), x=x, y="state", orientation="h", title=title), use_container_width=True)
    elif question == "Top-100 representation by state over time":
        values = selected[selected.rank.le(100)].groupby(["year", "state"], as_index=False).institution.nunique()
        leading = values.groupby("state").institution.sum().nlargest(5).index
        st.plotly_chart(px.line(values[values.state.isin(leading)], x="year", y="institution", color="state", markers=True, title="Top-100 representation over time"), use_container_width=True)
    elif question == "Compare two states by rank":
        states = [s for s in [state_b, st.session_state.get("primary_state", "All")] if s != "All"]
        if len(states) < 2: states = ["Karnataka", "Tamil Nadu"]
        values = base[base.state.isin(states)].groupby(["year", "state"], as_index=False).rank.mean()
        fig = px.line(values, x="year", y="rank", color="state", markers=True, title="Average rank by state")
        fig.update_yaxes(autorange="reversed"); st.plotly_chart(fig, use_container_width=True)
    elif question in {"Score distribution", "Rank distribution"}:
        field = "overall_score" if question == "Score distribution" else "rank"
        st.plotly_chart(px.histogram(year_rows, x=field, nbins=20, title=f"{question} ({analysis_year})"), use_container_width=True)
    elif question == "Score summary statistics":
        st.dataframe(year_rows.overall_score.describe()[["mean", "50%", "min", "max"]].rename({"50%": "median"}).to_frame("Overall score"))
    elif question == "Score quartiles":
        values = pd.qcut(year_rows.overall_score, 4, duplicates="drop").value_counts(sort=False).rename("institutions").reset_index(names="quartile")
        st.plotly_chart(px.bar(values, x="quartile", y="institutions", title="Score quartiles"), use_container_width=True)
    elif question == "Score spread by category":
        values = base[base.year.eq(analysis_year)].groupby("category", as_index=False).overall_score.std().rename(columns={"overall_score": "standard deviation"})
        st.plotly_chart(px.bar(values.sort_values("standard deviation", ascending=False), x="category", y="standard deviation", title="Score spread by category"), use_container_width=True)
    elif question in {"RPC versus overall score", "PR versus overall score", "TLR versus RPC"}:
        x, y = {"RPC versus overall score": ("RPC", "overall_score"), "PR versus overall score": ("PR", "overall_score"), "TLR versus RPC": ("TLR", "RPC")}[question]
        st.plotly_chart(px.scatter(year_rows, x=x, y=y, hover_name="institution", trendline="ols", title=question), use_container_width=True)
    elif question == "Strongest parameter correlation":
        values = year_rows[PARAMETERS + ["overall_score"]].corr(numeric_only=True).overall_score.drop("overall_score").abs().sort_values(ascending=False).rename("correlation").reset_index(names="parameter")
        st.plotly_chart(px.bar(values, x="parameter", y="correlation", title="Parameter correlation with overall score"), use_container_width=True)
    elif question == "RPC/overall correlation trend":
        values = selected.groupby("year").apply(lambda x: x.RPC.corr(x.overall_score), include_groups=False).rename("correlation").reset_index()
        st.plotly_chart(px.line(values, x="year", y="correlation", markers=True, title="RPC/overall correlation over time"), use_container_width=True)
    elif question == "Top five parameter heatmap":
        values = year_rows.nsmallest(5, "rank").set_index("institution")[PARAMETERS]
        st.plotly_chart(px.imshow(values, text_auto=".1f", color_continuous_scale="Teal", title="Top five: five-parameter comparison"), use_container_width=True)
    elif question in {"High research / low teaching", "Research versus teaching outliers", "High teaching / low perception", "Largest parameter gap", "High research relative to overall", "Strong parameter with poor rank"}:
        if question in {"High research / low teaching", "Research versus teaching outliers"}: values, label = year_rows.assign(gap=year_rows.RPC-year_rows.TLR), "RPC - TLR"
        elif question == "High teaching / low perception": values, label = year_rows.assign(gap=year_rows.TLR-year_rows.PR), "TLR - PR"
        elif question == "Largest parameter gap": values, label = year_rows.assign(gap=year_rows[PARAMETERS].max(axis=1)-year_rows[PARAMETERS].min(axis=1)), "Parameter gap"
        elif question == "High research relative to overall": values, label = year_rows.assign(gap=year_rows.RPC-year_rows.overall_score), "RPC - overall score"
        else: values, label = year_rows.assign(gap=year_rows[PARAMETERS].max(axis=1).rank(pct=True)+year_rows.rank.rank(pct=True)), "Strong score + poor rank index"
        values = values.nlargest(15, "gap")
        st.plotly_chart(px.bar(values, x="gap", y="institution", orientation="h", title=label), use_container_width=True)
    elif question == "Improved score despite parameter decline":
        old = selected[selected.year.eq(start)].set_index("institution"); new = selected[selected.year.eq(end)].set_index("institution"); common = old.index.intersection(new.index)
        values = pd.DataFrame({"institution": common, "score_change": new.loc[common, "overall_score"].to_numpy()-old.loc[common, "overall_score"].to_numpy(), "parameter_declined": (new.loc[common, PARAMETERS].to_numpy() < old.loc[common, PARAMETERS].to_numpy()).any(axis=1)})
        st.dataframe(values[(values.score_change > 0) & values.parameter_declined].sort_values("score_change", ascending=False).head(20), use_container_width=True)
    elif question in {"Top-10 versus remaining distribution", "Score dispersion by year", "Most variable parameter", "Median and IQR over time", "State score spread"}:
        if question == "Top-10 versus remaining distribution":
            values = year_rows.assign(group=np.where(year_rows.rank <= 10, "Top 10", "Remaining")); st.plotly_chart(px.box(values, x="group", y="overall_score", points="all", title="Score distribution: top 10 versus remaining"), use_container_width=True)
        elif question == "Score dispersion by year":
            values = selected.groupby("year", as_index=False).overall_score.std().rename(columns={"overall_score": "standard deviation"}); st.plotly_chart(px.line(values, x="year", y="standard deviation", markers=True, title="Score dispersion"), use_container_width=True)
        elif question == "Most variable parameter":
            values = year_rows[PARAMETERS].std().rename("standard deviation").reset_index(names="parameter"); st.plotly_chart(px.bar(values, x="parameter", y="standard deviation", title="Parameter variability"), use_container_width=True)
        elif question == "Median and IQR over time":
            values = selected.groupby("year").overall_score.agg(median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75)).reset_index(); values["IQR"] = values.q3-values.q1
            st.plotly_chart(px.line(values.melt("year", value_vars=["median", "IQR"], var_name="measure", value_name="score"), x="year", y="score", color="measure", markers=True, title="Median and IQR over time"), use_container_width=True)
        else:
            st.plotly_chart(px.box(year_rows, x="state", y="overall_score", points=False, title="Score distributions by state"), use_container_width=True)
    elif question in {"51-100 to top-50 transitions", "Top-10 entries and exits", "Rank-band retention", "Moved up/down/unchanged"}:
        if question == "51-100 to top-50 transitions": values = pd.DataFrame({"count": [((pair.rank_start.between(51, 100)) & (pair.rank_end <= 50)).sum()]}, index=["51-100 → Top 50"])
        elif question == "Top-10 entries and exits": values = pd.DataFrame({"count": [((pair.rank_start > 10) & (pair.rank_end <= 10)).sum(), ((pair.rank_start <= 10) & (pair.rank_end > 10)).sum()]}, index=["Entered top 10", "Exited top 10"])
        elif question == "Rank-band retention": values = pd.DataFrame({"count": [pair.apply(lambda x: rank_band(x.rank_start) == rank_band(x.rank_end), axis=1).mean()*100]}, index=["Same rank band (%)"])
        else: values = pd.DataFrame({"count": [(pair.rank_end < pair.rank_start).sum(), (pair.rank_end > pair.rank_start).sum(), (pair.rank_end == pair.rank_start).sum()]}, index=["Moved up", "Moved down", "Unchanged"])
        st.plotly_chart(px.bar(values.reset_index(names="outcome"), x="outcome", y="count", title=f"Ranking mobility: {start}-{end}"), use_container_width=True)
    elif question == "Large score change":
        values = paired_years(selected, start, end).assign(score_change=lambda x: (x.score_end-x.score_start).abs()).nlargest(15, "score_change")
        st.plotly_chart(px.bar(values, x="score_change", y="institution", orientation="h", title=f"Largest score change ({start}-{end})"), use_container_width=True)
    elif question == "Ranking volatility":
        values = selected.groupby("institution", as_index=False).rank.std().rename(columns={"rank": "rank volatility"}).nlargest(15, "rank volatility")
        st.plotly_chart(px.bar(values, x="rank volatility", y="institution", orientation="h", title="Ranking volatility"), use_container_width=True)


df = load_data(DATA_FILE)
years = sorted(df.year.unique())

st.title("NIRF Ranking Insight Dashboard")
st.caption("Python / Streamlit dashboard for NIRF ranking analysis, 2017-2025")

with st.sidebar:
    st.header("Filters")
    category = st.selectbox("Category", ["All", *sorted(df.category.unique())])
    primary_state = st.selectbox("State", ["All", *sorted(df.state.unique())], key="primary_state")
    analysis_year = st.selectbox("Analysis year", years, index=len(years)-1)
    start, end = st.select_slider("Comparison years", options=years, value=(years[0], years[-1]))
    institution = st.selectbox("Institution", sorted(df.institution.unique()))
    state_b = st.selectbox("Comparison state", sorted(df.state.unique()), index=min(1, len(df.state.unique())-1))

selected = df.copy()
if category != "All": selected = selected[selected.category.eq(category)]
if primary_state != "All": selected = selected[selected.state.eq(primary_state)]
base = df if category == "All" else df[df.category.eq(category)]

st.info(f"Loaded {len(df):,} cleaned NIRF records across {df.year.min()}-{df.year.max()}. Rank is interpreted as lower-is-better.")
metrics = st.columns(4)
metrics[0].metric("Records in view", f"{len(selected):,}")
metrics[1].metric("Institutions", f"{selected.institution.nunique():,}")
metrics[2].metric("Categories", f"{selected.category.nunique()}")
metrics[3].metric("Comparison", f"{start}-{end}")

st.markdown("## Dashboard levels")
level_one, level_two, level_three = st.columns(3)
level_one.info("**Level 1**\n\n40 Annexure A sample analyses")
level_two.success("**Level 2**\n\nBuild a chart for an unseen evaluation question")
level_three.warning("**Level 3**\n\nAsk a natural-language question about the data")
st.caption("All three levels are available below. Select a tab to open that level.")

tab_questions, tab_builder, tab_query = st.tabs(["📈 Level 1 · 40 sample analyses", "🛠️ Level 2 · Dynamic analysis builder", "💬 Level 3 · Natural-language query"])
with tab_questions:
    section = st.selectbox("Analysis group", list(QUESTIONS))
    question = st.selectbox("Sample question", QUESTIONS[section])
    render_question(question, selected, base, analysis_year, start, end, institution, state_b)

with tab_builder:
    left, mid, right, chart_col = st.columns(4)
    metric_label = left.selectbox("Measure", [*METRICS, "Institution count"])
    group = mid.selectbox("Group by", ["year", "state", "institution", "category"])
    aggregation = right.selectbox("Aggregation", ["mean", "median", "sum", "count"])
    chart_kind = chart_col.selectbox("Chart", ["bar", "line", "histogram", "scatter"])
    metric = METRICS.get(metric_label)
    builder_rows = selected if group == "year" else selected[selected.year.eq(analysis_year)]
    if chart_kind == "histogram":
        field = metric or "rank"; st.plotly_chart(px.histogram(builder_rows, x=field, title=f"{metric_label} distribution"), use_container_width=True)
    elif chart_kind == "scatter":
        field = metric or "rank"; st.plotly_chart(px.scatter(builder_rows, x=field, y="overall_score", hover_name="institution", title=f"{metric_label} versus overall score"), use_container_width=True)
    else:
        values = builder_rows.groupby(group).size().reset_index(name="value") if metric is None else builder_rows.groupby(group, as_index=False)[metric].agg(aggregation).rename(columns={metric: "value"})
        values = values.sort_values("value", ascending=metric_label == "Rank").head(20)
        fig = px.line(values, x=group, y="value", markers=True, title=f"{aggregation.title()} {metric_label} by {group}") if chart_kind == "line" else px.bar(values, x=group, y="value", title=f"{aggregation.title()} {metric_label} by {group}")
        st.plotly_chart(fig, use_container_width=True)

with tab_query:
    query = st.text_input("Ask about ranks, scores, states, institutions, years, or parameters", placeholder="Which states had the highest average RPC score in 2025?")
    if query:
        q = query.lower()
        metric_aliases = {"research": "RPC", "rpc": "RPC", "teaching": "TLR", "tlr": "TLR", "graduation": "GO", "go score": "GO", "outreach": "OI", "inclusivity": "OI", "oi": "OI", "perception": "PR", "pr": "PR", "rank": "Rank", "score": "Overall score"}
        metric = next((value for phrase, value in metric_aliases.items() if re.search(rf"\b{re.escape(phrase)}\b", q)), "Overall score")
        match = re.search(r"20(?:1[7-9]|2[0-5])", q)
        query_year = int(match.group()) if match else analysis_year
        query_rows = selected[selected.year.eq(query_year)]
        metric_key = METRICS[metric]
        found_institution = next((name for name in sorted(selected.institution.unique(), key=len, reverse=True) if name.lower() in q), None)
        if query_rows.empty:
            st.error(f"No records match the selected filters for {query_year}.")
        elif any(word in q for word in ["correlation", "relationship", "related", "associated"]):
            parameters_in_question = [p for p in PARAMETERS if p.lower() in q]
            x_key = parameters_in_question[0] if parameters_in_question else metric_key
            y_key = parameters_in_question[1] if len(parameters_in_question) > 1 else "overall_score"
            values = query_rows[["institution", x_key, y_key]].dropna()
            coefficient = values[x_key].corr(values[y_key])
            st.success(f"Answer: the Pearson correlation between {x_key} and {y_key} in {query_year} is {coefficient:.2f}.")
            st.plotly_chart(px.scatter(values, x=x_key, y=y_key, hover_name="institution", trendline="ols", title=f"{x_key} versus {y_key} ({query_year})"), use_container_width=True)
        elif found_institution:
            values = selected[selected.institution.eq(found_institution)].sort_values("year")
            field = "rank" if metric == "Rank" else metric_key
            st.success(f"Answer: showing {metric.lower()} for {found_institution} from {values.year.min()} to {values.year.max()}.")
            fig = px.line(values, x="year", y=field, markers=True, title=f"{found_institution}: {metric} trend")
            if field == "rank": fig.update_yaxes(autorange="reversed", title="Rank (lower is better)")
            st.plotly_chart(fig, use_container_width=True)
        elif any(word in q for word in ["state", "states"]):
            values = query_rows.groupby("state", as_index=False)[metric_key].mean().sort_values(metric_key, ascending=metric == "Rank").head(15)
            leader = values.iloc[0]
            st.success(f"Answer: {leader.state} leads this view with an average {metric.lower()} of {leader[metric_key]:.2f} in {query_year}.")
            st.plotly_chart(px.bar(values, x=metric_key, y="state", orientation="h", title=f"Average {metric} by state"), use_container_width=True)
        elif any(word in q for word in ["distribution", "histogram", "spread", "quartile"]):
            st.success(f"Answer: showing the distribution of {metric.lower()} in {query_year}.")
            st.plotly_chart(px.histogram(query_rows, x=metric_key, nbins=20, title=f"{metric} distribution ({query_year})"), use_container_width=True)
        elif any(word in q for word in ["trend", "over time", "change"]):
            values = selected.groupby("year", as_index=False)[metric_key].mean()
            st.success(f"Answer: showing the average {metric.lower()} trend from {values.year.min()} to {values.year.max()}.")
            fig = px.line(values, x="year", y=metric_key, markers=True, title=f"Average {metric} over time")
            if metric == "Rank": fig.update_yaxes(autorange="reversed", title="Rank (lower is better)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            limit_match = re.search(r"top\s+(\d+)", q)
            limit = int(limit_match.group(1)) if limit_match else 10
            values = query_rows.sort_values(metric_key, ascending=metric == "Rank").head(limit)
            st.success(f"Answer: the leading {len(values)} institutions by {metric.lower()} in {query_year} are shown below.")
            st.plotly_chart(px.bar(values.sort_values(metric_key), x=metric_key, y="institution", orientation="h", title=f"Top {len(values)} institutions by {metric} ({query_year})"), use_container_width=True)
