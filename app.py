from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Callable, Optional

import streamlit as st
from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain
from langchain.messages import HumanMessage


# =============================================================================
# Research pipeline (formerly pipeline.py / pipeline_progress.py)
# =============================================================================

ProgressCallback = Callable[[int, str, str], None]


def run_research_pipeline(
    topic: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict:
    """
    Run the four-stage research workflow.

    progress_callback(step_number, message, status) is optional.
    status is "running", "done", or "error".
    """
    state: dict[str, Any] = {}

    def report_progress(step: int, message: str, status: str = "running"):
        print("\n" + " =" * 50)
        print(message)
        print("=" * 50)
        if progress_callback:
            progress_callback(step, message, status)

    # Step 1: Search agent
    report_progress(1, "Step 1 — Search agent is working ...", "running")
    try:
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [
                HumanMessage(
                    content=f"Find recent, reliable and detailed information about: {topic}"
                )
            ]
        })
        state["search_results"] = search_result["messages"][-1].content
        print("\nSearch result:\n", state["search_results"])
        report_progress(1, "Search agent completed.", "done")
    except Exception:
        if progress_callback:
            progress_callback(1, "Search agent failed.", "error")
        raise

    # Step 2: Reader agent
    report_progress(2, "Step 2 — Reader agent is scraping a relevant resource ...", "running")
    try:
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [
                HumanMessage(content=(
                    f"Based on the following search results about '{topic}', "
                    "pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{state['search_results'][:800]}"
                ))
            ]
        })
        state["scraped_content"] = reader_result["messages"][-1].content
        print("\nScraped content:\n", state["scraped_content"])
        report_progress(2, "Reader agent completed.", "done")
    except Exception:
        if progress_callback:
            progress_callback(2, "Reader agent failed.", "error")
        raise

    # Step 3: Writer chain
    report_progress(3, "Step 3 — Writer is drafting the report ...", "running")
    try:
        research_combined = (
            f"Search Results:\n{state['search_results']}\n\n"
            f"Detailed Scraped Content:\n{state['scraped_content']}"
        )
        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
        print("\nFinal report:\n", state["report"])
        report_progress(3, "Report draft completed.", "done")
    except Exception:
        if progress_callback:
            progress_callback(3, "Writer failed to draft the report.", "error")
        raise

    # Step 4: Critic chain
    report_progress(4, "Step 4 — Critic is reviewing the report ...", "running")
    try:
        state["feedback"] = critic_chain.invoke({"report": state["report"]})
        print("\nCritic feedback:\n", state["feedback"])
        report_progress(4, "Critic review completed.", "done")
    except Exception:
        if progress_callback:
            progress_callback(4, "Critic review failed.", "error")
        raise

    return state


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(
    page_title="Atlas | Research Studio",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background:
      radial-gradient(ellipse at 12% 0%, rgba(139,124,255,.10), transparent 34%),
      radial-gradient(ellipse at 100% 18%, rgba(70,183,165,.07), transparent 28%); }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(140,140,160,.16); }
    .atlas-eyebrow { text-transform: uppercase; letter-spacing: .18em; font-size: .72rem;
      font-weight: 700; color: #9B91FF; margin-bottom: .7rem; }
    .atlas-hero { padding: 1.8rem 1.9rem; border: 1px solid rgba(150,145,190,.22);
      border-radius: 24px; background: linear-gradient(120deg, rgba(139,124,255,.13), rgba(70,183,165,.055));
      margin-bottom: 1.3rem; }
    .atlas-hero h1 { font-family: 'Manrope', sans-serif; letter-spacing: -.045em;
      font-size: clamp(2.2rem, 4vw, 3.25rem); line-height: 1.06; margin: 0 0 .8rem 0; }
    .atlas-hero p { max-width: 760px; opacity: .75; font-size: 1.03rem;
      margin: 0; line-height: 1.65; }
    .atlas-section { font-family: 'Manrope', sans-serif; font-weight: 750; letter-spacing: -.025em;
      font-size: 1.28rem; margin: .4rem 0 .25rem; }
    .atlas-muted { opacity: .68; font-size: .91rem; }
    .atlas-step { border: 1px solid rgba(140,140,160,.22); border-radius: 16px;
      padding: 1rem 1.05rem; margin-bottom: .65rem; background: rgba(128,128,150,.025); }
    .atlas-step.active { border-color: rgba(139,124,255,.65); background: rgba(139,124,255,.075); }
    .atlas-step.done { border-color: rgba(70,183,165,.35); }
    .atlas-step-title { font-weight: 700; font-size: 1rem; margin-bottom: .2rem; }
    .atlas-step-desc { opacity: .7; font-size: .86rem; line-height: 1.45; }
    .atlas-pill { display: inline-block; padding: .22rem .55rem; border-radius: 999px;
      border: 1px solid rgba(140,140,160,.25); font-size: .72rem; letter-spacing: .04em; }
    div[data-testid="stMetric"] { border: 1px solid rgba(140,140,160,.20);
      padding: 1rem 1.05rem; border-radius: 16px; background: rgba(128,128,150,.035); }
    div[data-testid="stMetricLabel"] p { opacity: .7; }
    .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
      border: 0; border-radius: 12px; min-height: 3rem; font-weight: 700;
      background: linear-gradient(100deg, #7565E8, #9385FF); }
    .stDownloadButton button { border-radius: 10px; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    .atlas-footer { border-top: 1px solid rgba(140,140,160,.18); padding-top: 1rem;
      opacity: .62; font-size: .8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

STEPS = [
    ("Search sources", "Discover recent, relevant information and candidate URLs.", "Search agent"),
    ("Read the evidence", "Scrape a useful source and gather deeper context.", "Reader agent"),
    ("Draft the report", "Turn the collected research into a structured report.", "Writer"),
    ("Review quality", "Score the report and identify strengths and gaps.", "Critic"),
]
RESULT_KEYS = ["search_results", "scraped_content", "report", "feedback"]


def score_from_feedback(text: str) -> str:
    match = re.search(r"Score\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10", text or "", re.I)
    return f"{match.group(1)}/10" if match else "—"


def text_value(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def render_step_tracker(container, statuses: list[str], active_index: int | None):
    with container.container():
        st.markdown('<div class="atlas-section">Run progress</div>', unsafe_allow_html=True)
        st.caption("Your agents work through the research in four connected stages.")
        for i, (title, desc, role) in enumerate(STEPS):
            if statuses[i] == "done":
                symbol, cls, status = "✓", "done", "COMPLETE"
            elif active_index == i:
                symbol, cls, status = "↻", "active", "RUNNING"
            elif statuses[i] == "error":
                symbol, cls, status = "!", "active", "FAILED"
            else:
                symbol, cls, status = str(i + 1), "", "QUEUED"
            st.markdown(
                f"""
                <div class="atlas-step {cls}">
                  <div style="display:flex;gap:.8rem;align-items:flex-start">
                    <div style="width:32px;height:32px;flex:0 0 32px;border-radius:11px;
                      display:flex;align-items:center;justify-content:center;
                      background:rgba(139,124,255,.13);font-weight:800">{symbol}</div>
                    <div style="flex:1">
                      <div class="atlas-step-title">{title}</div>
                      <div class="atlas-step-desc">{desc}</div>
                      <div style="display:flex;justify-content:space-between;gap:.5rem;margin-top:.6rem">
                        <span class="atlas-step-desc">{role}</span>
                        <span class="atlas-pill">{status}</span>
                      </div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


with st.sidebar:
    st.markdown("### ✳ Atlas")
    st.caption("Multi-agent research studio")
    st.divider()
    st.markdown("**Your workflow**")
    st.markdown("01  Search sources  \n02  Read evidence  \n03  Draft report  \n04  Critic review")
    st.divider()
    st.markdown("**Built for focused research**")
    st.caption("Clear reports, inspectable evidence, downloadable results.")
    st.divider()
    if st.button("Clear session results", use_container_width=True):
        for key in ("atlas_latest", "atlas_history"):
            st.session_state.pop(key, None)
        st.rerun()
    st.caption("Runs locally through your configured agents and tools.")

st.markdown(
    """
    <div class="atlas-hero">
      <div class="atlas-eyebrow">Multi-agent intelligence · Research workspace</div>
      <h1>From question to<br>clear thinking.</h1>
      <p>Research a topic with a coordinated team of AI agents. Follow every stage,
      inspect the evidence, review the critic's feedback, and export the report when it's ready.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.5, 1], gap="large")
with left:
    st.markdown('<div class="atlas-section">What are we investigating?</div>', unsafe_allow_html=True)
    st.markdown('<p class="atlas-muted">Use a focused question for more useful research.</p>', unsafe_allow_html=True)
    with st.form("atlas_research_form"):
        topic = st.text_area(
            "Research question",
            placeholder="e.g. How are agentic AI systems changing software engineering in 2026?",
            height=120,
            label_visibility="collapsed",
        )
        run_clicked = st.form_submit_button("✳  Start research", type="primary", use_container_width=True)
    st.caption("Tip: specify a timeframe, region, technology, or comparison to narrow the research.")
with right:
    st.markdown('<div class="atlas-section">What happens next</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="atlas-step">
          <div class="atlas-step-title">Evidence first</div>
          <div class="atlas-step-desc">Inspect search and source content—not just the final answer.</div>
        </div>
        <div class="atlas-step">
          <div class="atlas-step-title">A structured report</div>
          <div class="atlas-step-desc">Introduction, key findings, conclusion, and source URLs.</div>
        </div>
        <div class="atlas-step">
          <div class="atlas-step-title">A separate quality pass</div>
          <div class="atlas-step-desc">The critic provides a score, strengths, and areas to improve.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
progress_col, results_col = st.columns([0.78, 1.6], gap="large")
with progress_col:
    tracker_slot = st.empty()
    render_step_tracker(tracker_slot, ["queued"] * 4, None)

if run_clicked:
    clean_topic = (topic or "").strip()
    if not clean_topic:
        st.warning("Add a research question before starting.")
    else:
        statuses = ["queued"] * 4
        active_slot = st.empty()
        render_step_tracker(active_slot, statuses, 0)
        status_message = st.empty()
        started = time.perf_counter()

        def on_progress(step: int, message: str, status: str = "running"):
            index = max(0, min(step - 1, 3))
            if status == "done":
                statuses[index] = "done"
                next_active = index + 1 if index < 3 else None
            elif status == "error":
                statuses[index] = "error"
                next_active = None
            else:
                next_active = index
            render_step_tracker(active_slot, statuses, next_active)
            status_message.info(message)

        try:
            with st.spinner("The research team is working. Longer runs are normal for web research…"):
                output = run_research_pipeline(clean_topic, progress_callback=on_progress)

            elapsed = time.perf_counter() - started
            record = {
                "topic": clean_topic,
                "created_at": datetime.now().strftime("%d %b %Y, %H:%M"),
                "elapsed": elapsed,
                **{key: text_value(output.get(key)) for key in RESULT_KEYS},
            }
            st.session_state["atlas_latest"] = record
            st.session_state.setdefault("atlas_history", [])
            st.session_state["atlas_history"].insert(0, record)
            st.session_state["atlas_history"] = st.session_state["atlas_history"][:5]
            status_message.success(f"Research completed in {elapsed:.1f} seconds.")
        except Exception as exc:
            status_message.error("The pipeline stopped before completing all stages.")
            st.exception(exc)

record = st.session_state.get("atlas_latest")
with results_col:
    st.markdown('<div class="atlas-section">Research output</div>', unsafe_allow_html=True)
    if record:
        st.markdown(f"**{record['topic']}**")
        st.caption(f"Completed {record['created_at']} · {record['elapsed']:.1f}s")
        report = record["report"]
        feedback = record["feedback"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Report words", f"{len(report.split()):,}" if report else "0")
        m2.metric("Critic score", score_from_feedback(feedback))
        m3.metric("Stages", "4 / 4")
        tab_report, tab_review, tab_search, tab_sources = st.tabs(
            ["Report", "Critic review", "Search output", "Scraped evidence"]
        )
        with tab_report:
            if report:
                st.markdown(report)
                st.download_button(
                    "Download report · Markdown",
                    data=f"# {record['topic']}\n\n{report}",
                    file_name="atlas_research_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.info("No report was returned by the pipeline.")
        with tab_review:
            st.markdown(feedback or "No critic feedback was returned.")
            if feedback:
                st.download_button(
                    "Download review · TXT", feedback,
                    file_name="atlas_critic_review.txt", mime="text/plain",
                )
        with tab_search:
            st.text_area("Search agent output", record["search_results"], height=360)
            if record["search_results"]:
                st.download_button(
                    "Download search output", record["search_results"],
                    file_name="atlas_search_results.txt", mime="text/plain",
                )
        with tab_sources:
            st.text_area("Reader agent output", record["scraped_content"], height=360)
            if record["scraped_content"]:
                st.download_button(
                    "Download scraped evidence", record["scraped_content"],
                    file_name="atlas_scraped_evidence.txt", mime="text/plain",
                )
    else:
        st.markdown(
            """
            <div class="atlas-step" style="padding:1.5rem">
              <div class="atlas-eyebrow">Your workspace is ready</div>
              <div class="atlas-step-title" style="font-size:1.15rem">Your report will appear here.</div>
              <div class="atlas-step-desc">Start a research run to see the report, critic review, and evidence in one place.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

history = st.session_state.get("atlas_history", [])
if len(history) > 1:
    st.divider()
    with st.expander(f"Recent research runs · {len(history)} saved in this session"):
        for index, item in enumerate(history[1:], start=1):
            st.markdown(f"**{item['topic']}** · {item['created_at']}")
            st.download_button(
                "Download this report",
                data=f"# {item['topic']}\n\n{item['report']}",
                file_name=f"atlas_previous_report_{index}.md",
                mime="text/markdown",
                key=f"history_{index}_{item['created_at']}_{abs(hash(item['topic']))}",
            )

st.markdown(
    '<div class="atlas-footer">ATLAS RESEARCH STUDIO · Verify important claims against the source material.</div>',
    unsafe_allow_html=True,
)
