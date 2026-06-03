import argparse
import sys
from typing import List, Set
import streamlit as st
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from graph_abstract.trace_parser import TraceParser
from graph_abstract.trace_visualizer import TraceVisualizer

def parse_args() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-uri", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.db_uri or ""

def get_threads(db_uri: str) -> List[str]:
    threads: Set[str] = set()
    try:
        with psycopg.connect(db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT thread_id FROM agent_logs;")
                for row in cur.fetchall():
                    threads.add(row[0])
                cur.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name='checkpoints';"
                )
                if cur.fetchone():
                    cur.execute("SELECT DISTINCT thread_id FROM checkpoints;")
                    for row in cur.fetchall():
                        threads.add(row[0])
    except Exception as e:
        st.sidebar.error(f"Error fetching threads: {e}")
    return sorted(list(threads))

def show_checkpoints_tab(db_uri: str, thread_id: str) -> None:
    st.subheader("State Checkpoints")
    try:
        with PostgresSaver.from_conn_string(db_uri) as saver:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint_tuples = list(saver.list(config))
            
            if not checkpoint_tuples:
                st.info("No checkpoints found for this thread.")
                return
                
            for cpt in checkpoint_tuples:
                c_id = cpt.config["configurable"].get("checkpoint_id")
                ns = cpt.config["configurable"].get("checkpoint_ns", "root")
                step = cpt.metadata.get("step", -1)
                
                with st.expander(f"Step {step} - NS: {ns} - ID: {c_id}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            "config": cpt.config,
                            "metadata": cpt.metadata,
                            "parent_config": cpt.parent_config
                        })
                    with col2:
                        st.json(cpt.checkpoint)
    except Exception as e:
        st.error(f"Error listing checkpoints: {e}")

def show_traces_tab(db_uri: str, thread_id: str) -> None:
    st.subheader("Observability Traces")
    try:
        with psycopg.connect(db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_name, payload, created_at FROM agent_logs WHERE thread_id = %s ORDER BY created_at ASC",
                    (thread_id,)
                )
                rows = cur.fetchall()
                
                if not rows:
                    st.info("No traces found for this thread.")
                    return
                
                parser = TraceParser()
                invocations = parser.parse_rows(rows)
                visualizer = TraceVisualizer()
                
                for idx, invocation in enumerate(invocations):
                    status_badge = "🟢 Success" if invocation.status == "success" else "🔴 Error"
                    time_str = invocation.start_time.strftime("%Y-%m-%d %H:%M:%S") if invocation.start_time else "Unknown"
                    duration_str = f"{invocation.duration_ms:.2f}ms"
                    
                    with st.expander(f"Invocation {len(invocations) - idx} (Start: {time_str})"):
                        st.write(f"**Status:** {status_badge} | **Duration:** {duration_str}")
                        html_content = visualizer.render_timeline_html(invocation)
                        st.html(html_content)
                        
                        st.write("#### Inspect Spans")
                        span_options = invocation.spans
                        
                        def format_span(span):
                            indent = "\xa0\xa0" * span.depth
                            status_sym = "🟢" if span.status == "success" else "🔴"
                            return f"{indent}{status_sym} {span.name} ({span.duration_ms:.1f}ms)"
                            
                        selected_span = st.selectbox(
                            "Select a span to inspect details",
                            options=span_options,
                            format_func=format_span,
                            key=f"span_select_{thread_id}_{idx}"
                        )
                        
                        if selected_span:
                            st.write(f"### Span Details: {selected_span.name}")
                            tab_overview, tab_inputs, tab_outputs, tab_raw = st.tabs(
                                ["Overview", "Inputs", "Outputs", "Raw Payloads"]
                            )
                            
                            with tab_overview:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Type:** {selected_span.type.upper()}")
                                    st.write(f"**Status:** {selected_span.status.upper()}")
                                    st.write(f"**Duration:** {selected_span.duration_ms:.2f}ms")
                                with col2:
                                    st.write(f"**Start Time:** {selected_span.start_time}")
                                    st.write(f"**End Time:** {selected_span.end_time}")
                                if selected_span.error:
                                    st.error(f"Error: {selected_span.error}")
                                    
                            with tab_inputs:
                                if selected_span.inputs is not None:
                                    st.json(selected_span.inputs)
                                else:
                                    st.info("No input payload found for this span.")
                                    
                            with tab_outputs:
                                if selected_span.outputs is not None:
                                    st.json(selected_span.outputs)
                                else:
                                    st.info("No output payload found for this span.")
                                    
                            with tab_raw:
                                st.json(selected_span.payloads)
    except Exception as e:
        st.error(f"Error querying traces: {e}")

def draw_tabs(db_uri: str, thread_id: str) -> None:
    tab_traces, tab_checkpoints = st.tabs(["Observability Traces", "State Checkpoints"])
    with tab_traces:
        show_traces_tab(db_uri, thread_id)
    with tab_checkpoints:
        show_checkpoints_tab(db_uri, thread_id)

@st.fragment(run_every=5)
def render_tabs_auto(db_uri: str, thread_id: str) -> None:
    draw_tabs(db_uri, thread_id)

@st.fragment
def render_tabs_static(db_uri: str, thread_id: str) -> None:
    draw_tabs(db_uri, thread_id)

def main() -> None:
    st.set_page_config(page_title="graph-abstract Observability Console", layout="wide")
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp {
            background-color: #ffffff !important;
            color: #31333f !important;
        }
        [data-testid="stSidebar"] {
            background-color: #f0f2f6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("graph-abstract Observability Console")
    
    cli_uri = parse_args()
    db_uri = st.sidebar.text_input("Database URI", value=cli_uri or st.session_state.get("db_uri", ""))
    st.session_state["db_uri"] = db_uri
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    if not db_uri:
        st.warning("Please enter a database connection URI in the sidebar.")
        return
        
    threads = get_threads(db_uri)
    if not threads:
        st.info("No thread data found in the database.")
        return
        
    thread_id = st.sidebar.selectbox("Select Thread ID", threads)
    
    if auto_refresh:
        render_tabs_auto(db_uri, thread_id)
    else:
        render_tabs_static(db_uri, thread_id)

if __name__ == "__main__":
    main()
