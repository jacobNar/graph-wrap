import argparse
import os
import sys
import streamlit.web.cli as stcli

def main() -> None:
    parser = argparse.ArgumentParser(description="Start the graph-abstract telemetry and checkpoint UI.")
    parser.add_argument(
        "--db-uri",
        type=str,
        default=os.environ.get("GRAPH_WRAP_DB_URI") or os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URI"
    )
    args, unknown = parser.parse_known_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ui_app_path = os.path.join(current_dir, "ui.py")
    
    sys.argv = ["streamlit", "run", ui_app_path]
    if args.db_uri:
        sys.argv.extend(["--", "--db-uri", args.db_uri])
    stcli.main()
