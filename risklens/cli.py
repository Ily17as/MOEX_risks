import argparse
import sys
from pathlib import Path


def run_api(args=None):
    import uvicorn
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    uvicorn.run("risklens.api:app", host=host, port=port)


def run_app(args=None):
    import streamlit.web.cli as stcli
    app_path = Path(__file__).with_name("streamlit_app.py")
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


def main(argv=None):
    parser = argparse.ArgumentParser(description="RiskLens command line interface")
    sub = parser.add_subparsers(dest="command")

    p_api = sub.add_parser("api", help="Run FastAPI server")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.set_defaults(func=run_api)

    p_app = sub.add_parser("app", help="Run Streamlit web UI")
    p_app.set_defaults(func=run_app)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
