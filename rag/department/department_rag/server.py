"""HTTP server of the department RAG (POST /retrieve, GET /health: see ari_servers/rag.py).

Run: python -m department_rag [--port ...] [--skip-load] (PYTHONPATH: common and rag/department).
"""
import argparse

from ari_servers import execution
from ari_servers.http import setup_logging
from ari_servers.rag import RagService

from . import config
from .rag import DepartmentRag


def main(argv=None):
    setup_logging()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--skip-load", action="store_true", help="start without the models (/retrieve replies 501)")
    known, rest = pre.parse_known_args(argv)
    rag = DepartmentRag(config.load(rest))
    RagService(rag, execution.from_config(), load_models=not known.skip_load).serve_forever()


if __name__ == "__main__":
    main()
