import os
import subprocess
import sys


def test_imports_do_not_require_database_url(
    tmp_path,
):  # Directory without a .env file.
    environment = os.environ.copy()
    environment.pop("DATABASE_URL", None)
    environment.pop("JWT_SECRET", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import llm_gateway.main; import llm_gateway.models; "
            "from llm_gateway.db import get_engine, get_sessionmaker; "
            "assert get_engine.cache_info().currsize == 0; "
            "assert get_sessionmaker.cache_info().currsize == 0",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
