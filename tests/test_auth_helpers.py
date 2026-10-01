import re

import pytest

from llm_gateway.auth import generate_key, hash_key, parse_key


def test_generation_and_hashing():
    key = generate_key()
    assert re.fullmatch(r"gw_[0-9a-f]{8}_[A-Za-z0-9_-]{43}", key)
    assert parse_key(key) == key[:11]
    assert hash_key(key) == hash_key(key)
    assert len(hash_key(key)) == 64
    assert hash_key(key) != hash_key(key + "x")


def test_secret_with_underscore():
    assert parse_key("gw_1234abcd_" + "_" * 43) == "gw_1234abcd"


@pytest.mark.parametrize(
    "key",
    [
        "",
        "gw",
        "gw_1234abcd",
        "gw_1234abcd_",
        "xx_1234abcd_" + "a" * 43,
        "gw_1234ABCZ_" + "a" * 43,
        "gw_1234abcd_" + "a" * 42,
        "gw_1234abcd_" + "!" * 43,
        "gw_1234abcd_" + "a" * 44,
    ],
)
def test_malformed_keys(key):  # Candidate untrusted key.
    with pytest.raises(ValueError):
        parse_key(key)
