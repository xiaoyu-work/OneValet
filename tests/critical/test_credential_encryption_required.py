"""P0-4: Fail-secure credential encryption."""

import pytest

from koa.credentials.encryption import CredentialEncryptionError, CredentialEncryptor


def test_passthrough_when_not_required():
    enc = CredentialEncryptor(key=None, require_encryption=False)
    assert enc.enabled is False
    assert enc.encrypt({"a": 1}) == {"a": 1}


def test_require_without_key_raises():
    with pytest.raises(CredentialEncryptionError):
        CredentialEncryptor(key=None, require_encryption=True)


def test_require_rejects_plaintext_dict():
    enc = CredentialEncryptor(key="test-key-32-bytes-long-xxxxxxxxxx", require_encryption=True)
    # Legacy plaintext JSONB dict must not be silently accepted.
    with pytest.raises(CredentialEncryptionError):
        enc.decrypt({"username": "u"})


def test_require_rejects_plaintext_string():
    enc = CredentialEncryptor(key="test-key-32-bytes-long-xxxxxxxxxx", require_encryption=True)
    with pytest.raises(CredentialEncryptionError):
        enc.decrypt('{"username":"u"}')


def test_roundtrip_when_key_set():
    enc = CredentialEncryptor(key="abc" * 12, require_encryption=True)
    blob = enc.encrypt({"token": "secret"})
    assert isinstance(blob, str)
    assert blob.startswith("enc:v1:")
    assert enc.decrypt(blob) == {"token": "secret"}


def test_sanitize_for_log_redacts_secrets():
    from koa.credentials.encryption import sanitize_for_log

    sanitized = sanitize_for_log({"username": "u", "password": "p", "nested": {"api_key": "k"}})
    assert sanitized["username"] == "u"
    assert sanitized["password"] == "***REDACTED***"
    assert sanitized["nested"]["api_key"] == "***REDACTED***"
