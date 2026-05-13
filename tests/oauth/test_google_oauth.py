import pytest

from koa.oauth.google_oauth import GOOGLE_SCOPES, GoogleOAuth


def test_google_scopes_match_implemented_features():
    assert "https://www.googleapis.com/auth/userinfo.email" in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/userinfo.profile" in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/gmail.modify" in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/calendar.events" in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/tasks" in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/calendar" not in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/drive.readonly" not in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/drive" not in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/documents" not in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/spreadsheets" not in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/gmail.readonly" not in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/gmail.send" not in GOOGLE_SCOPES


@pytest.mark.asyncio
async def test_fetch_user_email_uses_google_profile_payload(monkeypatch):
    async def fake_fetch_user_profile(access_token: str):
        assert access_token == "token"
        return {"email": "user@example.com", "name": "Koi User"}

    monkeypatch.setattr(GoogleOAuth, "fetch_user_profile", fake_fetch_user_profile)

    assert await GoogleOAuth.fetch_user_email("token") == "user@example.com"
