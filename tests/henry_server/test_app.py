from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from henry_speech.reply import ReplyRequest
from henry_server.app import create_app
from henry_server.reply import ReplyEvent
from henry_server.session import SessionService


class EmptyReplyProvider:
    async def reply(
        self,
        thread_id: str,
        request: ReplyRequest,
    ) -> AsyncIterator[ReplyEvent]:
        if False:
            yield


def test_session_api_lifecycle() -> None:
    service = SessionService(EmptyReplyProvider())

    with TestClient(create_app(service)) as client:
        assert client.get("/health").json() == {"status": "OK"}

        created = client.post("/api/v1/sessions")
        assert created.status_code == 201
        thread_id = created.json()["thread_id"]

        submitted = client.post(
            f"/api/v1/sessions/{thread_id}/inputs",
            json={"content": "hello"},
        )
        assert submitted.status_code == 202
        signalled = client.post(
            f"/api/v1/sessions/{thread_id}/inputs",
            json={"signal": "ACTIVATION"},
        )
        assert signalled.status_code == 202

        deleted = client.delete(f"/api/v1/sessions/{thread_id}")
        assert deleted.status_code == 204
        assert client.delete(f"/api/v1/sessions/{thread_id}").status_code == 404


def test_session_input_validation_and_unknown_session() -> None:
    with TestClient(create_app(SessionService(EmptyReplyProvider()))) as client:
        response = client.post(
            "/api/v1/sessions/missing/inputs",
            json={"content": "hello"},
        )
        assert response.status_code == 404

        created = client.post("/api/v1/sessions")
        thread_id = created.json()["thread_id"]
        response = client.post(
            f"/api/v1/sessions/{thread_id}/inputs",
            json={"content": ""},
        )
        assert response.status_code == 422


def test_openapi_documents_event_stream() -> None:
    app = create_app(SessionService(EmptyReplyProvider()))

    content = app.openapi()["paths"]["/api/v1/sessions/{thread_id}/events"]["get"][
        "responses"
    ]["200"]["content"]

    assert "text/event-stream" in content
