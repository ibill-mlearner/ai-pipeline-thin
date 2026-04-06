from __future__ import annotations

import time
from types import SimpleNamespace

from ai_pipeline.interaction_job_manager import InteractionJobManager
from ai_pipeline.interaction_job_manager_registry import InteractionJobManagerRegistry
from ai_pipeline.interaction_persistence_runner import InteractionPersistenceRunner


def wait_for_terminal_status(manager: InteractionJobManager, job_id: str, timeout: float = 2.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = manager.get(job_id)
        if record and record.status in {"succeeded", "failed"}:
            return record.status
        time.sleep(0.01)
    raise AssertionError("job did not finish in time")


def test_submit_success_sets_result() -> None:
    manager = InteractionJobManager(max_workers=1)

    job_id = manager.submit(lambda: {"ok": True})
    assert wait_for_terminal_status(manager, job_id) == "succeeded"

    record = manager.get(job_id)
    assert record is not None
    assert record.result == {"ok": True}
    assert record.error is None


def test_submit_failure_sets_error_payload() -> None:
    manager = InteractionJobManager(max_workers=1)

    def fail() -> dict[str, bool]:
        raise ValueError("boom")

    job_id = manager.submit(fail)
    assert wait_for_terminal_status(manager, job_id) == "failed"

    record = manager.get(job_id)
    assert record is not None
    assert record.result is None
    assert record.error is not None
    assert record.error["code"] == "runtime_unavailable"
    assert record.error["details"]["exception"] == "ValueError"


def test_get_or_create_job_manager_reuses_existing_instance() -> None:
    app = SimpleNamespace(extensions={})
    first = InteractionJobManagerRegistry.get_or_create(app, max_workers=1)
    second = InteractionJobManagerRegistry.get_or_create(app, max_workers=3)

    assert first is second


def test_run_and_persist_interaction_requires_successful_persist() -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def warn(request_id: str, payload: dict[str, object]) -> None:
        seen.append((request_id, payload))

    result = InteractionPersistenceRunner.run_and_persist(
        run_fn=lambda: {"response": "ok"},
        warn_if_empty_fn=warn,
        persist_fn=lambda: None,
        request_id="req-1",
    )
    assert result == {"response": "ok"}
    assert seen == [("req-1", {"response": "ok"})]

    try:
        InteractionPersistenceRunner.run_and_persist(
            run_fn=lambda: {"response": "ok"},
            warn_if_empty_fn=warn,
            persist_fn=lambda: {"db": "failed"},
            request_id="req-2",
        )
    except RuntimeError as exc:
        assert str(exc) == "Interaction persistence failed"
    else:
        raise AssertionError("expected RuntimeError")
