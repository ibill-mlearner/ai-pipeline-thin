# Interaction Job Queue Experiment (Additive Prototype)

This repo keeps the original in-memory implementation untouched and adds a queue-based experiment in parallel.

## Existing implementation (unchanged)
- `InteractionJobManager`: in-memory `ThreadPoolExecutor` manager
- `InteractionJobManagerRegistry`: singleton manager resolution through `app.extensions`
- `InteractionJobRecord`: status/result/error model
- `InteractionPersistenceRunner`: run + warn + persist flow

## New experimental files
- `ai_pipeline/celery_app_experimental.py`
- `ai_pipeline/interaction_tasks_celery.py`
- `ai_pipeline/interaction_job_manager_celery.py`
- `ai_pipeline/interaction_job_registry_celery.py`
- `ai_pipeline/interaction_job_manager_huey.py` (small alternative sketch)

## Celery flow (prototype)
1. Build Celery app (`create_celery_app`) with Redis broker/backend by default.
2. Resolve manager from Flask app: `InteractionJobRegistryCelery.get_or_create(app, celery_app=...)`.
3. Submit job:
   - `submit(work=callable)` for local/eager experiments, or
   - `submit_work_ref(work_ref="pkg.module:function", kwargs={...})` for worker-safe mode.
4. Poll status using `get(job_id)` and compare status transitions with the in-memory manager.

## Setup for local testing
```bash
pip install celery redis
redis-server
```

Start worker in a separate shell:
```bash
celery -A ai_pipeline.celery_app_experimental:create_celery_app worker --loglevel=INFO
```

Set env vars if needed:
```bash
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

## Flask wiring example
```python
from ai_pipeline.celery_app_experimental import create_celery_app
from ai_pipeline.interaction_job_registry_celery import InteractionJobRegistryCelery

celery_app = create_celery_app()
manager = InteractionJobRegistryCelery.get_or_create(app, celery_app=celery_app)
```

## Submit/check example
```python
job_id = manager.submit_work_ref(
    work_ref="my_project.interactions:run_interaction",
    kwargs={"request_id": "abc-123"},
)
record = manager.get(job_id)
print(record.status, record.result, record.error)
```

## Tradeoffs in plain language
- **Current in-memory manager**: easiest to understand; no extra infrastructure; great for demos/small projects; jobs are lost on process restart.
- **Celery manager**: needs Redis/RabbitMQ and worker process; better for scaling/reliability; more moving parts.
- **Huey sketch**: lighter mental model than Celery, but still needs queue backend and consumer process.
