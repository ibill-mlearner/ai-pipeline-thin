# ai-pipeline-thin

This package ships with a simple default interaction job flow that runs in-process using a `ThreadPoolExecutor`, so it works well for single-file demos and small projects without extra infrastructure. Jobs are submitted and tracked by ID, and each record keeps status (`queued`, `running`, `succeeded`, `failed`) plus result/error details. This is the easiest way to get started when you just want to install the package and run a quick interaction pipeline.

Alongside that default, the repository also includes an additive **experimental** queue-backed path (Celery first, plus a lightweight Huey sketch) so you can compare distributed worker behavior without replacing the original implementation. The Celery prototype keeps similar concepts (job IDs, status checks, result/error access) but requires queue infrastructure like Redis/RabbitMQ and a worker process. For dependency usage in another project, install from Git with pip (directly or through `requirements.txt`) so deployments pull this package consistently.

If you prefer one object as a gateway to all exported classes, use `AIPipelineInterface` and access symbols from there instead of importing each class one-by-one.

```python
from ai_pipeline import AIPipelineInterface

api = AIPipelineInterface()
service = api.AIPipelineInteractionService()
jobs = api.InteractionJobManager(max_workers=2)
```

Ollama helpers are also available for server-first local workflows:

```python
from ai_pipeline import OllamaServerService, OllamaHealthService, OllamaInteractionService

server = OllamaServerService()
server.ensure_running(start_if_needed=True)
health = OllamaHealthService()
print(health.status())
ollama = OllamaInteractionService()
```

Download-only prefetch is available for admin/cache warmup workflows:

```python
from ai_pipeline import AIPipelineModelDownloadService

downloader = AIPipelineModelDownloadService()
payload = downloader.download("Qwen/Qwen2.5-0.5B-Instruct")
print(payload)
```

Install from git:

```bash
pip install "git+https://github.com/<org>/ai-pipeline-thin.git"
```

Or add this line to your `requirements.txt`:

```text
git+https://github.com/<org>/ai-pipeline-thin.git
```

Then import either package style:

```python
import ai_pipeline
from ai_pipeline import AIPipeline

import ai_pipeline_thin
```
