# AI Pipeline v0.1 Contract

**Status:** Draft, implementation-ready baseline for `ai-pipeline-thin` v0.1.x consumers.  
**Applies to:** package `ai-pipeline-thin` (pipeline library) and application adapter layer (example name: `AccessBackEnd`).

---

## 1) Purpose

This contract defines ownership and runtime truth boundaries so model-path and inventory mismatches do not occur.

Primary goals:
1. A single source of truth for model download/cache behavior and local model discovery.
2. Explicit adapter responsibilities for model/provider resolution and persistence.
3. Stable error and compatibility guarantees for application consumers.

---

## 2) Ownership Matrix

| Capability / Concern | Pipeline Package (`ai_pipeline`) owns | Application Adapter (`AccessBackEnd`) owns |
|---|---|---|
| Model loading and tokenizer loading | `ModelLoader.build()` and `TokenizerLoader.build()` behavior, including dependency handling and cache path used by loaders. | Invoking pipeline APIs and passing desired inputs; must not bypass pipeline internals when determining model-cache locations. |
| Download behavior | `AIPipelineModelDownloadService.download(model_id, provider=...)` semantics and returned payload shape. | Triggering downloads and handling user intent/workflow around them (authorization, UI status, retries). |
| Runtime local model enumeration | `AvailableModels.build()` output from runtime filesystem cache contents. | Exposing runtime inventory endpoint and deciding how (or whether) to merge DB metadata. |
| Model ID handling | Treat model IDs as opaque strings; no normalization/canonicalization beyond parsing cache folder names. | Preserving selected model ID exactly as chosen; avoid rewriting IDs before passing to pipeline. |
| Selection resolution order | Utility behavior in `AIPipelineInteractionService.resolve_model_id(...)`. | Supplying request override/session/config values and enforcing final precedence policy in app request flow. |
| Prompt/system/context request shape | `AIPipelineRequest` fields and fallback helpers (e.g., resolved system prompt, prompt fallback from messages). | Constructing valid request objects from API payloads and session state. |
| Error wrapping from pipeline operations | `AIPipelineUpstreamError` and stable details keys for wrapped failures. | Mapping pipeline errors to HTTP/API error responses and persistence/auditing policy. |
| Cache root env precedence | Pipeline contract defines accepted environment variables and precedence (see §5). | Setting environment consistently in runtime/deployment and documenting deployment-specific defaults. |
| DB persistence schema | Not owned by pipeline package. | Defining persisted records (selected model, provenance, user-visible labels, historical rows). |
| Inventory endpoint semantics | Pipeline provides runtime raw inventory builder only. | `/api/v1/ai/models/available` contract, including runtime truth requirement and stale DB handling. |

---

## 3) Pipeline Package Contract (Source of Truth)

### 3.1 Required Public APIs

The following APIs are required and considered public for v0.1.x application consumers:

1. **Model loading**
   - `ModelLoader(model_name, torch_dtype="auto", device_map="auto", download_locally=True).build() -> Any`
   - MUST load model by `model_name` and pass cache directory according to §5 when `download_locally=True`.

2. **Tokenizer loading**
   - `TokenizerLoader(model_name, download_locally=True).build() -> Any`
   - MUST use the same cache root resolution policy as model loading.

3. **Download behavior**
   - `AIPipelineModelDownloadService(default_provider="huggingface").download(model_id, provider=None) -> dict`
   - MUST prefetch model + tokenizer artifacts.
   - MUST return payload keys: `provider`, `model_id`, `status` (`"downloaded"` on success).

4. **Cache directory reporting**
   - Pipeline MUST expose an explicit cache-root reporting API in v0.1 contract surface:
     - `get_effective_models_dir() -> str`
   - Until implemented as code, adapters SHOULD treat this document as normative and derive value using §5 policy + package location fallback.

5. **Local model enumeration**
   - `AvailableModels(models_dir: Path | None = None).build() -> dict[str, list[str]]`
   - MUST enumerate runtime model cache content only (no DB reads).
   - MUST return provider-grouped model names.

### 3.2 Model ID Handling Rules

1. Model ID is an **opaque identifier string** at API boundaries.
2. Accepted input format for Hugging Face IDs is `"<namespace>/<model-name>"` (e.g., `Qwen/Qwen2.5-3B-Instruct`).
3. Pipeline MUST NOT silently rewrite case, separators, namespace, or revision suffixes.
4. Pipeline MAY reject malformed IDs early with `AIPipelineUpstreamError` details indicating validation failure class/message.
5. Inventory parsing from cache folders (`models--namespace--model`) is an internal representation detail; output model IDs reconstructed as `namespace/model` MUST be treated as canonical runtime-discovered values.

### 3.3 Cache/Download Location Behavior & Env Precedence

**Normative precedence (highest first):**
1. `AI_PIPELINE_MODELS_DIR`
2. `HF_HOME` (mapped to `${HF_HOME}/hub` semantics for Hugging Face cache)
3. `TRANSFORMERS_CACHE`
4. Pipeline local default: `<package_dir>/models` (current implementation baseline)

Rules:
- The same resolved directory MUST be used for model loading, tokenizer loading, downloads, and inventory enumeration unless an explicit override parameter is supplied.
- When env vars are set inconsistently, pipeline MUST follow precedence above deterministically.
- Adapter MUST NOT maintain an independent cache root for inventory that diverges from pipeline resolution.
- In containerized deployments, adapter SHOULD set exactly one authoritative env var (`AI_PIPELINE_MODELS_DIR`) to avoid ambiguity.

### 3.4 Error Contract (Classes + Stable Fields)

1. Pipeline wrapper error class: `AIPipelineUpstreamError`.
2. Required top-level fields:
   - exception message (human-readable)
   - `.details` dictionary.
3. Required stable keys in `.details` for wrapped pipeline failures:
   - `exception_class` (string)
   - `message` (string)
4. Download-specific failures MUST also include:
   - `model_id` (string)
5. Adapters MAY include additional keys when remapping to API errors but MUST preserve original stable keys.

### 3.5 Versioning & Compatibility Guarantees

For `0.1.x`:
1. Public API names in §3.1 are **minor-version stable**.
2. Stable error keys in §3.4 are **minor-version stable**.
3. Selection precedence behavior in service helper methods is **minor-version stable**.
4. New optional fields MAY be added to successful payloads; existing keys MUST NOT be removed in `0.1.x`.
5. Any breaking change requires bump beyond `0.1.x` and explicit migration documentation.

---

## 4) Application Adapter Contract (`AccessBackEnd`)

### 4.1 Inputs App Passes into Pipeline

Adapter MUST construct and pass:
- `prompt` (string; may be empty)
- `system_prompt` (nullable string)
- `context` (object/dict)
- `provider` (nullable string)
- `model_id` (nullable string)
- `messages` (list of `{role, content}` entries when available)

Behavior:
- If `prompt` empty, adapter MAY rely on pipeline helper fallback to latest user message.
- Adapter MUST pass `model_id` exactly as selected/resolved (opaque string rule).

### 4.2 Model/Provider Resolution Order

Required deterministic order for selected model:
1. Request override (`request.model_id`)
2. Session-selected model
3. Configured default model

Required deterministic order for provider:
1. Request override (`request.provider`)
2. Session provider
3. Config default provider

Adapter MUST record which source won (for observability) but MUST pass only final resolved values into pipeline request.

### 4.3 Persistence: DB vs Display-Only

Adapter persistence rules:

**Persist in DB (authoritative app state):**
- User/session selected `model_id`
- Selection source (`request|session|config`)
- Provider used
- Interaction request/response metadata needed for audit (timestamps, status, error shape)

**Display-only (non-authoritative):**
- Friendly model labels
- Last-seen inventory snapshots for UI rendering
- Optional tags/descriptions

DB rows describing "available models" are advisory only and MUST NOT override runtime-discovered truth.

### 4.4 `/api/v1/ai/models/available` Contract

Endpoint MUST represent **runtime truth**, not a stale DB snapshot.

Required behavior:
1. Build inventory from pipeline runtime cache (`AvailableModels.build()` over effective models dir).
2. Optionally join DB metadata for display enrichment only.
3. If a DB row exists for a model absent from runtime cache, mark as `stale_metadata=true` (or exclude), but do not present as available for inference.
4. Response SHOULD include provenance fields:
   - `source`: `"runtime"` or `"runtime+metadata"`
   - `models_dir`: resolved effective cache directory
   - `generated_at` timestamp

---

## 5) Runtime Diagnostics & Incident Triage (Required for v0.1)

This section is **operational**, not advisory. Every production-like environment MUST provide a one-shot diagnostics entrypoint and CI gate.

### 5.1 One-shot diagnostics command (required)

Required command shape:

```bash
python -m ai_pipeline.diagnostics
```

Required machine-readable command shape:

```bash
python -m ai_pipeline.diagnostics --json
```

Required outputs (human mode and JSON mode):
1. Installed package name + version.
2. Installed package commit/build reference (or explicit `unknown`).
3. Resolved source files for key classes:
   - `AIPipelineModelDownloadService`
   - `AvailableModels`
   - `AIPipelineInteractionService`
4. Effective cache root.
5. Cache-root precedence decision trace (which env var/path won and why).
6. Whether downloader service exists and is importable.
7. Runtime inventory count and inventory root used.
8. Final explicit verdict: `PASS` or `FAIL`.

Minimum JSON schema (stable for `0.1.x`):

```json
{
  "status": "PASS|FAIL",
  "package": {"name": "ai-pipeline-thin", "version": "0.1.0", "commit": "abc123|unknown"},
  "sources": {
    "AIPipelineModelDownloadService": "/.../model_download_service.py",
    "AvailableModels": "/.../available_models.py",
    "AIPipelineInteractionService": "/.../interaction_service.py"
  },
  "cache": {
    "effective_root": "/path",
    "decision_trace": [
      {"candidate": "AI_PIPELINE_MODELS_DIR", "value": "/x", "selected": true},
      {"candidate": "HF_HOME", "value": null, "selected": false},
      {"candidate": "TRANSFORMERS_CACHE", "value": null, "selected": false},
      {"candidate": "PACKAGE_DEFAULT", "value": "/pkg/ai_pipeline/models", "selected": false}
    ]
  },
  "downloader": {"available": true},
  "inventory": {"root": "/path", "count": 3},
  "findings": ["WARN: db_contains_stale_models"]
}
```

### 5.2 Expected outputs and failure interpretation

The diagnostics command MUST emit canonical finding codes so incidents can be triaged quickly.

Examples:
- `FAIL: downloader_missing_fallback_in_use`
  - Meaning: download path is unavailable and runtime is using a fallback behavior not approved for v0.1.
- `FAIL: inventory_root_mismatch(app=/x, pipeline=/y)`
  - Meaning: application inventory root differs from pipeline effective root.
- `FAIL: package_commit_drift(expected=abc123, actual=def456)`
  - Meaning: deployed package/commit differs from expected release lock.
- `WARN: db_contains_stale_models`
  - Meaning: DB contains model metadata not present in runtime inventory.

Pass criteria:
- No `FAIL:*` findings.
- Effective root and inventory root are identical.
- Downloader service is available.
- Class source paths resolve to the intended installed package tree.

### 5.3 Incident profiles (required runbooks)

#### profile: `path_mismatch`

Commands:

```bash
python -m ai_pipeline.diagnostics --json | jq .
python - <<'PY'
import os
print("AI_PIPELINE_MODELS_DIR", os.getenv("AI_PIPELINE_MODELS_DIR"))
print("HF_HOME", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE", os.getenv("TRANSFORMERS_CACHE"))
PY
```

Good looks like:
- Diagnostics `status` is `PASS`.
- `cache.effective_root == inventory.root`.
- Decision trace selects exactly one winning source consistent with §3.3.

#### profile: `stale_db_overrides`

Commands:

```bash
python -m ai_pipeline.diagnostics --json | jq '.findings'
curl -sS http://localhost:8000/api/v1/ai/models/available | jq .
```

Good looks like:
- Any stale DB rows produce `WARN: db_contains_stale_models` at most.
- Endpoint availability is runtime-derived; stale rows are marked stale or excluded.
- No model is marked selectable for inference solely from DB snapshot.

#### profile: `package_commit_drift`

Commands:

```bash
python -m ai_pipeline.diagnostics --json | jq '.package'
python - <<'PY'
import importlib.metadata as md
print(md.version("ai-pipeline-thin"))
PY
git rev-parse --short HEAD
```

Good looks like:
- Deployed package version and commit match release expectation.
- Source file paths in diagnostics point to the active environment, not an unintended editable install.

### 5.4 CI gating requirements

CI MUST run:

```bash
python -m ai_pipeline.diagnostics --json > diagnostics.json
```

Gate policy:
1. Fail build when `status != "PASS"`.
2. Fail build when any finding begins with `FAIL:`.
3. Allow warnings, but publish them in CI summary.
4. Archive `diagnostics.json` as a build artifact.

---

## 6) Minimal Integration Test Plan

### Test 1: Download → Discoverability → Selection → Inference roundtrip

1. Call pipeline download service for a known model ID.
2. Query `AvailableModels.build()` and assert model appears.
3. Call interaction endpoint selecting that model.
4. Assert response includes same `model_id` and non-empty generated `response`.
5. Assert no fallback model was substituted.

### Test 2: Stale DB rows do not override runtime truth

1. Insert DB row for model not present in runtime cache.
2. Call `/api/v1/ai/models/available`.
3. Assert row is excluded or marked stale.
4. Attempt inference selecting stale-only model and assert deterministic validation error.

### Test 3: Cache path precedence + deterministic Docker behavior

1. Start container with conflicting env vars (`AI_PIPELINE_MODELS_DIR`, `HF_HOME`, `TRANSFORMERS_CACHE`).
2. Assert effective cache root resolves by precedence in §3.3.
3. Download model and assert filesystem writes only under resolved root.
4. Assert inventory and inference use same root.

---

## 7) Non-goals

1. Defining provider-specific semantics beyond current Hugging Face-oriented behavior.
2. Standardizing UI presentation fields or front-end model grouping.
3. Replacing application authorization, rate limiting, or tenancy boundaries.
4. Defining long-term registry/distribution architecture for model artifacts.

---

## 8) Migration Notes (Current Behavior → Contract)

1. **Current package default cache path** is `<package_dir>/models`; contract keeps this as fallback but introduces explicit env precedence for deterministic deployments.
2. **Current model ID handling is effectively opaque** in selection flow; contract codifies this and forbids silent rewriting.
3. **Current inventory is filesystem-based via cache folder parsing**; contract formalizes runtime-truth requirement for app endpoint.
4. **Current error wrapping already exposes stable keys** (`exception_class`, `message`, and for download failures `model_id`); contract marks these as compatibility-guaranteed.
5. **Current app behavior may rely on DB snapshots**; contract requires DB to be metadata-only for availability.

---

## 9) Open Questions

1. Should `AI_PIPELINE_MODELS_DIR` be implemented as a mandatory first-class code path immediately in v0.1.1, or staged behind feature flag?
2. Should inventory expose revision/commit hashes when available in Hugging Face cache metadata?
3. For malformed model IDs, should pipeline raise dedicated `AIPipelineValidationError` (subclass) vs continuing to wrap all in `AIPipelineUpstreamError`?
4. Should provider become a strict enum in request contract for stronger validation?
5. Should `/api/v1/ai/models/available` include a health field (loadable vs merely present on disk)?

---

## 10) Acceptance Criteria Snapshot

This contract is considered implemented when:
1. Public APIs in §3.1 exist and are documented in package docs.
2. Runtime inventory endpoint behavior matches §4.4.
3. Integration tests in §6 pass in CI for at least one containerized profile.
4. Release notes for next `0.1.x` reference this contract as normative.
