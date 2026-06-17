# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import atexit
import gc
import os
import time
import json
import threading
from contextlib import asynccontextmanager, contextmanager

import pypdfium2 as pdfium
from loguru import logger
from tqdm import tqdm

from .utils import enable_custom_logits_processors, set_default_gpu_memory_utilization, set_default_batch_size, \
    set_lmdeploy_backend, mod_kwargs_by_device_type
from .model_output_to_middle_json import (
    append_page_blocks_to_middle_json,
    finalize_middle_json,
    init_middle_json,
)
from mineru.backend.utils.runtime_utils import exclude_progress_bar_idle_time
from ...data.data_reader_writer import DataWriter
from mineru.utils.pdf_image_tools import (
    aio_load_images_from_pdf_bytes_range,
    load_images_from_pdf_doc,
)
from ...utils.check_sys_env import is_mac_os_version_supported
from ...utils.config_reader import get_device, get_processing_window_size

from ...utils.enum_class import ImageType
from ...utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)
from ...utils.models_download_utils import auto_download_and_get_model_root_path

from mineru_vl_utils import MinerUClient, MinerUSamplingParams
from mineru_vl_utils.mineru_client import DEFAULT_PROMPTS
from mineru_vl_utils.vlm_client.base_client import DEFAULT_SYSTEM_PROMPT
from packaging import version


_QWEN_MINERU_COMPAT_SYSTEM_PROMPT = (
    "You are MinerU's vision extraction engine. "
    "Always follow the task suffix exactly and return only the requested payload. "
    "Do not output explanations, commentary, markdown fences, or JSON unless explicitly required by the task. "
    "For all ordinary prose, titles, headings, numbering, citations, references, captions, labels, "
    "biomedical abbreviations, units, HR/P values, percentages, and dates, use plain visible text only. "
    "Never wrap ordinary prose or section numbers in <eq>, LaTeX, markdown, HTML, or MinerU protocol tags. "
    "Write isotope notation as plain ASCII prefix text, for example 177Lu, 18F, 225Ac, and 68Ga. "
    "Do not use superscript Unicode or LaTeX wrappers for isotope names. "
    "Layout Detection output contract: return only MinerU layout tokens "
    "(<|box_start|>...<|box_end|><|ref_start|>type<|ref_end|>...), never JSON arrays. "
    "Table Recognition output contract: return exactly one complete HTML table fragment "
    "(<table>...</table>) with rows and cells preserved; no prose; no markdown table pipes; no code fences. "
    "Inside table cells, keep ordinary biomedical text and isotope names plain. "
    "Use <eq>...</eq> in table cells only for true mathematical formulas, never for isotope names, citations, "
    "section numbers, HR/P values, units, percentages, or dates. "
    "Formula Recognition output contract: return only the formula content in clean LaTeX-compatible form; "
    "no natural-language explanation and no surrounding ``` fences. Formula Recognition is only for true formulas. "
    "Image Analysis output contract: return only MinerU-tagged fields "
    "(<|class_start|>...<|class_end|>, <|sub_class_start|>...<|sub_class_end|>, "
    "<|caption_start|>...<|caption_end|>, <|content_start|>...<|content_end|>). "
    "Inside caption and content fields, use plain visible text and plain isotope notation. "
    "For diagrams, charts, and slide-like figures, include the visible labels, legends, axes, "
    "group names, numeric annotations, flow steps, and main trend or comparison inside "
    "<|content_start|>...<|content_end|>. Be detailed enough for downstream review, but do "
    "not invent values or commentary that are not supported by visible content. "
    "Notation consistency: keep nuclear-medicine symbols in one stable, readable form across "
    "body text and table cells; do not switch markup style for the same label in adjacent regions. "
    "Prefer plain characters over math wrappers when the source shows a simple nuclide or tracer name. "
    "If a formula truly requires math markup, keep it compact inside <eq>...</eq> without spaced tokens."
)

_NURAYDIA_QWEN_MINERU_COMPAT_PROMPTS = {
    **DEFAULT_PROMPTS,
    "table": (
        "\nTable Recognition: Return exactly one complete HTML <table>...</table> fragment. "
        "Preserve rows and cells. Do not add prose, markdown table pipes, or code fences. "
        "Use plain text inside cells for isotope names, citations, units, percentages, HR/P values, "
        "and dates. Use <eq>...</eq> only for true mathematical formulas."
    ),
    "equation": (
        "\nFormula Recognition: Return only the formula content in clean LaTeX-compatible form. "
        "Do not include explanations, markdown fences, HTML, or MinerU tags. "
        "Only use this formula style for true standalone formulas."
    ),
    "image": (
        "\nImage Analysis: Return only MinerU image fields "
        "<|class_start|>...<|class_end|><|sub_class_start|>...<|sub_class_end|>"
        "<|caption_start|>...<|caption_end|><|content_start|>...<|content_end|>. "
        "Inside caption/content, use plain visible text and plain isotope notation. "
        "Include visible labels, legends, group names, numeric annotations, and the main "
        "visual comparison or process when present."
    ),
    "image_block": (
        "\nImage Analysis: Return only MinerU image fields "
        "<|class_start|>...<|class_end|><|sub_class_start|>...<|sub_class_end|>"
        "<|caption_start|>...<|caption_end|><|content_start|>...<|content_end|>. "
        "Inside caption/content, use plain visible text and plain isotope notation. "
        "Include visible labels, legends, group names, numeric annotations, and the main "
        "visual comparison or process when present."
    ),
    "chart": (
        "\nImage Analysis: Return only MinerU image fields "
        "<|class_start|>chart<|class_end|><|sub_class_start|>...<|sub_class_end|>"
        "<|caption_start|>...<|caption_end|><|content_start|>...<|content_end|>. "
        "Transcribe chart titles, axes, legends, group labels, visible numeric annotations, "
        "and the main trend or comparison using plain text. Do not fabricate a full data "
        "table from approximate plotted points unless the values are explicitly printed."
    ),
    "[default]": (
        "\nText Recognition: Return only the plain visible text in reading order. "
        "Do not output markdown, HTML/XML tags, MinerU tags, LaTeX, <eq>, or explanations. "
        "Preserve visible punctuation, numbers, citations, abbreviations, HR/P values, and units. "
        "Write isotope notation as plain text, for example 177Lu, 18F, 225Ac, and 68Ga."
    ),
}


def _resolve_vl_system_prompt() -> str:
    raw = os.getenv("MINERU_VL_SYSTEM_PROMPT", "").strip()
    if raw:
        return raw
    profile = os.getenv("MINERU_VL_PROMPT_PROFILE", "").strip().lower()
    if profile in {"qwen-mineru-compat", "mineru-qwen-compat"}:
        return _QWEN_MINERU_COMPAT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT


def _resolve_vl_prompts() -> dict[str, str]:
    profile = os.getenv("MINERU_VL_PROMPT_PROFILE", "").strip().lower()
    if profile in {"qwen-mineru-compat", "mineru-qwen-compat"}:
        return dict(_NURAYDIA_QWEN_MINERU_COMPAT_PROMPTS)
    return dict(DEFAULT_PROMPTS)


def _is_http_client_backend(backend: str) -> bool:
    return backend == "http-client" or backend.endswith("-http-client")


def _resolve_http_client_int_env(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid {}={!r}", key, raw)
        return default
    if value <= 0:
        return default
    return value


def _resolve_http_client_float_env(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid {}={!r}", key, raw)
        return default
    if value < 0:
        return default
    return value


def _resolve_http_client_server_headers(server_headers):
    if server_headers is not None and not isinstance(server_headers, dict):
        return server_headers

    headers = dict(server_headers or {})
    has_authorization = any(key.lower() == "authorization" for key in headers)
    api_key = os.getenv("MINERU_VL_API_KEY", "").strip()
    if api_key and not has_authorization:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers or None


def _resolve_vl_max_new_tokens_for_key(param_key: str, default: int | None) -> int | None:
    env_by_key = {
        "image": "MINERU_VL_IMAGE_MAX_NEW_TOKENS",
        "image_block": "MINERU_VL_IMAGE_MAX_NEW_TOKENS",
        "chart": "MINERU_VL_IMAGE_MAX_NEW_TOKENS",
        "table": "MINERU_VL_TABLE_MAX_NEW_TOKENS",
        "equation": "MINERU_VL_EQUATION_MAX_NEW_TOKENS",
    }
    env_key = env_by_key.get(param_key)
    if env_key:
        raw = os.getenv(env_key, "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError:
                logger.warning("Ignoring invalid {}={!r}", env_key, raw)
                return default
            return value if value > 0 else default
    return default


def _apply_vl_max_new_tokens_from_env(predictor: MinerUClient, backend: str) -> None:
    if not _is_http_client_backend(backend):
        return
    raw = os.getenv("MINERU_VL_MAX_NEW_TOKENS", "").strip()
    limit: int | None = None
    if raw:
        try:
            parsed_limit = int(raw)
        except ValueError:
            logger.warning("Ignoring invalid MINERU_VL_MAX_NEW_TOKENS={!r}", raw)
        else:
            if parsed_limit > 0:
                limit = parsed_limit
    for name, params in list(predictor.sampling_params.items()):
        param_key = name.strip("[]")
        effective_limit = _resolve_vl_max_new_tokens_for_key(param_key, limit)
        if effective_limit is None:
            continue
        predictor.sampling_params[name] = MinerUSamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            repetition_penalty=params.repetition_penalty,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
            max_new_tokens=effective_limit,
        )
    logger.info(
        "http-client VLM max_new_tokens applied: global={} image/chart={} table={} equation={} backend={}",
        limit,
        os.getenv("MINERU_VL_IMAGE_MAX_NEW_TOKENS", "").strip() or None,
        os.getenv("MINERU_VL_TABLE_MAX_NEW_TOKENS", "").strip() or None,
        os.getenv("MINERU_VL_EQUATION_MAX_NEW_TOKENS", "").strip() or None,
        backend,
    )


class ModelSingleton:
    _instance = None
    _models = {}
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        backend: str,
        model_path: str | None,
        server_url: str | None,
        vl_system_prompt: str | None = None,
        **kwargs,
    ) -> MinerUClient:
        key = (backend, model_path, server_url, vl_system_prompt)
        with self._lock:
            if key not in self._models:
                start_time = time.time()
                model = None
                processor = None
                vllm_llm = None
                lmdeploy_engine = None
                vllm_async_llm = None
                batch_size = kwargs.get("batch_size", 0)  # for transformers backend only
                max_concurrency = kwargs.get(
                    "max_concurrency",
                    _resolve_http_client_int_env("MINERU_VL_MAX_CONCURRENCY", 100),
                )
                http_timeout = kwargs.get(
                    "http_timeout",
                    _resolve_http_client_int_env("MINERU_VL_HTTP_TIMEOUT", 600),
                )
                max_retries = kwargs.get(
                    "max_retries",
                    _resolve_http_client_int_env("MINERU_VL_MAX_RETRIES", 1),
                )
                retry_backoff_factor = kwargs.get(
                    "retry_backoff_factor",
                    _resolve_http_client_float_env("MINERU_VL_RETRY_BACKOFF_FACTOR", 0.2),
                )
                if _is_http_client_backend(backend):
                    logger.info(
                        "http-client VLM limits: max_concurrency={} http_timeout={}s max_retries={} retry_backoff_factor={}",
                        max_concurrency,
                        http_timeout,
                        max_retries,
                        retry_backoff_factor,
                    )
                server_headers = _resolve_http_client_server_headers(
                    kwargs.get("server_headers", None)
                )  # for http-client backend only
                # 从kwargs中移除这些参数，避免传递给不相关的初始化函数
                for param in ["batch_size", "max_concurrency", "http_timeout", "server_headers", "max_retries", "retry_backoff_factor"]:
                    if param in kwargs:
                        del kwargs[param]
                client_backend = "http-client" if _is_http_client_backend(backend) else backend
                if client_backend != "http-client" and not model_path:
                    model_path = auto_download_and_get_model_root_path("/","vlm")
                if backend == "transformers":
                    try:
                        from transformers import (
                            AutoProcessor,
                            Qwen2VLForConditionalGeneration,
                        )
                        from transformers import __version__ as transformers_version
                    except ImportError:
                        raise ImportError("Please install transformers to use the transformers backend.")

                    if version.parse(transformers_version) >= version.parse("4.56.0"):
                        dtype_key = "dtype"
                    else:
                        dtype_key = "torch_dtype"
                    device = get_device()
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        device_map={"": device},
                        **{dtype_key: "auto"},  # type: ignore
                    )
                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        use_fast=True,
                    )
                    if batch_size == 0:
                        batch_size = set_default_batch_size()
                elif backend == "mlx-engine":
                    mlx_supported = is_mac_os_version_supported()
                    if not mlx_supported:
                        raise EnvironmentError("mlx-engine backend is only supported on macOS 13.5+ with Apple Silicon.")
                    from mineru_vl_utils.mlx_compat import load_mlx_model
                    model, processor = load_mlx_model(model_path)
                else:
                    if os.getenv('OMP_NUM_THREADS') is None:
                        os.environ["OMP_NUM_THREADS"] = "1"

                    if backend == "vllm-engine":
                        try:
                            import vllm
                        except ImportError:
                            raise ImportError("Please install vllm to use the vllm-engine backend.")

                        kwargs = mod_kwargs_by_device_type(kwargs, vllm_mode="sync_engine")

                        if "compilation_config" in kwargs:
                            if isinstance(kwargs["compilation_config"], str):
                                try:
                                    kwargs["compilation_config"] = json.loads(kwargs["compilation_config"])
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to parse compilation_config as JSON: {kwargs['compilation_config']}")
                                    del kwargs["compilation_config"]
                        if "gpu_memory_utilization" not in kwargs:
                            kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                        if "model" not in kwargs:
                            kwargs["model"] = model_path
                        if enable_custom_logits_processors() and ("logits_processors" not in kwargs):
                            from mineru_vl_utils import MinerULogitsProcessor
                            kwargs["logits_processors"] = [MinerULogitsProcessor]
                        # 使用kwargs为 vllm初始化参数
                        vllm_llm = vllm.LLM(**kwargs)
                    elif backend == "vllm-async-engine":
                        try:
                            from vllm.engine.arg_utils import AsyncEngineArgs
                            from vllm.v1.engine.async_llm import AsyncLLM
                            from vllm.config import CompilationConfig
                        except ImportError:
                            raise ImportError("Please install vllm to use the vllm-async-engine backend.")

                        kwargs = mod_kwargs_by_device_type(kwargs, vllm_mode="async_engine")

                        if "compilation_config" in kwargs:
                            if isinstance(kwargs["compilation_config"], dict):
                                # 如果是字典，转换为 CompilationConfig 对象
                                kwargs["compilation_config"] = CompilationConfig(**kwargs["compilation_config"])
                            elif isinstance(kwargs["compilation_config"], str):
                                # 如果是 JSON 字符串，先解析再转换
                                try:
                                    config_dict = json.loads(kwargs["compilation_config"])
                                    kwargs["compilation_config"] = CompilationConfig(**config_dict)
                                except (json.JSONDecodeError, TypeError) as e:
                                    logger.warning(
                                        f"Failed to parse compilation_config: {kwargs['compilation_config']}, error: {e}")
                                    del kwargs["compilation_config"]
                        if "gpu_memory_utilization" not in kwargs:
                            kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                        if "model" not in kwargs:
                            kwargs["model"] = model_path
                        if enable_custom_logits_processors() and ("logits_processors" not in kwargs):
                            from mineru_vl_utils import MinerULogitsProcessor
                            kwargs["logits_processors"] = [MinerULogitsProcessor]
                        # 使用kwargs为 vllm初始化参数
                        vllm_async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
                    elif backend == "lmdeploy-engine":
                        try:
                            from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig
                            from lmdeploy.serve.vl_async_engine import VLAsyncEngine
                        except ImportError:
                            raise ImportError("Please install lmdeploy to use the lmdeploy-engine backend.")
                        if "cache_max_entry_count" not in kwargs:
                            kwargs["cache_max_entry_count"] = 0.5

                        device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", "")
                        if device_type == "":
                            if "lmdeploy_device" in kwargs:
                                device_type = kwargs.pop("lmdeploy_device")
                                if device_type not in ["cuda", "ascend", "maca", "camb"]:
                                    raise ValueError(f"Unsupported lmdeploy device type: {device_type}")
                            else:
                                device_type = "cuda"
                        lm_backend = os.getenv("MINERU_LMDEPLOY_BACKEND", "")
                        if lm_backend == "":
                            if "lmdeploy_backend" in kwargs:
                                lm_backend = kwargs.pop("lmdeploy_backend")
                                if lm_backend not in ["pytorch", "turbomind"]:
                                    raise ValueError(f"Unsupported lmdeploy backend: {lm_backend}")
                            else:
                                lm_backend = set_lmdeploy_backend(device_type)
                        logger.info(f"lmdeploy device is: {device_type}, lmdeploy backend is: {lm_backend}")

                        if lm_backend == "pytorch":
                            kwargs["device_type"] = device_type
                            backend_config = PytorchEngineConfig(**kwargs)
                        elif lm_backend == "turbomind":
                            backend_config = TurbomindEngineConfig(**kwargs)
                        else:
                            raise ValueError(f"Unsupported lmdeploy backend: {lm_backend}")

                        log_level = 'ERROR'
                        from lmdeploy.utils import get_logger
                        lm_logger = get_logger('lmdeploy')
                        lm_logger.setLevel(log_level)
                        if os.getenv('TM_LOG_LEVEL') is None:
                            os.environ['TM_LOG_LEVEL'] = log_level

                        lmdeploy_engine = VLAsyncEngine(
                            model_path,
                            backend=lm_backend,
                            backend_config=backend_config,
                        )
                predictor = MinerUClient(
                    backend=client_backend,
                    model=model,
                    processor=processor,
                    lmdeploy_engine=lmdeploy_engine,
                    vllm_llm=vllm_llm,
                    vllm_async_llm=vllm_async_llm,
                    server_url=server_url,
                    batch_size=batch_size,
                    max_concurrency=max_concurrency,
                    http_timeout=http_timeout,
                    server_headers=server_headers,
                    max_retries=max_retries,
                    retry_backoff_factor=retry_backoff_factor,
                    system_prompt=vl_system_prompt or _resolve_vl_system_prompt(),
                    prompts=_resolve_vl_prompts(),
                    enable_table_formula_eq_wrap=True,
                    image_analysis=True,
                    enable_cross_page_table_merge=True,
                )
                predictor._mineru_runtime_handles = {
                    "backend": backend,
                    "model": model,
                    "processor": processor,
                    "vllm_llm": vllm_llm,
                    "vllm_async_llm": vllm_async_llm,
                    "lmdeploy_engine": lmdeploy_engine,
                }
                _maybe_enable_serial_execution(predictor, backend)
                _apply_vl_max_new_tokens_from_env(predictor, backend)
                self._models[key] = predictor
                elapsed = round(time.time() - start_time, 2)
                logger.info(f"get {backend} predictor cost: {elapsed}s")
        return self._models[key]

    def shutdown(self) -> None:
        with self._lock:
            predictors = list(self._models.values())
            self._models.clear()

        for predictor in predictors:
            _shutdown_predictor_runtime(predictor)

        gc.collect()


async def _get_model_async(
    backend: str,
    model_path: str | None,
    server_url: str | None,
    vl_system_prompt: str | None = None,
    **kwargs,
) -> MinerUClient:
    return await asyncio.to_thread(
        ModelSingleton().get_model,
        backend,
        model_path,
        server_url,
        vl_system_prompt,
        **kwargs,
    )


def _iter_shutdown_candidates(predictor: MinerUClient):
    runtime_handles = getattr(predictor, "_mineru_runtime_handles", {})
    client = getattr(predictor, "client", None)

    seen_ids = set()

    def _yield_candidate(candidate):
        if candidate is None:
            return
        candidate_id = id(candidate)
        if candidate_id in seen_ids:
            return
        seen_ids.add(candidate_id)
        yield candidate

    for key in ("vllm_llm", "vllm_async_llm", "lmdeploy_engine", "model"):
        yield from _yield_candidate(runtime_handles.get(key))

    if client is not None:
        for key in ("vllm_llm", "vllm_async_llm", "lmdeploy_engine", "model"):
            yield from _yield_candidate(getattr(client, key, None))


def _call_nested_shutdown(target, method_path: str, label: str) -> bool:
    current = target
    for attr in method_path.split("."):
        current = getattr(current, attr, None)
        if current is None:
            return False

    if not callable(current):
        return False

    try:
        current()
        logger.debug(f"Shutdown {label} via `{method_path}`")
        return True
    except TypeError:
        logger.debug(f"Skip unsupported shutdown call {label}.{method_path}")
        return False
    except Exception as exc:
        logger.debug(f"Failed to shutdown {label} via `{method_path}`: {exc}")
        return False


def _shutdown_runtime_handle(handle) -> None:
    for method_path in (
        "shutdown",
        "close",
        "stop",
        "terminate",
        "destroy",
        "engine.shutdown",
        "engine.close",
        "engine_core.shutdown",
        "engine_core.close",
        "llm_engine.shutdown",
        "llm_engine.close",
        "llm_engine.model_executor.shutdown",
        "llm_engine.model_executor.close",
        "model_executor.shutdown",
        "model_executor.close",
    ):
        if _call_nested_shutdown(handle, method_path, type(handle).__name__):
            return


def _clear_predictor_references(predictor: MinerUClient) -> None:
    runtime_handles = getattr(predictor, "_mineru_runtime_handles", {})
    for key in tuple(runtime_handles.keys()):
        runtime_handles[key] = None

    client = getattr(predictor, "client", None)
    if client is not None:
        for attr in ("vllm_llm", "vllm_async_llm", "lmdeploy_engine", "model", "processor"):
            if hasattr(client, attr):
                setattr(client, attr, None)


def _shutdown_predictor_runtime(predictor: MinerUClient) -> None:
    for handle in _iter_shutdown_candidates(predictor):
        _shutdown_runtime_handle(handle)
    _clear_predictor_references(predictor)


def shutdown_cached_models() -> None:
    ModelSingleton().shutdown()


atexit.register(shutdown_cached_models)


def _predictor_uses_mlx(predictor: MinerUClient, backend: str | None = None) -> bool:
    if backend == "mlx-engine":
        return True
    client = getattr(predictor, "client", None)
    return type(client).__module__.endswith(".mlx_client")


def _maybe_enable_serial_execution(
    predictor: MinerUClient,
    backend: str | None = None,
) -> MinerUClient:
    if _predictor_uses_mlx(predictor, backend) and not hasattr(
        predictor, "_mineru_execution_lock"
    ):
        predictor._mineru_execution_lock = threading.Lock()
    return predictor


@contextmanager
def predictor_execution_guard(predictor: MinerUClient):
    lock = getattr(predictor, "_mineru_execution_lock", None)
    if lock is None:
        yield
        return
    with lock:
        yield


@asynccontextmanager
async def aio_predictor_execution_guard(predictor: MinerUClient):
    lock = getattr(predictor, "_mineru_execution_lock", None)
    if lock is None:
        yield
        return
    await asyncio.to_thread(lock.acquire)
    try:
        yield
    finally:
        lock.release()


def _close_images(images_list):
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


def doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs,
):
    client_side_output_generation = bool(
        kwargs.pop("client_side_output_generation", False)
    )
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)
    predictor = _maybe_enable_serial_execution(predictor, backend)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    middle_json = init_middle_json()
    results = []
    doc_closed = False
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
        configured_window_size = get_processing_window_size(default=64)
        effective_window_size = min(page_count, configured_window_size) if page_count else 0
        total_windows = (
            (page_count + effective_window_size - 1) // effective_window_size
            if effective_window_size
            else 0
        )
        logger.info(
            f'VLM processing-window run. page_count={page_count}, '
            f'window_size={configured_window_size}, total_windows={total_windows}'
        )

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_index, window_start in enumerate(range(0, page_count, effective_window_size or 1)):
                window_end = min(page_count - 1, window_start + effective_window_size - 1)
                images_list = load_images_from_pdf_doc(
                    pdf_doc,
                    start_page_id=window_start,
                    end_page_id=window_end,
                    image_type=ImageType.PIL,
                    pdf_bytes=pdf_bytes,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    logger.info(
                        f'VLM processing window {window_index + 1}/{total_windows}: '
                        f'pages {window_start + 1}-{window_end + 1}/{page_count} '
                        f'({len(images_pil_list)} pages)'
                    )
                    with predictor_execution_guard(predictor):
                        window_results = predictor.batch_two_step_extract(
                            images=images_pil_list,
                            image_analysis=image_analysis,
                        )
                    results.extend(window_results)
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(
                            progress_bar,
                            last_append_end_time,
                            now=time.time(),
                        )
                    append_page_blocks_to_middle_json(
                        middle_json,
                        window_results,
                        images_list,
                        pdf_doc,
                        image_writer,
                        page_start_index=window_start,
                        progress_bar=progress_bar,
                    )
                    last_append_end_time = time.time()
                finally:
                    _close_images(images_list)
        finally:
            if progress_bar is not None:
                progress_bar.close()
        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0 and page_count > 0:
            logger.debug(
                f"processing-window infer finished, cost: {infer_time}, "
                f"speed: {round(len(results) / infer_time, 3)} page/s"
            )
        if not client_side_output_generation:
            finalize_middle_json(middle_json["pdf_info"])
        close_pdfium_document(pdf_doc)
        doc_closed = True
        return middle_json, results
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)


async def aio_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs,
):
    client_side_output_generation = bool(
        kwargs.pop("client_side_output_generation", False)
    )
    if predictor is None:
        predictor = await _get_model_async(backend, model_path, server_url, **kwargs)
    predictor = _maybe_enable_serial_execution(predictor, backend)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    middle_json = init_middle_json()
    results = []
    doc_closed = False
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
        configured_window_size = get_processing_window_size(default=64)
        effective_window_size = min(page_count, configured_window_size) if page_count else 0
        total_windows = (
            (page_count + effective_window_size - 1) // effective_window_size
            if effective_window_size
            else 0
        )
        logger.info(
            f'VLM processing-window run. page_count={page_count}, '
            f'window_size={configured_window_size}, total_windows={total_windows}'
        )

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_index, window_start in enumerate(range(0, page_count, effective_window_size or 1)):
                window_end = min(page_count - 1, window_start + effective_window_size - 1)
                images_list = await aio_load_images_from_pdf_bytes_range(
                    pdf_bytes,
                    start_page_id=window_start,
                    end_page_id=window_end,
                    image_type=ImageType.PIL,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    logger.info(
                        f'VLM processing window {window_index + 1}/{total_windows}: '
                        f'pages {window_start + 1}-{window_end + 1}/{page_count} '
                        f'({len(images_pil_list)} pages)'
                    )
                    async with aio_predictor_execution_guard(predictor):
                        window_results = await predictor.aio_batch_two_step_extract(
                            images=images_pil_list,
                            image_analysis=image_analysis,
                        )
                    results.extend(window_results)
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(
                            progress_bar,
                            last_append_end_time,
                            now=time.time(),
                        )
                    append_page_blocks_to_middle_json(
                        middle_json,
                        window_results,
                        images_list,
                        pdf_doc,
                        image_writer,
                        page_start_index=window_start,
                        progress_bar=progress_bar,
                    )
                    last_append_end_time = time.time()
                finally:
                    _close_images(images_list)
        finally:
            if progress_bar is not None:
                progress_bar.close()
        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0 and page_count > 0:
            logger.debug(
                f"processing-window infer finished, cost: {infer_time}, "
                f"speed: {round(len(results) / infer_time, 3)} page/s"
            )
        if not client_side_output_generation:
            await asyncio.to_thread(finalize_middle_json, middle_json["pdf_info"])
        close_pdfium_document(pdf_doc)
        doc_closed = True
        return middle_json, results
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)
