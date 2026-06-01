# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import os
import re
import time
from collections import defaultdict

import cv2
import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from mineru_vl_utils import MinerUClient
from mineru_vl_utils.structs import BLOCK_TYPES, BlockType, ExtractResult
from mineru_vl_utils.vlm_client.base_client import DEFAULT_SYSTEM_PROMPT
from mineru_vl_utils.vlm_client.utils import gather_tasks
from tqdm import tqdm

from mineru.backend.hybrid.hybrid_model_output_to_middle_json import (
    apply_server_side_postprocess,
    append_page_model_list_to_middle_json,
    finalize_middle_json,
    init_middle_json,
)
from mineru.backend.utils.runtime_utils import exclude_progress_bar_idle_time
from mineru.backend.pipeline.model_init import HybridModelSingleton
from mineru.backend.vlm.vlm_analyze import (
    ModelSingleton,
    aio_predictor_execution_guard,
    predictor_execution_guard,
    _maybe_enable_serial_execution,
    _get_model_async,
)
from mineru.data.data_reader_writer import DataWriter
from mineru.utils.boxbase import calculate_overlap_area_2_minbox_area_ratio
from mineru.utils.config_reader import get_device, get_processing_window_size
from mineru.utils.enum_class import ImageType, NotExtractType, BlockType as MineruBlockType
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.model_utils import crop_img, get_vram, clean_memory
from mineru.utils.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list, sorted_boxes, merge_det_boxes, \
    update_det_boxes, OcrConfidence
from mineru.utils.pdf_classify import classify
from mineru.utils.pdf_image_tools import (
    aio_load_images_from_pdf_bytes_range,
    load_images_from_pdf_doc,
)
from mineru.utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 让mps可以fallback

LAYOUT_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 8
LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD = 0.8

not_extract_list = [item.value for item in NotExtractType]
HYBRID_OCR_DET_TEXT_TYPES = set(not_extract_list)

BENCH_REMOTE_EXTRACT_TYPES = frozenset(
    {BlockType.TABLE, BlockType.EQUATION, BlockType.IMAGE, BlockType.CHART, "image_block"}
)
BENCH_REMOTE_TEXT_TYPES = frozenset(
    {
        BlockType.TEXT,
        "ocr_text",
        MineruBlockType.DOC_TITLE,
        MineruBlockType.PARAGRAPH_TITLE,
        BlockType.TITLE,
        BlockType.LIST,
        BlockType.REF_TEXT,
        BlockType.PHONETIC,
        BlockType.LIST_ITEM,
    }
)
BENCH_REMOTE_TEXT_PRUNE_THRESHOLD = 0.85
_BENCH_SUSPICIOUS_TEXT_RE = re.compile(
    r"(\\mathsf|\\mathrm|\\mathfrak|\\left|\\right|\^\s*\{|\$\s*\^)",
    re.IGNORECASE,
)


def _bench_dual_predictor_enabled(backend: str, server_url: str | None) -> bool:
    if backend != "http-client":
        return False
    if not (server_url and server_url.strip()):
        return False
    return os.getenv("MINERU_HYBRID_BENCH_DUAL_PREDICTOR", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _resolve_bench_layout_backend() -> str:
    explicit = os.getenv("MINERU_HYBRID_BENCH_LAYOUT_BACKEND", "").strip()
    if explicit:
        return explicit
    return get_vlm_engine("auto", is_async=False)


def _bench_remote_not_extract_list() -> list[str]:
    return [block_type for block_type in BLOCK_TYPES if block_type not in BENCH_REMOTE_EXTRACT_TYPES]


def _apply_remote_extract_outputs(
    layout_results: list[ExtractResult],
    all_indices: list[tuple[int, int]],
    outputs,
) -> None:
    for (img_idx, idx), output in zip(all_indices, outputs):
        layout_results[img_idx][idx].content = output.text
        layout_results[img_idx][idx].scored = output.scored


def _bbox_area(bbox) -> float:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0,
        float(bbox[3]) - float(bbox[1]),
    )


def _bbox_overlap_ratio_in_first_bbox(first_bbox, second_bbox) -> float:
    first_area = _bbox_area(first_bbox)
    if first_area <= 0:
        return 0.0
    x0 = max(float(first_bbox[0]), float(second_bbox[0]))
    y0 = max(float(first_bbox[1]), float(second_bbox[1]))
    x1 = min(float(first_bbox[2]), float(second_bbox[2]))
    y1 = min(float(first_bbox[3]), float(second_bbox[3]))
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return overlap / first_area


def _is_nested_in_remote_visual_block(
    text_block: dict,
    visual_block: dict,
    *,
    threshold: float,
) -> bool:
    text_bbox = text_block.get("bbox")
    visual_bbox = visual_block.get("bbox")
    if not (
        isinstance(text_bbox, list)
        and len(text_bbox) == 4
        and isinstance(visual_bbox, list)
        and len(visual_bbox) == 4
    ):
        return False
    if _bbox_area(visual_bbox) <= _bbox_area(text_bbox):
        return False
    return _bbox_overlap_ratio_in_first_bbox(text_bbox, visual_bbox) >= threshold


def _prune_text_blocks_nested_in_remote_visuals(model_list: list[list[dict]]) -> int:
    removed = 0
    for page_blocks in model_list:
        remote_visual_blocks = [
            block
            for block in page_blocks
            if block.get("type") in BENCH_REMOTE_EXTRACT_TYPES
        ]
        if not remote_visual_blocks:
            continue

        kept_blocks = []
        for block in page_blocks:
            block_type = block.get("type")
            if block_type not in BENCH_REMOTE_TEXT_TYPES:
                kept_blocks.append(block)
                continue
            if any(
                _is_nested_in_remote_visual_block(
                    block,
                    visual_block,
                    threshold=BENCH_REMOTE_TEXT_PRUNE_THRESHOLD,
                )
                for visual_block in remote_visual_blocks
            ):
                removed += 1
                continue
            kept_blocks.append(block)
        page_blocks[:] = kept_blocks
    if removed:
        logger.info("Bench dual-predictor pruned nested visual text blocks: {}", removed)
    return removed


def _bench_remote_fail_open_types() -> set[str]:
    raw = os.getenv("MINERU_HYBRID_BENCH_REMOTE_FAIL_OPEN_TYPES", "image,chart").strip()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _is_bench_remote_fail_open_block(block_type: str) -> bool:
    return block_type in _bench_remote_fail_open_types()


def _format_bench_remote_extract_error(exc: BaseException) -> str:
    message = str(exc).strip()
    exc_type = type(exc).__name__
    return f"{exc_type}: {message}" if message else f"{exc_type}: {repr(exc)}"


def _describe_bench_remote_extract(
    layout_results: list[ExtractResult],
    remote_index: int,
    remote_total: int,
    img_idx: int,
    block_idx: int,
    image,
) -> dict:
    block = layout_results[img_idx][block_idx]
    image_shape = getattr(image, "shape", None)
    return {
        "remote_index": remote_index,
        "remote_total": remote_total,
        "page_index": img_idx,
        "block_index": block_idx,
        "block_type": block.type,
        "bbox": block.bbox,
        "image_shape": tuple(image_shape) if image_shape is not None else None,
    }


async def _aio_batch_predict_bench_remote_resilient(
    extract_predictor: MinerUClient,
    layout_results: list[ExtractResult],
    all_images,
    all_prompts,
    all_params,
    all_indices,
    semaphore: asyncio.Semaphore,
) -> None:
    total = len(all_images)
    for remote_index, (image, prompt, params, (img_idx, block_idx)) in enumerate(
        zip(all_images, all_prompts, all_params, all_indices),
        start=1,
    ):
        descriptor = _describe_bench_remote_extract(
            layout_results,
            remote_index,
            total,
            img_idx,
            block_idx,
            image,
        )
        logger.info("Bench remote extract start: {}", descriptor)
        try:
            predicted = await extract_predictor._aio_batch_predict(
                [image],
                [prompt],
                [params],
                None,
                semaphore,
                None,
                use_tqdm=False,
                tqdm_desc="Extraction",
            )
        except Exception as exc:
            error_text = _format_bench_remote_extract_error(exc)
            descriptor["error"] = error_text
            if _is_bench_remote_fail_open_block(descriptor["block_type"]):
                block = layout_results[img_idx][block_idx]
                block.content = ""
                block["bench_remote_error"] = error_text
                block["bench_remote_failed_open"] = True
                logger.warning("Bench remote extract failed open: {}", descriptor)
                continue
            logger.exception("Bench remote extract failed: {}", descriptor)
            raise
        _apply_remote_extract_outputs(
            layout_results,
            [(img_idx, block_idx)],
            predicted,
        )
        logger.info("Bench remote extract complete: {}", descriptor)


def _bench_remote_text_repair_enabled() -> bool:
    raw = os.getenv("MINERU_HYBRID_BENCH_REMOTE_TEXT_REPAIR", "").strip().lower()
    if raw == "":
        return True
    return raw in {"1", "true", "yes"}


def _bench_remote_text_repair_max_blocks() -> int:
    raw = os.getenv("MINERU_HYBRID_BENCH_REMOTE_TEXT_REPAIR_MAX_BLOCKS", "").strip()
    if not raw:
        return 8
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid MINERU_HYBRID_BENCH_REMOTE_TEXT_REPAIR_MAX_BLOCKS={!r}, using default 8",
            raw,
        )
        return 8
    return max(1, value)


def _crop_normalized_bbox_from_page_image(page_image, bbox):
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    width, height = page_image.size
    left = max(0, min(width, int(x0 * width)))
    top = max(0, min(height, int(y0 * height)))
    right = max(0, min(width, int(x1 * width)))
    bottom = max(0, min(height, int(y1 * height)))
    if right <= left or bottom <= top:
        return None
    return page_image.crop((left, top, right, bottom))


def _build_suspicious_text_repair_candidates(
    images,
    page_model_list,
    max_blocks: int,
) -> list[tuple[dict, object, str]]:
    candidates: list[tuple[dict, object, str]] = []
    for page_image, page_blocks in zip(images, page_model_list):
        for block in page_blocks:
            block_type = block.get("type")
            if block_type not in BENCH_REMOTE_TEXT_TYPES:
                continue
            text_field = "text" if block_type == "ocr_text" else "content"
            content = block.get(text_field)
            if not isinstance(content, str) or not content.strip():
                continue
            if not _BENCH_SUSPICIOUS_TEXT_RE.search(content):
                continue
            crop = _crop_normalized_bbox_from_page_image(page_image, block.get("bbox"))
            if crop is None:
                continue
            candidates.append((block, crop, text_field))
            if len(candidates) >= max_blocks:
                return candidates
    return candidates


def _repair_suspicious_text_blocks_with_remote(
    extract_predictor: MinerUClient,
    images,
    page_model_list,
) -> None:
    if not _bench_remote_text_repair_enabled():
        return
    candidates = _build_suspicious_text_repair_candidates(
        images,
        page_model_list,
        _bench_remote_text_repair_max_blocks(),
    )
    if not candidates:
        return
    outputs = extract_predictor.batch_content_extract(
        [crop for _, crop, _ in candidates],
        ["text"] * len(candidates),
    )
    replaced = 0
    for (block, _, text_field), output in zip(candidates, outputs):
        if output is None:
            continue
        text = str(output).strip()
        if not text:
            continue
        block[text_field] = text
        replaced += 1
    logger.info(
        "Bench dual-predictor remote text repairs: candidates={} replaced={}",
        len(candidates),
        replaced,
    )


async def _aio_repair_suspicious_text_blocks_with_remote(
    extract_predictor: MinerUClient,
    images,
    page_model_list,
    semaphore: asyncio.Semaphore | None = None,
) -> None:
    if not _bench_remote_text_repair_enabled():
        return
    candidates = _build_suspicious_text_repair_candidates(
        images,
        page_model_list,
        _bench_remote_text_repair_max_blocks(),
    )
    if not candidates:
        return
    outputs = await extract_predictor.aio_batch_content_extract(
        [crop for _, crop, _ in candidates],
        ["text"] * len(candidates),
        semaphore=semaphore,
    )
    replaced = 0
    for (block, _, text_field), output in zip(candidates, outputs):
        if output is None:
            continue
        text = str(output).strip()
        if not text:
            continue
        block[text_field] = text
        replaced += 1
    logger.info(
        "Bench dual-predictor remote text repairs: candidates={} replaced={}",
        len(candidates),
        replaced,
    )


def _summarize_remote_prompt_mix(prompts: list[str]) -> dict[str, int]:
    mix = {"table": 0, "equation": 0, "image_analysis": 0, "other": 0}
    for prompt in prompts:
        if "Table Recognition" in prompt:
            mix["table"] += 1
        elif "Formula Recognition" in prompt:
            mix["equation"] += 1
        elif "Image Analysis" in prompt:
            mix["image_analysis"] += 1
        else:
            mix["other"] += 1
    return {name: count for name, count in mix.items() if count > 0}


def _dual_predictor_batch_two_step_extract(
    layout_predictor: MinerUClient,
    extract_predictor: MinerUClient,
    images: list,
    *,
    image_analysis: bool = True,
) -> list[ExtractResult]:
    remote_skip = _bench_remote_not_extract_list()
    layout_results = layout_predictor.batch_layout_detect(images)
    prepared_inputs = extract_predictor.helper.batch_prepare_for_extract(
        extract_predictor.executor,
        images,
        layout_results,
        remote_skip,
        image_analysis,
    )
    all_images, all_prompts, all_params, all_indices = extract_predictor._flatten_prepared_inputs(
        prepared_inputs
    )
    if all_images:
        logger.info(
            "Bench dual-predictor remote extracts: blocks={} mix={}",
            len(all_images),
            _summarize_remote_prompt_mix(all_prompts),
        )
        outputs = extract_predictor._batch_predict(all_images, all_prompts, all_params, None, None)
        _apply_remote_extract_outputs(layout_results, all_indices, outputs)
    processed_list = extract_predictor.helper.batch_post_process(
        extract_predictor.executor,
        layout_results,
    )
    results = [
        ExtractResult(blocks, layout.layout_scored)
        for layout, blocks in zip(layout_results, processed_list)
    ]
    if extract_predictor.helper.enable_cross_page_table_merge:
        from mineru_vl_utils.post_process.cross_page_table import detect_cross_page_cell_merge

        params = extract_predictor.sampling_params.get("[cross_page_table_merge]")

        def batch_predict_fn(prompts: list[str]) -> list[str]:
            return extract_predictor.client.batch_predict(
                [None] * len(prompts),
                prompts,
                [params] * len(prompts),
            )

        detect_cross_page_cell_merge(results, batch_predict_fn)
    return results


async def _dual_predictor_aio_batch_two_step_extract(
    layout_predictor: MinerUClient,
    extract_predictor: MinerUClient,
    images: list,
    *,
    semaphore: asyncio.Semaphore | None = None,
    image_analysis: bool = True,
) -> list[ExtractResult]:
    remote_skip = _bench_remote_not_extract_list()
    semaphore = semaphore or asyncio.Semaphore(extract_predictor.max_concurrency)
    layout_results = await layout_predictor.aio_batch_layout_detect(images, semaphore=semaphore)
    prepared_inputs = await gather_tasks(
        tasks=[
            extract_predictor.helper.aio_prepare_for_extract(
                extract_predictor.executor,
                image,
                layout_result,
                remote_skip,
                image_analysis,
            )
            for image, layout_result in zip(images, layout_results)
        ],
        use_tqdm=extract_predictor.use_tqdm,
        tqdm_desc="Extract Preparation",
    )
    all_images, all_prompts, all_params, all_indices = extract_predictor._flatten_prepared_inputs(
        prepared_inputs
    )
    if all_images:
        logger.info(
            "Bench dual-predictor remote extracts: blocks={} mix={}",
            len(all_images),
            _summarize_remote_prompt_mix(all_prompts),
        )
        if _bench_remote_fail_open_types():
            await _aio_batch_predict_bench_remote_resilient(
                extract_predictor,
                layout_results,
                all_images,
                all_prompts,
                all_params,
                all_indices,
                semaphore,
            )
        else:
            outputs = await extract_predictor._aio_batch_predict(
                all_images,
                all_prompts,
                all_params,
                None,
                semaphore,
                None,
                use_tqdm=extract_predictor.use_tqdm,
                tqdm_desc="Extraction",
            )
            _apply_remote_extract_outputs(layout_results, all_indices, outputs)
    processed_list = await gather_tasks(
        tasks=[
            extract_predictor.helper.aio_post_process(extract_predictor.executor, layout_result)
            for layout_result in layout_results
        ],
        use_tqdm=extract_predictor.use_tqdm,
        tqdm_desc="Post Processing",
    )
    results = [
        ExtractResult(blocks, layout.layout_scored)
        for layout, blocks in zip(layout_results, processed_list)
    ]
    if extract_predictor.helper.enable_cross_page_table_merge:
        from mineru_vl_utils.post_process.cross_page_table import aio_detect_cross_page_cell_merge

        params = extract_predictor.sampling_params.get("[cross_page_table_merge]")

        async def aio_batch_predict_fn(prompts: list[str]) -> list[str]:
            return await extract_predictor.client.aio_batch_predict(
                [None] * len(prompts),
                prompts,
                [params] * len(prompts),
            )

        await aio_detect_cross_page_cell_merge(results, aio_batch_predict_fn)
    return results


def _resolve_hybrid_predictors(
    backend: str,
    model_path: str | None,
    server_url: str | None,
    predictor: MinerUClient | None,
    kwargs: dict,
) -> tuple[MinerUClient, MinerUClient | None]:
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)
    if not _bench_dual_predictor_enabled(backend, server_url):
        return predictor, None
    layout_backend = _resolve_bench_layout_backend()
    # Keep layout detection on MinerU's native/default prompt so extract-tuning
    # prompt profiles do not perturb block detection behavior.
    layout_predictor = ModelSingleton().get_model(
        layout_backend,
        model_path,
        None,
        vl_system_prompt=DEFAULT_SYSTEM_PROMPT,
        **kwargs,
    )
    logger.info(
        "Bench dual-predictor hybrid: layout backend={} extract backend={} server_url={}",
        layout_backend,
        backend,
        server_url,
    )
    return layout_predictor, predictor


def _is_hybrid_ocr_det_candidate(block):
    """判断 Hybrid 文本类块是否需要 OCR det 生成行级视觉信息。"""
    return (block.get("type") or block.get("label")) in HYBRID_OCR_DET_TEXT_TYPES

def ocr_classify(pdf_bytes, parse_method: str = 'auto',) -> bool:
    # 确定OCR设置
    _ocr_enable = False
    if parse_method == 'auto':
        if classify(pdf_bytes) == 'ocr':
            _ocr_enable = True
    elif parse_method == 'ocr':
        _ocr_enable = True
    return _ocr_enable

def ocr_det(
    hybrid_pipeline_model,
    np_images,
    model_list,
    mfd_res,
    _ocr_enable,
    batch_ratio: int = 1,
    *,
    fill_text: bool = True,
):
    def _set_temp_pixel_bbox(res, pixel_bbox):
        res["_normalized_bbox"] = list(res["bbox"])
        res["bbox"] = pixel_bbox

    def _restore_normalized_bbox(res):
        normalized_bbox = res.pop("_normalized_bbox", None)
        if normalized_bbox is not None:
            res["bbox"] = normalized_bbox

    ocr_res_list = []
    if not hybrid_pipeline_model.enable_ocr_det_batch:
        # 非批处理模式 - 逐页处理
        for np_image, page_mfd_res, page_results in tqdm(
            zip(np_images, mfd_res, model_list),
            total=len(np_images),
            desc="OCR-det"
        ):
            ocr_res_list.append([])
            img_height, img_width = np_image.shape[:2]
            for res in page_results:
                if not _is_hybrid_ocr_det_candidate(res):
                    continue
                x0 = max(0, int(res['bbox'][0] * img_width))
                y0 = max(0, int(res['bbox'][1] * img_height))
                x1 = min(img_width, int(res['bbox'][2] * img_width))
                y1 = min(img_height, int(res['bbox'][3] * img_height))
                if x1 <= x0 or y1 <= y0:
                    continue
                _set_temp_pixel_bbox(res, [x0, y0, x1, y1])
                try:
                    new_image, useful_list = crop_img(
                        res, np_image, crop_paste_x=50, crop_paste_y=50
                    )
                finally:
                    _restore_normalized_bbox(res)
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    page_mfd_res, useful_list
                )
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                ocr_res = hybrid_pipeline_model.ocr_model.ocr(
                    bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False
                )[0]
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(
                        ocr_res,
                        useful_list,
                        _ocr_enable if fill_text else False,
                        bgr_image,
                        hybrid_pipeline_model.lang,
                    )

                    ocr_res_list[-1].extend(ocr_result_list)
    else:
        # 批处理模式 - 按语言和分辨率分组
        # 收集所有需要OCR检测的裁剪图像
        all_cropped_images_info = []

        for np_image, page_mfd_res, page_results in zip(
                np_images, mfd_res, model_list
        ):
            ocr_res_list.append([])
            img_height, img_width = np_image.shape[:2]
            for res in page_results:
                if not _is_hybrid_ocr_det_candidate(res):
                    continue
                x0 = max(0, int(res['bbox'][0] * img_width))
                y0 = max(0, int(res['bbox'][1] * img_height))
                x1 = min(img_width, int(res['bbox'][2] * img_width))
                y1 = min(img_height, int(res['bbox'][3] * img_height))
                if x1 <= x0 or y1 <= y0:
                    continue
                _set_temp_pixel_bbox(res, [x0, y0, x1, y1])
                try:
                    new_image, useful_list = crop_img(
                        res, np_image, crop_paste_x=50, crop_paste_y=50
                    )
                finally:
                    _restore_normalized_bbox(res)
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    page_mfd_res, useful_list
                )
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                all_cropped_images_info.append((
                    bgr_image, useful_list, adjusted_mfdetrec_res, ocr_res_list[-1]
                ))

        # 按分辨率分组并同时完成padding
        RESOLUTION_GROUP_STRIDE = 64  # 32

        resolution_groups = defaultdict(list)
        for crop_info in all_cropped_images_info:
            cropped_img = crop_info[0]
            h, w = cropped_img.shape[:2]
            # 直接计算目标尺寸并用作分组键
            target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            group_key = (target_h, target_w)
            resolution_groups[group_key].append(crop_info)

        # 对每个分辨率组进行批处理
        for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det"):
            # 对所有图像进行padding到统一尺寸
            batch_images = []
            for crop_info in group_crops:
                img = crop_info[0]
                h, w = img.shape[:2]
                # 创建目标尺寸的白色背景
                padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                padded_img[:h, :w] = img
                batch_images.append(padded_img)

            # 批处理检测
            det_batch_size = min(len(batch_images), batch_ratio * OCR_DET_BASE_BATCH_SIZE)
            batch_results = hybrid_pipeline_model.ocr_model.text_detector.batch_predict(batch_images, det_batch_size)

            # 处理批处理结果
            for crop_info, (dt_boxes, _) in zip(group_crops, batch_results):
                bgr_image, useful_list, adjusted_mfdetrec_res, ocr_page_res_list = crop_info

                if dt_boxes is not None and len(dt_boxes) > 0:
                    # 处理检测框
                    dt_boxes_sorted = sorted_boxes(dt_boxes)
                    dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                    # 根据公式位置更新检测框
                    dt_boxes_final = (update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                                      if dt_boxes_merged and adjusted_mfdetrec_res
                                      else dt_boxes_merged)

                    if dt_boxes_final:
                        ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                        ocr_result_list = get_ocr_result_list(
                            ocr_res,
                            useful_list,
                            _ocr_enable if fill_text else False,
                            bgr_image,
                            hybrid_pipeline_model.lang,
                        )
                        ocr_page_res_list.extend(ocr_result_list)
    return ocr_res_list

def mask_image_regions(np_images, model_list):
    # 根据vlm返回的结果，在每一页中将image、table、equation块mask成白色背景图像
    for np_image, vlm_page_results in zip(np_images, model_list):
        img_height, img_width = np_image.shape[:2]
        # 收集需要mask的区域
        mask_regions = []
        for block in vlm_page_results:
            if block['type'] in [BlockType.IMAGE, BlockType.TABLE, BlockType.EQUATION]:
                bbox = block['bbox']
                # 批量转换归一化坐标到像素坐标,并进行边界检查
                x0 = max(0, int(bbox[0] * img_width))
                y0 = max(0, int(bbox[1] * img_height))
                x1 = min(img_width, int(bbox[2] * img_width))
                y1 = min(img_height, int(bbox[3] * img_height))
                # 只添加有效区域
                if x1 > x0 and y1 > y0:
                    mask_regions.append((y0, y1, x0, x1))
        # 批量应用mask
        for y0, y1, x0, x1 in mask_regions:
            np_image[y0:y1, x0:x1, :] = 255
    return np_images


def normalize_bbox_to_unit(item, page_width, page_height):
    """将像素级bbox归一化为[0, 1]区间"""
    bbox = item.get('bbox')
    if bbox is None or len(bbox) != 4:
        return False

    x0, y0, x1, y1 = [float(v) for v in bbox]
    if (
        0.0 <= x0 <= 1.0
        and 0.0 <= y0 <= 1.0
        and 0.0 <= x1 <= 1.0
        and 0.0 <= y1 <= 1.0
    ):
        normalized_bbox = [x0, y0, x1, y1]
    else:
        normalized_bbox = [
            x0 / page_width,
            y0 / page_height,
            x1 / page_width,
            y1 / page_height,
        ]
    item['bbox'] = [round(min(max(v, 0), 1), 3) for v in normalized_bbox]
    return True


def _formula_item_to_pixel_bbox(item):
    bbox = item.get('bbox')
    if bbox is not None and len(bbox) == 4:
        return [int(float(v)) for v in bbox]

    return None


def _build_inline_formula_inputs(images_layout_res):
    inline_formula_inputs = []
    for layout_res in images_layout_res:
        page_inline_formula_inputs = []
        for res in layout_res:
            if res.get('label') not in ['inline_formula', 'display_formula']:
                continue
            bbox = res.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue
            page_inline_formula_inputs.append(
                {
                    "label": "inline_formula",
                    "bbox": list(bbox),
                    "score": float(res.get('score', 0.0)),
                    "latex": res.get('latex', ''),
                }
            )
        inline_formula_inputs.append(page_inline_formula_inputs)
    return inline_formula_inputs


def _build_formula_mask_inputs(images_layout_res):
    """从 layout 检测结果提取公式框，供 OCR det 规避行内/行间公式区域。"""
    page_formula_masks = []
    for layout_res in images_layout_res:
        page_masks = []
        for res in layout_res:
            if res.get('label') not in ['inline_formula', 'display_formula']:
                continue
            bbox = _formula_item_to_pixel_bbox(res)
            if bbox is not None:
                page_masks.append({"bbox": bbox})
        page_formula_masks.append(page_masks)
    return page_formula_masks


def _normalize_page_size(page_image):
    """从PIL或numpy图像中读取页面宽高，供归一化bbox还原为像素bbox。"""
    if hasattr(page_image, "size"):
        return page_image.size

    height, width = page_image.shape[:2]
    return width, height


def _bbox_to_pixel_bbox(bbox, page_size):
    """将归一化或像素bbox统一成像素bbox，异常bbox返回None。"""
    if bbox is None or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    width, height = page_size
    if all(0.0 <= value <= 1.0 for value in [x0, y0, x1, y1]):
        x0, y0, x1, y1 = x0 * width, y0 * height, x1 * width, y1 * height

    left, right = sorted([x0, x1])
    top, bottom = sorted([y0, y1])
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _collect_layout_doc_title_bboxes(layout_res, page_size):
    """只收集layout小模型输出的doc_title框，忽略paragraph_title等其他类型。"""
    doc_title_bboxes = []
    for layout_item in layout_res or []:
        if layout_item.get("label") != MineruBlockType.DOC_TITLE:
            continue
        bbox = _bbox_to_pixel_bbox(layout_item.get("bbox"), page_size)
        if bbox is not None:
            doc_title_bboxes.append(bbox)
    return doc_title_bboxes


def _has_doc_title_overlap(title_bbox, doc_title_bboxes, overlap_threshold):
    """判断VLM标题框是否与任一layout doc_title框达到最小框重叠阈值。"""
    return any(
        calculate_overlap_area_2_minbox_area_ratio(title_bbox, doc_title_bbox)
        >= overlap_threshold
        for doc_title_bbox in doc_title_bboxes
    )


def _apply_layout_title_split(
    model_list,
    images_layout_res,
    page_sizes,
    overlap_threshold=LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD,
):
    """用layout doc_title框将VLM title拆分为doc_title和paragraph_title。"""
    for page_model_list, layout_res, page_size in zip(model_list, images_layout_res, page_sizes):
        doc_title_bboxes = _collect_layout_doc_title_bboxes(layout_res, page_size)
        for block in page_model_list:
            if block.get("type") != MineruBlockType.TITLE:
                continue
            title_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if title_bbox is None:
                continue
            if _has_doc_title_overlap(title_bbox, doc_title_bboxes, overlap_threshold):
                block["type"] = MineruBlockType.DOC_TITLE
            else:
                block["type"] = MineruBlockType.PARAGRAPH_TITLE


def _predict_layout_for_title_split(
    hybrid_pipeline_model,
    images,
    batch_ratio,
):
    """执行layout小模型检测，专门为Hybrid标题拆分提供页面layout结果。"""
    return hybrid_pipeline_model.layout_model.batch_predict(
        images,
        batch_size=min(8, batch_ratio * LAYOUT_BASE_BATCH_SIZE),
    )


def _process_ocr_and_formulas(
    images_pil_list,
    model_list,
    language,
    inline_formula_enable,
    _ocr_enable,
    batch_ratio: int = 1,
):
    """处理OCR和公式识别"""

    # 遍历model_list,对文本块截图交由OCR识别
    # 根据_ocr_enable决定ocr只开det还是det+rec
    # 根据inline_formula_enable决定是使用mfd和ocr结合的方式,还是纯ocr方式

    # 将PIL图片转换为numpy数组
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]

    # 获取混合模型实例
    hybrid_model_singleton = HybridModelSingleton()
    hybrid_pipeline_model = hybrid_model_singleton.get_model(
        lang=language,
        formula_enable=inline_formula_enable,
    )

    # 在进行`行内`公式检测和识别前，先将图像中的图片、表格、`行间`公式区域mask掉
    layout_images = mask_image_regions(np_images, model_list) if inline_formula_enable else np_images
    images_layout_res = _predict_layout_for_title_split(
        hybrid_pipeline_model,
        layout_images,
        batch_ratio,
    )

    if inline_formula_enable:
        images_mfd_res = _build_inline_formula_inputs(images_layout_res)
        # 公式识别
        inline_formula_list = hybrid_pipeline_model.mfr_model.batch_predict(
            images_mfd_res,
            np_images,
            batch_size=batch_ratio * MFR_BASE_BATCH_SIZE,
            interline_enable=True,
        )
    else:
        inline_formula_list = [[] for _ in range(len(images_pil_list))]

    mfd_res = []
    for page_inline_formula_list in inline_formula_list:
        page_mfd_res = []
        for formula in page_inline_formula_list:
            bbox = _formula_item_to_pixel_bbox(formula)
            if bbox is None:
                continue
            page_mfd_res.append({"bbox": bbox})
        mfd_res.append(page_mfd_res)

    # vlm没有执行ocr，需要ocr_det
    ocr_res_list = ocr_det(
        hybrid_pipeline_model,
        np_images,
        model_list,
        mfd_res,
        _ocr_enable,
        batch_ratio=batch_ratio,
    )

    # 如果需要ocr则做ocr_rec
    if _ocr_enable:
        need_ocr_list = []
        img_crop_list = []
        for page_ocr_res_list in ocr_res_list:
            for ocr_res in page_ocr_res_list:
                if 'np_img' in ocr_res:
                    need_ocr_list.append((page_ocr_res_list, ocr_res))
                    img_crop_list.append(ocr_res.pop('np_img'))
        if len(img_crop_list) > 0:
            # Process OCR
            ocr_result_list = hybrid_pipeline_model.ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

            # Verify we have matching counts
            assert len(ocr_result_list) == len(need_ocr_list), f'ocr_result_list: {len(ocr_result_list)}, need_ocr_list: {len(need_ocr_list)}'

            items_to_remove = []
            # Process OCR results for this language
            for index, (page_ocr_res_list, need_ocr_res) in enumerate(need_ocr_list):
                ocr_text, ocr_score = ocr_result_list[index]
                need_ocr_res['text'] = ocr_text
                need_ocr_res['score'] = float(f"{ocr_score:.3f}")
                should_remove = False
                if ocr_score < OcrConfidence.min_confidence:
                    should_remove = True
                else:
                    layout_res_bbox = need_ocr_res.get("bbox")
                    if layout_res_bbox is None and need_ocr_res.get("poly") is not None:
                        layout_res_bbox = [
                            need_ocr_res['poly'][0],
                            need_ocr_res['poly'][1],
                            need_ocr_res['poly'][4],
                            need_ocr_res['poly'][5],
                        ]
                    if layout_res_bbox is None:
                        should_remove = True
                        continue
                    layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
                    layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
                    if (
                            ocr_text in [
                                '（204号', '（20', '（2', '（2号', '（20号', '号','（204',
                                '(cid:)', '(ci:)', '(cd:1)', 'cd:)', 'c)', '(cd:)', 'c', 'id:)',
                                ':)', '√:)', '√i:)', '−i:)', '−:' , 'i:)',
                            ]
                            and ocr_score < 0.8
                            and layout_res_width < layout_res_height
                    ):
                        should_remove = True

                if should_remove:
                    items_to_remove.append((page_ocr_res_list, need_ocr_res))

            for page_ocr_res_list, need_ocr_res in items_to_remove:
                if need_ocr_res in page_ocr_res_list:
                    page_ocr_res_list.remove(need_ocr_res)

    _apply_layout_title_split(
        model_list,
        images_layout_res,
        [_normalize_page_size(image) for image in images_pil_list],
    )

    _normalize_bbox(inline_formula_list, ocr_res_list, images_pil_list)
    merged_model_list = _merge_page_sidecar_items(
        model_list,
        inline_formula_list,
        ocr_res_list,
    )
    return merged_model_list, hybrid_pipeline_model


def _apply_layout_title_split_for_window(
    images_pil_list,
    model_list,
    language,
    batch_ratio,
):
    """为VLM-OCR路径补跑layout小模型，先基于VLM原始title做OCR det，再拆分标题。"""
    hybrid_model_singleton = HybridModelSingleton()
    hybrid_pipeline_model = hybrid_model_singleton.get_model(
        lang=language,
        formula_enable=False,
    )
    images_layout_res = _predict_layout_for_title_split(
        hybrid_pipeline_model,
        images_pil_list,
        batch_ratio,
    )
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
    ocr_res_list = ocr_det(
        hybrid_pipeline_model,
        np_images,
        model_list,
        _build_formula_mask_inputs(images_layout_res),
        False,
        batch_ratio=batch_ratio,
        fill_text=False,
    )
    _normalize_bbox([[] for _ in images_pil_list], ocr_res_list, images_pil_list)
    model_list[:] = _merge_page_sidecar_items(
        model_list,
        [[] for _ in images_pil_list],
        ocr_res_list,
        keep_ocr_text=False,
    )
    _apply_layout_title_split(
        model_list,
        images_layout_res,
        [_normalize_page_size(image) for image in images_pil_list],
    )
    return hybrid_pipeline_model


def _normalize_bbox(
    inline_formula_list,
    ocr_res_list,
    images_pil_list,
):
    """归一化坐标并生成最终结果"""
    for page_inline_formula_list, page_ocr_res_list, page_pil_image in zip(
            inline_formula_list, ocr_res_list, images_pil_list
    ):
        if page_inline_formula_list or page_ocr_res_list:
            page_width, page_height = page_pil_image.size
            # 处理公式列表
            for formula in page_inline_formula_list:
                normalize_bbox_to_unit(formula, page_width, page_height)
            # 处理OCR结果列表
            for ocr_res in page_ocr_res_list:
                normalize_bbox_to_unit(ocr_res, page_width, page_height)


def _build_inline_formula_model_item(formula):
    return {
        "type": "inline_formula",
        "bbox": list(formula["bbox"]),
        "latex": formula.get("latex", ""),
        "score": float(formula.get("score", 0.0)),
    }


def _build_ocr_text_model_item(ocr_res, keep_text=True):
    """构造 OCR det sidecar；VLM-OCR 路径可只保留空文本行提示。"""
    return {
        "type": "ocr_text",
        "bbox": list(ocr_res["bbox"]),
        "text": ocr_res.get("text", "") if keep_text else "",
        "score": float(ocr_res.get("score", 0.0)),
    }


def _merge_page_sidecar_items(
    model_list,
    inline_formula_list,
    ocr_res_list,
    keep_ocr_text=True,
):
    merged_model_list = []
    for page_model_list, page_inline_formula_list, page_ocr_res_list in zip(
            model_list, inline_formula_list, ocr_res_list
    ):
        merged_page_model_list = list(page_model_list)
        merged_page_model_list.extend(
            _build_inline_formula_model_item(formula)
            for formula in page_inline_formula_list
            if formula.get("bbox") is not None
        )
        merged_page_model_list.extend(
            _build_ocr_text_model_item(ocr_res, keep_text=keep_ocr_text)
            for ocr_res in page_ocr_res_list
            if ocr_res.get("bbox") is not None
        )
        merged_model_list.append(merged_page_model_list)
    return merged_model_list


def get_batch_ratio(device):
    """
    根据显存大小或环境变量获取 batch ratio
    """
    # 1. 优先尝试从环境变量获取
    """
    c/s架构分离部署时，建议通过设置环境变量 MINERU_HYBRID_BATCH_RATIO 来指定 batch ratio
    建议的设置值如如下，以下配置值已考虑一定的冗余，单卡多终端部署时为了保证稳定性，可以额外保留一个client端的显存作为整体冗余
    单个client端显存大小 | MINERU_HYBRID_BATCH_RATIO
    ------------------|------------------------
    <= 6   GB         | 8
    <= 4   GB         | 4
    <= 3   GB         | 2
    <= 2   GB         | 1
    例如：
    export MINERU_HYBRID_BATCH_RATIO=4
    """
    env_val = os.getenv("MINERU_HYBRID_BATCH_RATIO")
    if env_val:
        try:
            batch_ratio = int(env_val)
            logger.info(f"hybrid batch ratio (from env): {batch_ratio}")
            return batch_ratio
        except ValueError as e:
            logger.warning(f"Invalid MINERU_HYBRID_BATCH_RATIO value: {env_val}, switching to auto mode. Error: {e}")

    # 2. 根据显存自动推断
    """
    根据总显存大小粗略估计 batch ratio，需要排除掉vllm等推理框架占用的显存开销
    """
    gpu_memory = get_vram(device)
    if gpu_memory >= 32:
        batch_ratio = 16
    elif gpu_memory >= 16:
        batch_ratio = 8
    elif gpu_memory >= 12:
        batch_ratio = 4
    elif gpu_memory >= 8:
        batch_ratio = 2
    else:
        batch_ratio = 1

    logger.info(f"hybrid batch ratio (auto, vram={gpu_memory}GB): {batch_ratio}")
    return batch_ratio


def _should_enable_vlm_ocr(ocr_enable: bool, language: str, inline_formula_enable: bool) -> bool:
    """判断是否启用VLM OCR"""
    force_enable = os.getenv("MINERU_FORCE_VLM_OCR_ENABLE", "0").lower() in ("1", "true", "yes")
    if force_enable:
        return True

    force_pipeline = os.getenv("MINERU_HYBRID_FORCE_PIPELINE_ENABLE", "0").lower() in ("1", "true", "yes")
    return (
            ocr_enable
            and language in ["ch", "en"]
            and inline_formula_enable
            and not force_pipeline
    )


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
        parse_method: str = 'auto',
        language: str = 'ch',
        inline_formula_enable: bool = True,
        model_path: str | None = None,
        server_url: str | None = None,
        image_analysis: bool = True,
        **kwargs,
):
    client_side_output_generation = bool(
        kwargs.pop("client_side_output_generation", False)
    )
    layout_predictor, extract_predictor = _resolve_hybrid_predictors(
        backend,
        model_path,
        server_url,
        predictor,
        kwargs,
    )
    if extract_predictor is not None:
        layout_backend = _resolve_bench_layout_backend()
        layout_predictor = _maybe_enable_serial_execution(layout_predictor, layout_backend)
        extract_predictor = _maybe_enable_serial_execution(extract_predictor, backend)
    else:
        layout_predictor = _maybe_enable_serial_execution(layout_predictor, backend)
    predictor = layout_predictor

    device = get_device()
    _ocr_enable = ocr_classify(pdf_bytes, parse_method=parse_method)
    _vlm_ocr_enable = _should_enable_vlm_ocr(_ocr_enable, language, inline_formula_enable)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    middle_json = init_middle_json(_ocr_enable, _vlm_ocr_enable)
    model_list = []
    doc_closed = False
    hybrid_pipeline_model = None
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
            f'Hybrid processing-window run. page_count={page_count}, '
            f'window_size={configured_window_size}, total_windows={total_windows}'
        )

        batch_ratio = get_batch_ratio(device) if not _vlm_ocr_enable else 1

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
                        f'Hybrid processing window {window_index + 1}/{total_windows}: '
                        f'pages {window_start + 1}-{window_end + 1}/{page_count} '
                        f'({len(images_pil_list)} pages)'
                    )
                    if _vlm_ocr_enable:
                        with predictor_execution_guard(predictor):
                            window_model_list = predictor.batch_two_step_extract(
                                images=images_pil_list,
                                image_analysis=image_analysis,
                            )
                        hybrid_pipeline_model = _apply_layout_title_split_for_window(
                            images_pil_list,
                            window_model_list,
                            language,
                            batch_ratio,
                        )
                    else:
                        if extract_predictor is not None:
                            with predictor_execution_guard(layout_predictor):
                                window_model_list = _dual_predictor_batch_two_step_extract(
                                    layout_predictor,
                                    extract_predictor,
                                    images=images_pil_list,
                                    image_analysis=image_analysis,
                                )
                            _prune_text_blocks_nested_in_remote_visuals(window_model_list)
                        else:
                            with predictor_execution_guard(predictor):
                                window_model_list = predictor.batch_two_step_extract(
                                    images=images_pil_list,
                                    not_extract_list=not_extract_list,
                                    image_analysis=image_analysis,
                                )
                        window_model_list, hybrid_pipeline_model = _process_ocr_and_formulas(
                            images_pil_list,
                            window_model_list,
                            language,
                            inline_formula_enable,
                            _ocr_enable,
                            batch_ratio=batch_ratio,
                        )
                        if extract_predictor is not None:
                            _repair_suspicious_text_blocks_with_remote(
                                extract_predictor,
                                images_pil_list,
                                window_model_list,
                            )

                    model_list.extend(window_model_list)
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(
                            progress_bar,
                            last_append_end_time,
                            now=time.time(),
                        )
                    append_page_model_list_to_middle_json(
                        middle_json,
                        window_model_list,
                        images_list,
                        pdf_doc,
                        image_writer,
                        page_start_index=window_start,
                        _ocr_enable=_ocr_enable,
                        _vlm_ocr_enable=_vlm_ocr_enable,
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
                f"speed: {round(len(model_list) / infer_time, 3)} page/s"
            )

        if client_side_output_generation:
            apply_server_side_postprocess(
                middle_json["pdf_info"],
                hybrid_pipeline_model,
                _ocr_enable,
                _vlm_ocr_enable,
            )
        else:
            finalize_middle_json(
                middle_json["pdf_info"],
                hybrid_pipeline_model,
                _ocr_enable,
                _vlm_ocr_enable,
            )
        close_pdfium_document(pdf_doc)
        doc_closed = True
        clean_memory(device)
        return middle_json, model_list, _vlm_ocr_enable
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)


async def aio_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    parse_method: str = 'auto',
    language: str = 'ch',
    inline_formula_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs,
):
    client_side_output_generation = bool(
        kwargs.pop("client_side_output_generation", False)
    )
    layout_predictor, extract_predictor = await asyncio.to_thread(
        _resolve_hybrid_predictors,
        backend,
        model_path,
        server_url,
        predictor,
        kwargs,
    )
    if extract_predictor is not None:
        layout_backend = _resolve_bench_layout_backend()
        layout_predictor = _maybe_enable_serial_execution(layout_predictor, layout_backend)
        extract_predictor = _maybe_enable_serial_execution(extract_predictor, backend)
    else:
        layout_predictor = _maybe_enable_serial_execution(layout_predictor, backend)
    predictor = layout_predictor

    device = get_device()
    _ocr_enable = ocr_classify(pdf_bytes, parse_method=parse_method)
    _vlm_ocr_enable = _should_enable_vlm_ocr(_ocr_enable, language, inline_formula_enable)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    middle_json = init_middle_json(_ocr_enable, _vlm_ocr_enable)
    model_list = []
    doc_closed = False
    hybrid_pipeline_model = None
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
            f'Hybrid processing-window run. page_count={page_count}, '
            f'window_size={configured_window_size}, total_windows={total_windows}'
        )

        batch_ratio = get_batch_ratio(device) if not _vlm_ocr_enable else 1

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
                        f'Hybrid processing window {window_index + 1}/{total_windows}: '
                        f'pages {window_start + 1}-{window_end + 1}/{page_count} '
                        f'({len(images_pil_list)} pages)'
                    )
                    if _vlm_ocr_enable:
                        async with aio_predictor_execution_guard(predictor):
                            window_model_list = await predictor.aio_batch_two_step_extract(
                                images=images_pil_list,
                                image_analysis=image_analysis,
                            )
                        hybrid_pipeline_model = await asyncio.to_thread(
                            _apply_layout_title_split_for_window,
                            images_pil_list,
                            window_model_list,
                            language,
                            batch_ratio,
                        )
                    else:
                        if extract_predictor is not None:
                            async with aio_predictor_execution_guard(layout_predictor):
                                window_model_list = await _dual_predictor_aio_batch_two_step_extract(
                                    layout_predictor,
                                    extract_predictor,
                                    images=images_pil_list,
                                    image_analysis=image_analysis,
                                )
                            _prune_text_blocks_nested_in_remote_visuals(window_model_list)
                        else:
                            async with aio_predictor_execution_guard(predictor):
                                window_model_list = await predictor.aio_batch_two_step_extract(
                                    images=images_pil_list,
                                    not_extract_list=not_extract_list,
                                    image_analysis=image_analysis,
                                )
                        window_model_list, hybrid_pipeline_model = await asyncio.to_thread(
                            _process_ocr_and_formulas,
                            images_pil_list,
                            window_model_list,
                            language,
                            inline_formula_enable,
                            _ocr_enable,
                            batch_ratio=batch_ratio,
                        )
                        if extract_predictor is not None:
                            await _aio_repair_suspicious_text_blocks_with_remote(
                                extract_predictor,
                                images_pil_list,
                                window_model_list,
                                semaphore=None,
                            )

                    model_list.extend(window_model_list)
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(
                            progress_bar,
                            last_append_end_time,
                            now=time.time(),
                        )
                    append_page_model_list_to_middle_json(
                        middle_json,
                        window_model_list,
                        images_list,
                        pdf_doc,
                        image_writer,
                        page_start_index=window_start,
                        _ocr_enable=_ocr_enable,
                        _vlm_ocr_enable=_vlm_ocr_enable,
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
                f"speed: {round(len(model_list) / infer_time, 3)} page/s"
            )

        if client_side_output_generation:
            await asyncio.to_thread(
                apply_server_side_postprocess,
                middle_json["pdf_info"],
                hybrid_pipeline_model,
                _ocr_enable,
                _vlm_ocr_enable,
            )
        else:
            await asyncio.to_thread(
                finalize_middle_json,
                middle_json["pdf_info"],
                hybrid_pipeline_model,
                _ocr_enable,
                _vlm_ocr_enable,
            )
        close_pdfium_document(pdf_doc)
        doc_closed = True
        clean_memory(device)
        return middle_json, model_list, _vlm_ocr_enable
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)
