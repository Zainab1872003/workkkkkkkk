import os
import io
import zipfile
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET
from typing import List
import time

# --- Try to use LangChain's Document; if not installed, use a minimal fallback ---
try:
    try:
        from langchain_core.documents import Document  # LangChain v0.1+
    except Exception:
        from langchain.schema import Document          # Older LangChain
except Exception:
    class Document:  # minimal fallback with the same attributes your uploader expects
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# --- OCR availability ---
try:
    import pytesseract  # Requires system Tesseract installed
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

# --- Namespaces for OOXML parts ---
XML_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "ws": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}

def _safe_text(elem):
    if elem is None:
        return ""
    texts = []
    for t in elem.findall(".//a:t", XML_NS):
        texts.append(t.text or "")
    return " ".join(t.strip() for t in texts if t is not None).strip()

def _parse_chart_text(chart_xml_bytes: bytes) -> str:
    try:
        root = ET.fromstring(chart_xml_bytes)
    except ET.ParseError:
        return ""
    parts = []
    for title in root.findall(".//c:title", XML_NS):
        t = _safe_text(title)
        if t:
            parts.append(f"Chart Title: {t}")
    for ax in root.findall(".//c:axTitle", XML_NS):
        t = _safe_text(ax)
        if t:
            parts.append(f"Axis Title: {t}")
    for ser in root.findall(".//c:ser", XML_NS):
        nm = _safe_text(ser.find(".//c:tx", XML_NS))
        if nm:
            parts.append(f"Series: {nm}")
    return " | ".join(parts).strip()

def _parse_shape_text(drawing_xml_bytes: bytes) -> list[str]:
    try:
        root = ET.fromstring(drawing_xml_bytes)
    except ET.ParseError:
        return []
    results = []
    for sp in root.findall(".//xdr:sp", XML_NS):
        t = _safe_text(sp.find(".//xdr:txBody", XML_NS))
        if t:
            results.append(t)
    return results

def _collect_sheet_to_drawing(zipf: zipfile.ZipFile) -> dict:
    sheet_to_drawing = {}
    for name in zipf.namelist():
        if name.startswith("xl/worksheets/_rels/") and name.endswith(".rels"):
            try:
                rel_xml = zipf.read(name)
                rel_root = ET.fromstring(rel_xml)
            except Exception:
                continue
            sheet_basename = os.path.basename(name).replace(".rels", "")
            sheet_path = f"xl/worksheets/{sheet_basename}"
            for rel in rel_root.findall(".//pr:Relationship", XML_NS):
                rtype = rel.attrib.get("Type", "")
                target = rel.attrib.get("Target", "")
                if rtype.endswith("/drawing") and target:
                    if not target.startswith("/"):
                        target_path = os.path.normpath(os.path.join("xl/worksheets/_rels/..", target)).replace("\\", "/")
                        target_path = target_path.replace("worksheets/", "drawings/")
                    else:
                        target_path = target.lstrip("/")
                    if "xl/drawings/" not in target_path:
                        target_path = "xl/drawings/" + os.path.basename(target_path)
                    sheet_to_drawing[sheet_path] = target_path
    return sheet_to_drawing

def _resolve_drawing_rels(zipf: zipfile.ZipFile, drawing_path: str):
    rels_path = drawing_path.replace("xl/drawings/", "xl/drawings/_rels/") + ".rels"
    img_map, chart_map = {}, {}
    if rels_path in zipf.namelist():
        try:
            rel_xml = zipf.read(rels_path)
            rel_root = ET.fromstring(rel_xml)
            for rel in rel_root.findall(".//pr:Relationship", XML_NS):
                rId = rel.attrib.get("Id")
                rtype = rel.attrib.get("Type", "")
                target = rel.attrib.get("Target", "")
                if not rId or not target:
                    continue
                if not target.startswith("/"):
                    target_path = os.path.normpath(os.path.join("xl/drawings/", target)).replace("\\", "/")
                else:
                    target_path = target.lstrip("/")
                if rtype.endswith("/image"):
                    img_map[rId] = target_path
                elif rtype.endswith("/chart"):
                    chart_map[rId] = target_path
        except Exception:
            pass
    return img_map, chart_map

def _extract_drawing_objects(zipf: zipfile.ZipFile, drawing_path: str) -> list[dict]:
    out = []
    if drawing_path not in zipf.namelist():
        return out
    drawing_xml = zipf.read(drawing_path)
    img_map, chart_map = _resolve_drawing_rels(zipf, drawing_path)
    try:
        root = ET.fromstring(drawing_xml)
    except ET.ParseError:
        root = None
    if root is not None:
        # Images
        for blip in root.findall(".//a:blip", XML_NS):
            rid = blip.attrib.get(f"{{{XML_NS['r']}}}embed")
            if rid and rid in img_map:
                img_path = img_map[rid]
                if img_path in zipf.namelist():
                    out.append({"type": "image", "rid": rid, "data": zipf.read(img_path), "path": img_path})
        # Charts
        for ch in root.findall(".//c:chart", XML_NS):
            rid = ch.attrib.get(f"{{{XML_NS['r']}}}id")
            if rid and rid in chart_map:
                ch_path = chart_map[rid]
                if ch_path in zipf.namelist():
                    out.append({"type": "chart", "rid": rid, "data": zipf.read(ch_path), "path": ch_path})
        # Text boxes / shapes
        for t in _parse_shape_text(drawing_xml):
            out.append({"type": "shape_text", "rid": None, "data": t, "path": drawing_path})
    return out

def _ocr_bytes(img_bytes: bytes, lang: str = "eng") -> str:
    if not TESS_AVAILABLE:
        return "[OCR unavailable: pytesseract or Tesseract not installed]"
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            return pytesseract.image_to_string(im, lang=lang).strip()
    except Exception as e:
        return f"[OCR error: {e}]"

def load_and_chunk_excel_with_ocr(
    file_path: str,
    rows_per_chunk: int = 100,
    ocr_lang: str = "eng",
) -> List[Document]:
    """
    Returns a list of Document-like objects with `.page_content` and `.metadata`.
    This fixes 'list' object has no attribute 'metadata' errors in uploaders
    expecting documents.
    """
    filename = os.path.basename(file_path)
    docs: List[Document] = []

    # ---- Table chunks ----
    with pd.ExcelFile(file_path) as xls:
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            total_rows = len(df)
            if total_rows <= 0:
                continue
            for start in range(0, total_rows, rows_per_chunk):
                end = min(start + rows_per_chunk, total_rows)
                chunk_df = df.iloc[start:end]
                chunk_text = (
                    chunk_df.fillna("")
                            .astype(str)
                            .agg(" ".join, axis=1)
                            .str.cat(sep=" ")
                            .strip()
                )
                if chunk_text:
                    docs.append(
                        Document(
                            page_content=chunk_text,
                            metadata={
                                "source": filename,
                                "sheet": sheet_name,
                                "start_row": start + 1,
                                "end_row": end,
                                "object_type": "table_rows",
                            },
                        )
                    )

    # ---- Embedded objects: images, charts, shapes ----
    if file_path.lower().endswith(".xlsx") and zipfile.is_zipfile(file_path):
        try:  
            with zipfile.ZipFile(file_path, "r") as zipf:
                sheet_to_drawing = _collect_sheet_to_drawing(zipf)
                sheet_files = sorted(
                    [p for p in zipf.namelist() if p.startswith("xl/worksheets/sheet") and p.endswith(".xml")]
                )
                try:
                    with pd.ExcelFile(file_path) as xls2:
                        sheet_names = list(xls2.sheet_names)
                except Exception:
                    sheet_names = []

                name_by_sheet_path = {}
                for i, sheet_path in enumerate(sheet_files):
                    sheet_name = sheet_names[i] if i < len(sheet_names) else f"Sheet{i+1}"
                    name_by_sheet_path[sheet_path] = sheet_name

                for sheet_path, drawing_path in sheet_to_drawing.items():
                    sheet_name = name_by_sheet_path.get(sheet_path, os.path.basename(sheet_path))
                    objects = _extract_drawing_objects(zipf, drawing_path)
                    for obj in objects:
                        if obj["type"] == "image":
                            text = _ocr_bytes(obj["data"], lang=ocr_lang).strip()
                            if text:
                                docs.append(
                                    Document(
                                        page_content=text,
                                        metadata={
                                            "source": filename,
                                            "sheet": sheet_name,
                                            "object_type": "image_ocr",
                                            "object_path": obj.get("path"),
                                            "rid": obj.get("rid"),
                                        },
                                    )
                                )
                        elif obj["type"] == "chart":
                            chart_text = _parse_chart_text(obj["data"])
                            if chart_text:
                                docs.append(
                                    Document(
                                        page_content=chart_text,
                                        metadata={
                                            "source": filename,
                                            "sheet": sheet_name,
                                            "object_type": "chart_text",
                                            "object_path": obj.get("path"),
                                            "rid": obj.get("rid"),
                                        },
                                    )
                                )
                        elif obj["type"] == "shape_text":
                            t = (obj["data"] or "").strip()
                            if t:
                                docs.append(
                                    Document(
                                        page_content=t,
                                        metadata={
                                            "source": filename,
                                            "sheet": sheet_name,
                                            "object_type": "shape_text",
                                            "object_path": obj.get("path"),
                                            "rid": obj.get("rid"),
                                        },
                                    )
                                )
            return docs
        except Exception as e:
            print(f"⚠ Excel loading with OCR failed: {e}")
            return docs