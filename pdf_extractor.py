import io, numpy as np
import PyPDF2
from PIL import Image, ImageOps

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

try:
    import pytesseract
    from pdf2image import convert_from_bytes

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


def _tesseract_available():
    """Check if Tesseract is available and properly configured"""
    if not OCR_AVAILABLE:
        return False, "pytesseract or pdf2image not installed"

    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version} available"
    except pytesseract.TesseractNotFoundError:
        return False, "Tesseract executable not found in PATH"
    except Exception as e:
        return False, f"Tesseract error: {str(e)}"


def _preprocess_pil(img):
    if img.mode != "L":
        img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.point(lambda p: 255 if p > 0x7F else 0, mode="1")
    return img


def _avg_confidence(img, lang, config):
    try:
        data = pytesseract.image_to_data(
            img, lang=lang, config=config, output_type=pytesseract.Output.DICT
        )
        confs = [int(c) for c in data.get("conf", []) if c not in ("-1", "-")]
        return sum(confs) / len(confs) if confs else 0.0
    except Exception:
        return 0.0


def _detect_script(img):
    try:
        osd = pytesseract.image_to_osd(img)
        for line in osd.splitlines():
            if "Script:" in line:
                return line.split(":")[1].strip()
    except Exception:
        pass
    return "Unknown"


def _auto_choose_langs_from_image(img):
    script = _detect_script(img)
    config = "--oem 1 --psm 6"
    if script.lower().startswith("han"):
        sim_conf = _avg_confidence(_preprocess_pil(img), "chi_sim", config)
        tra_conf = _avg_confidence(_preprocess_pil(img), "chi_tra", config)
        return "chi_sim" if sim_conf >= tra_conf else "chi_tra"
    return "eng"


def _auto_choose_langs_for_pdf(file_bytes):
    try:
        first_page = convert_from_bytes(file_bytes, dpi=200, first_page=1, last_page=1)[
            0
        ]
        return _auto_choose_langs_from_image(first_page)
    except Exception:
        return "eng"


def _uppercase_ascii(s):
    return "".join(ch.upper() if "a" <= ch <= "z" else ch for ch in s)


def extract_text_from_pdf(file, ocr_langs=None):
    if hasattr(file, "read"):
        file.seek(0)
        file_bytes = file.read()
    elif isinstance(file, (bytes, bytearray)):
        file_bytes = bytes(file)
    else:
        raise TypeError("`file` must be a file-like object or bytes.")

    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    except Exception:
        text = ""

    if text.strip():
        return text

    if OCR_AVAILABLE:
        available, error_msg = _tesseract_available()
        if available:
            return extract_text_from_pdf_with_ocr(file_bytes, ocr_langs=ocr_langs)
        else:
            return f"OCR not available: {error_msg}. Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki"
    else:
        return "OCR not available. Install pytesseract, pdf2image, and Tesseract."


def extract_text_from_pdf_with_ocr(file_bytes, ocr_langs=None):
    if not OCR_AVAILABLE:
        return "OCR not available. Install pytesseract, pdf2image."
    try:
        images = convert_from_bytes(file_bytes, dpi=300)
    except Exception as e:
        return f"Error converting PDF: {e}"

    if PADDLE_AVAILABLE:
        ocr = PaddleOCR(lang="ch", use_angle_cls=True)
        pages_text = []
        for i, image in enumerate(images):
            np_im = np.array(image)[:, :, ::-1]
            res = ocr.ocr(np_im, cls=True)[0] or []
            lines = [r[1][0] for r in res if r[1][0].strip()]
            if lines:
                pages_text.append(
                    f"--- Page {i + 1} (paddle-ch) ---\n" + "\n".join(lines)
                )
        return "\n\n".join(pages_text) if pages_text else ""

    if _tesseract_available():
        pages_text = []
        config = "--oem 1 --psm 6"
        for i, image in enumerate(images):
            proc = _preprocess_pil(image)
            lang = ocr_langs or _auto_choose_langs_from_image(image)
            page_text = pytesseract.image_to_string(proc, lang=lang, config=config)
            if page_text and page_text.strip():
                pages_text.append(f"--- Page {i + 1} ({lang}) ---\n{page_text.strip()}")
        return "\n\n".join(pages_text) if pages_text else ""
    return "OCR not available. Install Tesseract."


def extract_text_from_image(file, ocr_langs=None):
    if PADDLE_AVAILABLE:
        ocr = PaddleOCR(lang="ch", use_angle_cls=True)
        im = Image.open(file)
        np_im = np.array(im)[:, :, ::-1]
        res = ocr.ocr(np_im, cls=True)[0] or []
        return "\n".join([r[1][0] for r in res if r[1][0].strip()])

    if not (_tesseract_available() and OCR_AVAILABLE):
        return (
            "OCR not available. Install pytesseract and Tesseract with chi_sim/chi_tra."
        )
    try:
        img = Image.open(file)
        proc = _preprocess_pil(img)
        lang = ocr_langs or _auto_choose_langs_from_image(img)
        config = "--oem 1 --psm 6"
        text = pytesseract.image_to_string(proc, lang=lang, config=config)
        return (text or "").strip()
    except Exception as e:
        return f"Error during image OCR: {str(e)}"


def extract_cn_pairs_from_pdfbytes(file_bytes, conf_thres=0.5):
    if not OCR_AVAILABLE:
        return {"pairs": [], "raw": []}
    images = convert_from_bytes(file_bytes, dpi=300)
    if not PADDLE_AVAILABLE:
        return {"pairs": [], "raw": []}

    ocr = PaddleOCR(lang="ch", use_angle_cls=True)
    pairs_all, raw_all = [], []
    for page_idx, image in enumerate(images, 1):
        np_im = np.array(image)[:, :, ::-1]
        result = ocr.ocr(np_im, cls=True)[0] or []
        items = []
        for box, (txt, conf) in result:
            if not txt or not txt.strip():
                continue
            if conf is not None and conf < conf_thres:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            items.append(
                {
                    "text": txt.strip(),
                    "conf": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "h": y2 - y1,
                }
            )
            raw_all.append((txt.strip(), conf))

        if not items:
            continue

        hs = np.array([it["h"] for it in items], dtype=float)
        h_med = float(np.median(hs)) if len(hs) else 0.0
        thresh = max(28, h_med * 1.35)
        labels = [it for it in items if it["h"] < thresh]
        fields = [it for it in items if it["h"] >= thresh]

        for f in fields:
            fx, fy = (f["x1"] + f["x2"]) / 2.0, (f["y1"] + f["y2"]) / 2.0
            best, bestd = None, 1e9
            for lb in labels:
                dx = max(0.0, fx - lb["x2"])
                dy = max(0.0, fy - lb["y2"])
                d = dx + 0.6 * dy
                if d < bestd:
                    bestd, best = d, lb
            value_text = _uppercase_ascii(f["text"])
            pairs_all.append(
                {
                    "page": page_idx,
                    "label": (best["text"] if best else "UNKNOWN_LABEL"),
                    "value": value_text,
                    "conf": f["conf"],
                }
            )
    return {"pairs": pairs_all, "raw": raw_all}
