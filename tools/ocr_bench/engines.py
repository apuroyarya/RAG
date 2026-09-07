"""OCR engine adapters for the Bengali benchmark.

Every adapter is optional. `available()` returns (bool, reason) so a missing
credential or uninstalled package downgrades that engine to "skipped" instead of
failing the run — you can benchmark whatever you have set up today and add more
later without touching the harness.

Adapters take a rendered PNG path so all engines see byte-identical input. Azure
and Google can both ingest PDFs directly and do their own rasterisation, which
may well score better; that is worth testing separately, but it is not a
controlled comparison.
"""
import os
import time


class Engine:
    name = "base"
    #: rough cost note; verify against current published pricing before quoting
    pricing_note = "check current pricing"

    def available(self):
        raise NotImplementedError

    def ocr(self, png_path):
        raise NotImplementedError

    def run(self, png_path):
        """Returns (text, seconds). Raises on engine error."""
        t0 = time.perf_counter()
        text = self.ocr(png_path)
        return text, time.perf_counter() - t0


class TextLayer(Engine):
    """Not OCR — the PDF's own text layer, as a baseline.

    Included so the report shows what you are replacing. On the two known
    samples this is the corrupt text; its orthographic score should be visibly
    worse than any working OCR engine.
    """
    name = "textlayer"
    pricing_note = "free"

    def __init__(self, pdf_path, page_index):
        self.pdf_path, self.page_index = pdf_path, page_index

    def available(self):
        try:
            import pymupdf  # noqa: F401
            return True, "ok"
        except ImportError:
            return False, "pip install pymupdf"

    def ocr(self, png_path):
        import pymupdf
        doc = pymupdf.open(self.pdf_path)
        try:
            return doc[self.page_index].get_text()
        finally:
            doc.close()


class GoogleVision(Engine):
    """Google Cloud Vision DOCUMENT_TEXT_DETECTION with a Bengali language hint.

    Cheaper and far simpler to set up than Document AI (no processor to create),
    and the right first datapoint. If it wins, retest with Document AI's
    layout-aware processor before committing — it usually handles multi-column
    and table structure better, which matters for chunking.
    """
    name = "google_vision"
    pricing_note = "per-1000-pages, first 1000/month free tier; verify"

    def available(self):
        try:
            from google.cloud import vision  # noqa: F401
        except ImportError:
            return False, "pip install google-cloud-vision"
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            return False, "set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON path"
        return True, "ok"

    def ocr(self, png_path):
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        with open(png_path, "rb") as fh:
            image = vision.Image(content=fh.read())
        resp = client.document_text_detection(
            image=image, image_context={"language_hints": ["bn"]}
        )
        if resp.error.message:
            raise RuntimeError(resp.error.message)
        return resp.full_text_annotation.text


class AzureDocumentIntelligence(Engine):
    """Azure Document Intelligence, prebuilt-read model.

    Bengali is supported for printed text. Set AZURE_DI_ENDPOINT and AZURE_DI_KEY.
    """
    name = "azure_di"
    pricing_note = "per-1000-pages read tier; verify"

    def available(self):
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient  # noqa: F401
        except ImportError:
            return False, "pip install azure-ai-documentintelligence"
        if not (os.environ.get("AZURE_DI_ENDPOINT") and os.environ.get("AZURE_DI_KEY")):
            return False, "set AZURE_DI_ENDPOINT and AZURE_DI_KEY"
        return True, "ok"

    def ocr(self, png_path):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        client = DocumentIntelligenceClient(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_DI_KEY"]),
        )
        with open(png_path, "rb") as fh:
            poller = client.begin_analyze_document("prebuilt-read", body=fh)
        result = poller.result()
        return result.content or ""


class Surya(Engine):
    """Surya — self-hosted, strong on Indic scripts, no per-page cost.

    Heads up: Surya's Python API has changed shape several times across releases.
    If this adapter breaks, check the API for your installed version rather than
    assuming the engine is at fault — the failure message will tell you.
    """
    name = "surya"
    pricing_note = "free, self-hosted (GPU strongly recommended)"

    _predictors = None

    def available(self):
        try:
            import surya  # noqa: F401
            return True, "ok"
        except ImportError:
            return False, "pip install surya-ocr"

    def _load(self):
        if Surya._predictors is None:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            Surya._predictors = (DetectionPredictor(), RecognitionPredictor())
        return Surya._predictors

    def ocr(self, png_path):
        from PIL import Image
        det, rec = self._load()
        image = Image.open(png_path)
        preds = rec([image], det_predictor=det)
        lines = []
        for page in preds:
            for line in getattr(page, "text_lines", []):
                lines.append(line.text)
        return "\n".join(lines)


class Tesseract(Engine):
    """Tesseract with the Bengali model — the pessimistic baseline.

    Included to quantify how much the paid engines actually buy you. Expect it
    to struggle badly on conjuncts; if it does not, you can save real money.
    Needs the `ben` traineddata installed alongside the binary.
    """
    name = "tesseract"
    pricing_note = "free, self-hosted"

    def available(self):
        try:
            import pytesseract
        except ImportError:
            return False, "pip install pytesseract (and install the tesseract binary)"
        try:
            langs = pytesseract.get_languages(config="")
        except Exception as exc:
            return False, f"tesseract binary not found: {exc}"
        if "ben" not in langs:
            return False, "tesseract found but the 'ben' language pack is missing"
        return True, "ok"

    def ocr(self, png_path):
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(png_path), lang="ben")


#: name -> factory. TextLayer is constructed per page, so it is handled separately.
OCR_ENGINES = {
    e.name: e
    for e in (GoogleVision, AzureDocumentIntelligence, Surya, Tesseract)
}
