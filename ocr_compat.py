"""Compatibility with the legacy RapidOCR release installable on Python 3.13.

1.2.3 CTCLabelDecode.decode averages conf_list + [1e-50], lowering scores
by n/(n+1). Correct it before RapidOCR's text_score filter, not after it.
Remove this adapter when migrating to a supported modern OCR runtime.
"""
from importlib.metadata import version


def restore_legacy_confidence(results):
    return [(text, min(1.0, float(score) * (len(text)+1)/len(text)) if text else 0.0)
            for text, score in results]


def fix_legacy_scores(reader):
    if version('rapidocr_onnxruntime') != '1.2.3':
        return False
    postprocess = reader.text_recognizer.postprocess_op
    original = postprocess.decode
    postprocess.decode = lambda *args, **kw: restore_legacy_confidence(original(*args, **kw))
    return True
