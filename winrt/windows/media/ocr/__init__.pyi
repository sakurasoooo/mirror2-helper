# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.globalization
except Exception:
    pass

try:
    import winrt.windows.graphics.imaging
except Exception:
    pass

class OcrEngine(_winrt.winrt_base):
    ...
    recognizer_language: winrt.windows.globalization.Language
    available_recognizer_languages: winrt.windows.foundation.collections.IVectorView[winrt.windows.globalization.Language]
    max_image_dimension: int
    def is_language_supported(language: winrt.windows.globalization.Language) -> bool:
        ...
    def recognize_async(bitmap: winrt.windows.graphics.imaging.SoftwareBitmap) -> winrt.windows.foundation.IAsyncOperation[OcrResult]:
        ...
    def try_create_from_language(language: winrt.windows.globalization.Language) -> OcrEngine:
        ...
    def try_create_from_user_profile_languages() -> OcrEngine:
        ...

class OcrLine(_winrt.winrt_base):
    ...
    text: str
    words: winrt.windows.foundation.collections.IVectorView[OcrWord]

class OcrResult(_winrt.winrt_base):
    ...
    lines: winrt.windows.foundation.collections.IVectorView[OcrLine]
    text: str
    text_angle: typing.Optional[float]

class OcrWord(_winrt.winrt_base):
    ...
    bounding_rect: winrt.windows.foundation.Rect
    text: str
