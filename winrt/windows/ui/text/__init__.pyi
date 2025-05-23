# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

try:
    import winrt.windows.ui
except Exception:
    pass

class CaretType(enum.IntEnum):
    NORMAL = 0
    NULL = 1

class FindOptions(enum.IntFlag):
    NONE = 0
    WORD = 0x2
    CASE = 0x4

class FontStretch(enum.IntEnum):
    UNDEFINED = 0
    ULTRA_CONDENSED = 1
    EXTRA_CONDENSED = 2
    CONDENSED = 3
    SEMI_CONDENSED = 4
    NORMAL = 5
    SEMI_EXPANDED = 6
    EXPANDED = 7
    EXTRA_EXPANDED = 8
    ULTRA_EXPANDED = 9

class FontStyle(enum.IntEnum):
    NORMAL = 0
    OBLIQUE = 1
    ITALIC = 2

class FormatEffect(enum.IntEnum):
    OFF = 0
    ON = 1
    TOGGLE = 2
    UNDEFINED = 3

class HorizontalCharacterAlignment(enum.IntEnum):
    LEFT = 0
    RIGHT = 1
    CENTER = 2

class LetterCase(enum.IntEnum):
    LOWER = 0
    UPPER = 1

class LineSpacingRule(enum.IntEnum):
    UNDEFINED = 0
    SINGLE = 1
    ONE_AND_HALF = 2
    DOUBLE = 3
    AT_LEAST = 4
    EXACTLY = 5
    MULTIPLE = 6
    PERCENT = 7

class LinkType(enum.IntEnum):
    UNDEFINED = 0
    NOT_A_LINK = 1
    CLIENT_LINK = 2
    FRIENDLY_LINK_NAME = 3
    FRIENDLY_LINK_ADDRESS = 4
    AUTO_LINK = 5
    AUTO_LINK_EMAIL = 6
    AUTO_LINK_PHONE = 7
    AUTO_LINK_PATH = 8

class MarkerAlignment(enum.IntEnum):
    UNDEFINED = 0
    LEFT = 1
    CENTER = 2
    RIGHT = 3

class MarkerStyle(enum.IntEnum):
    UNDEFINED = 0
    PARENTHESIS = 1
    PARENTHESES = 2
    PERIOD = 3
    PLAIN = 4
    MINUS = 5
    NO_NUMBER = 6

class MarkerType(enum.IntEnum):
    UNDEFINED = 0
    NONE = 1
    BULLET = 2
    ARABIC = 3
    LOWERCASE_ENGLISH_LETTER = 4
    UPPERCASE_ENGLISH_LETTER = 5
    LOWERCASE_ROMAN = 6
    UPPERCASE_ROMAN = 7
    UNICODE_SEQUENCE = 8
    CIRCLED_NUMBER = 9
    BLACK_CIRCLE_WINGDING = 10
    WHITE_CIRCLE_WINGDING = 11
    ARABIC_WIDE = 12
    SIMPLIFIED_CHINESE = 13
    TRADITIONAL_CHINESE = 14
    JAPAN_SIMPLIFIED_CHINESE = 15
    JAPAN_KOREA = 16
    ARABIC_DICTIONARY = 17
    ARABIC_ABJAD = 18
    HEBREW = 19
    THAI_ALPHABETIC = 20
    THAI_NUMERIC = 21
    DEVANAGARI_VOWEL = 22
    DEVANAGARI_CONSONANT = 23
    DEVANAGARI_NUMERIC = 24

class ParagraphAlignment(enum.IntEnum):
    UNDEFINED = 0
    LEFT = 1
    CENTER = 2
    RIGHT = 3
    JUSTIFY = 4

class ParagraphStyle(enum.IntEnum):
    UNDEFINED = 0
    NONE = 1
    NORMAL = 2
    HEADING1 = 3
    HEADING2 = 4
    HEADING3 = 5
    HEADING4 = 6
    HEADING5 = 7
    HEADING6 = 8
    HEADING7 = 9
    HEADING8 = 10
    HEADING9 = 11

class PointOptions(enum.IntFlag):
    NONE = 0
    INCLUDE_INSET = 0x1
    START = 0x20
    CLIENT_COORDINATES = 0x100
    ALLOW_OFF_CLIENT = 0x200
    TRANSFORM = 0x400
    NO_HORIZONTAL_SCROLL = 0x10000
    NO_VERTICAL_SCROLL = 0x40000

class RangeGravity(enum.IntEnum):
    U_I_BEHAVIOR = 0
    BACKWARD = 1
    FORWARD = 2
    INWARD = 3
    OUTWARD = 4

class RichEditMathMode(enum.IntEnum):
    NO_MATH = 0
    MATH_ONLY = 1

class SelectionOptions(enum.IntFlag):
    START_ACTIVE = 0x1
    AT_END_OF_LINE = 0x2
    OVERTYPE = 0x4
    ACTIVE = 0x8
    REPLACE = 0x10

class SelectionType(enum.IntEnum):
    NONE = 0
    INSERTION_POINT = 1
    NORMAL = 2
    INLINE_SHAPE = 7
    SHAPE = 8

class TabAlignment(enum.IntEnum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2
    DECIMAL = 3
    BAR = 4

class TabLeader(enum.IntEnum):
    SPACES = 0
    DOTS = 1
    DASHES = 2
    LINES = 3
    THICK_LINES = 4
    EQUALS = 5

class TextDecorations(enum.IntFlag):
    NONE = 0
    UNDERLINE = 0x1
    STRIKETHROUGH = 0x2

class TextGetOptions(enum.IntFlag):
    NONE = 0
    ADJUST_CRLF = 0x1
    USE_CRLF = 0x2
    USE_OBJECT_TEXT = 0x4
    ALLOW_FINAL_EOP = 0x8
    NO_HIDDEN = 0x20
    INCLUDE_NUMBERING = 0x40
    FORMAT_RTF = 0x2000
    USE_LF = 0x1000000

class TextRangeUnit(enum.IntEnum):
    CHARACTER = 0
    WORD = 1
    SENTENCE = 2
    PARAGRAPH = 3
    LINE = 4
    STORY = 5
    SCREEN = 6
    SECTION = 7
    WINDOW = 8
    CHARACTER_FORMAT = 9
    PARAGRAPH_FORMAT = 10
    OBJECT = 11
    HARD_PARAGRAPH = 12
    CLUSTER = 13
    BOLD = 14
    ITALIC = 15
    UNDERLINE = 16
    STRIKETHROUGH = 17
    PROTECTED_TEXT = 18
    LINK = 19
    SMALL_CAPS = 20
    ALL_CAPS = 21
    HIDDEN = 22
    OUTLINE = 23
    SHADOW = 24
    IMPRINT = 25
    DISABLED = 26
    REVISED = 27
    SUBSCRIPT = 28
    SUPERSCRIPT = 29
    FONT_BOUND = 30
    LINK_PROTECTED = 31
    CONTENT_LINK = 32

class TextScript(enum.IntEnum):
    UNDEFINED = 0
    ANSI = 1
    EAST_EUROPE = 2
    CYRILLIC = 3
    GREEK = 4
    TURKISH = 5
    HEBREW = 6
    ARABIC = 7
    BALTIC = 8
    VIETNAMESE = 9
    DEFAULT = 10
    SYMBOL = 11
    THAI = 12
    SHIFT_JIS = 13
    G_B2312 = 14
    HANGUL = 15
    BIG5 = 16
    P_C437 = 17
    OEM = 18
    MAC = 19
    ARMENIAN = 20
    SYRIAC = 21
    THAANA = 22
    DEVANAGARI = 23
    BENGALI = 24
    GURMUKHI = 25
    GUJARATI = 26
    ORIYA = 27
    TAMIL = 28
    TELUGU = 29
    KANNADA = 30
    MALAYALAM = 31
    SINHALA = 32
    LAO = 33
    TIBETAN = 34
    MYANMAR = 35
    GEORGIAN = 36
    JAMO = 37
    ETHIOPIC = 38
    CHEROKEE = 39
    ABORIGINAL = 40
    OGHAM = 41
    RUNIC = 42
    KHMER = 43
    MONGOLIAN = 44
    BRAILLE = 45
    YI = 46
    LIMBU = 47
    TAI_LE = 48
    NEW_TAI_LUE = 49
    SYLOTI_NAGRI = 50
    KHAROSHTHI = 51
    KAYAHLI = 52
    UNICODE_SYMBOL = 53
    EMOJI = 54
    GLAGOLITIC = 55
    LISU = 56
    VAI = 57
    N_KO = 58
    OSMANYA = 59
    PHAGS_PA = 60
    GOTHIC = 61
    DESERET = 62
    TIFINAGH = 63

class TextSetOptions(enum.IntFlag):
    NONE = 0
    UNICODE_BIDI = 0x1
    UNLINK = 0x8
    UNHIDE = 0x10
    CHECK_TEXT_LIMIT = 0x20
    FORMAT_RTF = 0x2000
    APPLY_RTF_DOCUMENT_DEFAULTS = 0x4000

class UnderlineType(enum.IntEnum):
    UNDEFINED = 0
    NONE = 1
    SINGLE = 2
    WORDS = 3
    DOUBLE = 4
    DOTTED = 5
    DASH = 6
    DASH_DOT = 7
    DASH_DOT_DOT = 8
    WAVE = 9
    THICK = 10
    THIN = 11
    DOUBLE_WAVE = 12
    HEAVY_WAVE = 13
    LONG_DASH = 14
    THICK_DASH = 15
    THICK_DASH_DOT = 16
    THICK_DASH_DOT_DOT = 17
    THICK_DOTTED = 18
    THICK_LONG_DASH = 19

class VerticalCharacterAlignment(enum.IntEnum):
    TOP = 0
    BASELINE = 1
    BOTTOM = 2

class FontWeight(_winrt.winrt_base):
    ...

class ContentLinkInfo(_winrt.winrt_base):
    ...
    uri: winrt.windows.foundation.Uri
    secondary_text: str
    link_content_kind: str
    id: int
    display_text: str

class FontWeights(_winrt.winrt_base):
    ...
    black: FontWeight
    bold: FontWeight
    extra_black: FontWeight
    extra_bold: FontWeight
    extra_light: FontWeight
    light: FontWeight
    medium: FontWeight
    normal: FontWeight
    semi_bold: FontWeight
    semi_light: FontWeight
    thin: FontWeight

class RichEditTextDocument(ITextDocument, _winrt.winrt_base):
    ...
    undo_limit: int
    default_tab_stop: float
    caret_type: CaretType
    selection: ITextSelection
    ignore_trailing_character_spacing: bool
    alignment_includes_trailing_whitespace: bool
    def apply_display_updates() -> int:
        ...
    def batch_display_updates() -> int:
        ...
    def begin_undo_group() -> None:
        ...
    def can_copy() -> bool:
        ...
    def can_paste() -> bool:
        ...
    def can_redo() -> bool:
        ...
    def can_undo() -> bool:
        ...
    def clear_undo_redo_history() -> None:
        ...
    def end_undo_group() -> None:
        ...
    def get_default_character_format() -> ITextCharacterFormat:
        ...
    def get_default_paragraph_format() -> ITextParagraphFormat:
        ...
    def get_math() -> str:
        ...
    def get_range(start_position: int, end_position: int) -> ITextRange:
        ...
    def get_range_from_point(point: winrt.windows.foundation.Point, options: PointOptions) -> ITextRange:
        ...
    def get_text(options: TextGetOptions) -> str:
        ...
    def load_from_stream(options: TextSetOptions, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def redo() -> None:
        ...
    def save_to_stream(options: TextGetOptions, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def set_default_character_format(value: ITextCharacterFormat) -> None:
        ...
    def set_default_paragraph_format(value: ITextParagraphFormat) -> None:
        ...
    def set_math(value: str) -> None:
        ...
    def set_math_mode(mode: RichEditMathMode) -> None:
        ...
    def set_text(options: TextSetOptions, value: str) -> None:
        ...
    def undo() -> None:
        ...

class RichEditTextRange(ITextRange, _winrt.winrt_base):
    ...
    content_link_info: ContentLinkInfo
    text: str
    start_position: int
    paragraph_format: ITextParagraphFormat
    link: str
    gravity: RangeGravity
    formatted_text: ITextRange
    end_position: int
    character_format: ITextCharacterFormat
    character: int
    length: int
    story_length: int
    def can_paste(format: int) -> bool:
        ...
    def change_case(value: LetterCase) -> None:
        ...
    def collapse(value: bool) -> None:
        ...
    def copy() -> None:
        ...
    def cut() -> None:
        ...
    def delete(unit: TextRangeUnit, count: int) -> int:
        ...
    def end_of(unit: TextRangeUnit, extend: bool) -> int:
        ...
    def expand(unit: TextRangeUnit) -> int:
        ...
    def find_text(value: str, scan_length: int, options: FindOptions) -> int:
        ...
    def get_character_utf32(offset: int) -> int:
        ...
    def get_clone() -> ITextRange:
        ...
    def get_index(unit: TextRangeUnit) -> int:
        ...
    def get_point(horizontal_align: HorizontalCharacterAlignment, vertical_align: VerticalCharacterAlignment, options: PointOptions) -> winrt.windows.foundation.Point:
        ...
    def get_rect(options: PointOptions) -> typing.Tuple[winrt.windows.foundation.Rect, int]:
        ...
    def get_text(options: TextGetOptions) -> str:
        ...
    def get_text_via_stream(options: TextGetOptions, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def in_range(range: ITextRange) -> bool:
        ...
    def in_story(range: ITextRange) -> bool:
        ...
    def insert_image(width: int, height: int, ascent: int, vertical_align: VerticalCharacterAlignment, alternate_text: str, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def is_equal(range: ITextRange) -> bool:
        ...
    def match_selection() -> None:
        ...
    def move(unit: TextRangeUnit, count: int) -> int:
        ...
    def move_end(unit: TextRangeUnit, count: int) -> int:
        ...
    def move_start(unit: TextRangeUnit, count: int) -> int:
        ...
    def paste(format: int) -> None:
        ...
    def scroll_into_view(value: PointOptions) -> None:
        ...
    def set_index(unit: TextRangeUnit, index: int, extend: bool) -> None:
        ...
    def set_point(point: winrt.windows.foundation.Point, options: PointOptions, extend: bool) -> None:
        ...
    def set_range(start_position: int, end_position: int) -> None:
        ...
    def set_text(options: TextSetOptions, value: str) -> None:
        ...
    def set_text_via_stream(options: TextSetOptions, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def start_of(unit: TextRangeUnit, extend: bool) -> int:
        ...

class TextConstants(_winrt.winrt_base):
    ...
    auto_color: winrt.windows.ui.Color
    max_unit_count: int
    min_unit_count: int
    undefined_color: winrt.windows.ui.Color
    undefined_float_value: float
    undefined_font_stretch: FontStretch
    undefined_font_style: FontStyle
    undefined_int32_value: int

class ITextCharacterFormat(_winrt.winrt_base):
    ...
    all_caps: FormatEffect
    background_color: winrt.windows.ui.Color
    bold: FormatEffect
    font_stretch: FontStretch
    font_style: FontStyle
    foreground_color: winrt.windows.ui.Color
    hidden: FormatEffect
    italic: FormatEffect
    kerning: float
    language_tag: str
    link_type: LinkType
    name: str
    outline: FormatEffect
    position: float
    protected_text: FormatEffect
    size: float
    small_caps: FormatEffect
    spacing: float
    strikethrough: FormatEffect
    subscript: FormatEffect
    superscript: FormatEffect
    text_script: TextScript
    underline: UnderlineType
    weight: int
    def get_clone() -> ITextCharacterFormat:
        ...
    def is_equal(format: ITextCharacterFormat) -> bool:
        ...
    def set_clone(value: ITextCharacterFormat) -> None:
        ...

class ITextDocument(_winrt.winrt_base):
    ...
    caret_type: CaretType
    default_tab_stop: float
    selection: ITextSelection
    undo_limit: int
    def apply_display_updates() -> int:
        ...
    def batch_display_updates() -> int:
        ...
    def begin_undo_group() -> None:
        ...
    def can_copy() -> bool:
        ...
    def can_paste() -> bool:
        ...
    def can_redo() -> bool:
        ...
    def can_undo() -> bool:
        ...
    def end_undo_group() -> None:
        ...
    def get_default_character_format() -> ITextCharacterFormat:
        ...
    def get_default_paragraph_format() -> ITextParagraphFormat:
        ...
    def get_range(start_position: int, end_position: int) -> ITextRange:
        ...
    def get_range_from_point(point: winrt.windows.foundation.Point, options: PointOptions) -> ITextRange:
        ...
    def get_text(options: TextGetOptions) -> str:
        ...
    def load_from_stream(options: TextSetOptions, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def redo() -> None:
        ...
    def save_to_stream(options: TextGetOptions, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def set_default_character_format(value: ITextCharacterFormat) -> None:
        ...
    def set_default_paragraph_format(value: ITextParagraphFormat) -> None:
        ...
    def set_text(options: TextSetOptions, value: str) -> None:
        ...
    def undo() -> None:
        ...

class ITextParagraphFormat(_winrt.winrt_base):
    ...
    alignment: ParagraphAlignment
    first_line_indent: float
    keep_together: FormatEffect
    keep_with_next: FormatEffect
    left_indent: float
    line_spacing: float
    line_spacing_rule: LineSpacingRule
    list_alignment: MarkerAlignment
    list_level_index: int
    list_start: int
    list_style: MarkerStyle
    list_tab: float
    list_type: MarkerType
    no_line_number: FormatEffect
    page_break_before: FormatEffect
    right_indent: float
    right_to_left: FormatEffect
    space_after: float
    space_before: float
    style: ParagraphStyle
    tab_count: int
    widow_control: FormatEffect
    def add_tab(position: float, align: TabAlignment, leader: TabLeader) -> None:
        ...
    def clear_all_tabs() -> None:
        ...
    def delete_tab(position: float) -> None:
        ...
    def get_clone() -> ITextParagraphFormat:
        ...
    def get_tab(index: int) -> typing.Tuple[float, TabAlignment, TabLeader]:
        ...
    def is_equal(format: ITextParagraphFormat) -> bool:
        ...
    def set_clone(format: ITextParagraphFormat) -> None:
        ...
    def set_indents(start: float, left: float, right: float) -> None:
        ...
    def set_line_spacing(rule: LineSpacingRule, spacing: float) -> None:
        ...

class ITextRange(_winrt.winrt_base):
    ...
    character: int
    character_format: ITextCharacterFormat
    end_position: int
    formatted_text: ITextRange
    gravity: RangeGravity
    length: int
    link: str
    paragraph_format: ITextParagraphFormat
    start_position: int
    story_length: int
    text: str
    def can_paste(format: int) -> bool:
        ...
    def change_case(value: LetterCase) -> None:
        ...
    def collapse(value: bool) -> None:
        ...
    def copy() -> None:
        ...
    def cut() -> None:
        ...
    def delete(unit: TextRangeUnit, count: int) -> int:
        ...
    def end_of(unit: TextRangeUnit, extend: bool) -> int:
        ...
    def expand(unit: TextRangeUnit) -> int:
        ...
    def find_text(value: str, scan_length: int, options: FindOptions) -> int:
        ...
    def get_character_utf32(offset: int) -> int:
        ...
    def get_clone() -> ITextRange:
        ...
    def get_index(unit: TextRangeUnit) -> int:
        ...
    def get_point(horizontal_align: HorizontalCharacterAlignment, vertical_align: VerticalCharacterAlignment, options: PointOptions) -> winrt.windows.foundation.Point:
        ...
    def get_rect(options: PointOptions) -> typing.Tuple[winrt.windows.foundation.Rect, int]:
        ...
    def get_text(options: TextGetOptions) -> str:
        ...
    def get_text_via_stream(options: TextGetOptions, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def in_range(range: ITextRange) -> bool:
        ...
    def in_story(range: ITextRange) -> bool:
        ...
    def insert_image(width: int, height: int, ascent: int, vertical_align: VerticalCharacterAlignment, alternate_text: str, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def is_equal(range: ITextRange) -> bool:
        ...
    def match_selection() -> None:
        ...
    def move(unit: TextRangeUnit, count: int) -> int:
        ...
    def move_end(unit: TextRangeUnit, count: int) -> int:
        ...
    def move_start(unit: TextRangeUnit, count: int) -> int:
        ...
    def paste(format: int) -> None:
        ...
    def scroll_into_view(value: PointOptions) -> None:
        ...
    def set_index(unit: TextRangeUnit, index: int, extend: bool) -> None:
        ...
    def set_point(point: winrt.windows.foundation.Point, options: PointOptions, extend: bool) -> None:
        ...
    def set_range(start_position: int, end_position: int) -> None:
        ...
    def set_text(options: TextSetOptions, value: str) -> None:
        ...
    def set_text_via_stream(options: TextSetOptions, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def start_of(unit: TextRangeUnit, extend: bool) -> int:
        ...

class ITextSelection(ITextRange, _winrt.winrt_base):
    ...
    options: SelectionOptions
    type: SelectionType
    character: int
    character_format: ITextCharacterFormat
    end_position: int
    formatted_text: ITextRange
    gravity: RangeGravity
    length: int
    link: str
    paragraph_format: ITextParagraphFormat
    start_position: int
    story_length: int
    text: str
    def end_key(unit: TextRangeUnit, extend: bool) -> int:
        ...
    def home_key(unit: TextRangeUnit, extend: bool) -> int:
        ...
    def move_down(unit: TextRangeUnit, count: int, extend: bool) -> int:
        ...
    def move_left(unit: TextRangeUnit, count: int, extend: bool) -> int:
        ...
    def move_right(unit: TextRangeUnit, count: int, extend: bool) -> int:
        ...
    def move_up(unit: TextRangeUnit, count: int, extend: bool) -> int:
        ...
    def type_text(value: str) -> None:
        ...
    def can_paste(format: int) -> bool:
        ...
    def change_case(value: LetterCase) -> None:
        ...
    def collapse(value: bool) -> None:
        ...
    def copy() -> None:
        ...
    def cut() -> None:
        ...
    def delete(unit: TextRangeUnit, count: int) -> int:
        ...
    def end_of(unit: TextRangeUnit, extend: bool) -> int:
        ...
    def expand(unit: TextRangeUnit) -> int:
        ...
    def find_text(value: str, scan_length: int, options: FindOptions) -> int:
        ...
    def get_character_utf32(offset: int) -> int:
        ...
    def get_clone() -> ITextRange:
        ...
    def get_index(unit: TextRangeUnit) -> int:
        ...
    def get_point(horizontal_align: HorizontalCharacterAlignment, vertical_align: VerticalCharacterAlignment, options: PointOptions) -> winrt.windows.foundation.Point:
        ...
    def get_rect(options: PointOptions) -> typing.Tuple[winrt.windows.foundation.Rect, int]:
        ...
    def get_text(options: TextGetOptions) -> str:
        ...
    def get_text_via_stream(options: TextGetOptions, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def in_range(range: ITextRange) -> bool:
        ...
    def in_story(range: ITextRange) -> bool:
        ...
    def insert_image(width: int, height: int, ascent: int, vertical_align: VerticalCharacterAlignment, alternate_text: str, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def is_equal(range: ITextRange) -> bool:
        ...
    def match_selection() -> None:
        ...
    def move(unit: TextRangeUnit, count: int) -> int:
        ...
    def move_end(unit: TextRangeUnit, count: int) -> int:
        ...
    def move_start(unit: TextRangeUnit, count: int) -> int:
        ...
    def paste(format: int) -> None:
        ...
    def scroll_into_view(value: PointOptions) -> None:
        ...
    def set_index(unit: TextRangeUnit, index: int, extend: bool) -> None:
        ...
    def set_point(point: winrt.windows.foundation.Point, options: PointOptions, extend: bool) -> None:
        ...
    def set_range(start_position: int, end_position: int) -> None:
        ...
    def set_text(options: TextSetOptions, value: str) -> None:
        ...
    def set_text_via_stream(options: TextSetOptions, value: winrt.windows.storage.streams.IRandomAccessStream) -> None:
        ...
    def start_of(unit: TextRangeUnit, extend: bool) -> int:
        ...

