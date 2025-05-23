# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.applicationmodel.datatransfer
except Exception:
    pass

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class PrintBinding(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    NONE = 3
    BALE = 4
    BIND_BOTTOM = 5
    BIND_LEFT = 6
    BIND_RIGHT = 7
    BIND_TOP = 8
    BOOKLET = 9
    EDGE_STITCH_BOTTOM = 10
    EDGE_STITCH_LEFT = 11
    EDGE_STITCH_RIGHT = 12
    EDGE_STITCH_TOP = 13
    FOLD = 14
    JOG_OFFSET = 15
    TRIM = 16

class PrintBordering(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    BORDERED = 3
    BORDERLESS = 4

class PrintCollation(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    COLLATED = 3
    UNCOLLATED = 4

class PrintColorMode(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    COLOR = 3
    GRAYSCALE = 4
    MONOCHROME = 5

class PrintDuplex(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    ONE_SIDED = 3
    TWO_SIDED_SHORT_EDGE = 4
    TWO_SIDED_LONG_EDGE = 5

class PrintHolePunch(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    NONE = 3
    LEFT_EDGE = 4
    RIGHT_EDGE = 5
    TOP_EDGE = 6
    BOTTOM_EDGE = 7

class PrintMediaSize(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    BUSINESS_CARD = 3
    CREDIT_CARD = 4
    ISO_A0 = 5
    ISO_A1 = 6
    ISO_A10 = 7
    ISO_A2 = 8
    ISO_A3 = 9
    ISO_A3_EXTRA = 10
    ISO_A3_ROTATED = 11
    ISO_A4 = 12
    ISO_A4_EXTRA = 13
    ISO_A4_ROTATED = 14
    ISO_A5 = 15
    ISO_A5_EXTRA = 16
    ISO_A5_ROTATED = 17
    ISO_A6 = 18
    ISO_A6_ROTATED = 19
    ISO_A7 = 20
    ISO_A8 = 21
    ISO_A9 = 22
    ISO_B0 = 23
    ISO_B1 = 24
    ISO_B10 = 25
    ISO_B2 = 26
    ISO_B3 = 27
    ISO_B4 = 28
    ISO_B4_ENVELOPE = 29
    ISO_B5_ENVELOPE = 30
    ISO_B5_EXTRA = 31
    ISO_B7 = 32
    ISO_B8 = 33
    ISO_B9 = 34
    ISO_C0 = 35
    ISO_C1 = 36
    ISO_C10 = 37
    ISO_C2 = 38
    ISO_C3 = 39
    ISO_C3_ENVELOPE = 40
    ISO_C4 = 41
    ISO_C4_ENVELOPE = 42
    ISO_C5 = 43
    ISO_C5_ENVELOPE = 44
    ISO_C6 = 45
    ISO_C6_C5_ENVELOPE = 46
    ISO_C6_ENVELOPE = 47
    ISO_C7 = 48
    ISO_C8 = 49
    ISO_C9 = 50
    ISO_D_L_ENVELOPE = 51
    ISO_D_L_ENVELOPE_ROTATED = 52
    ISO_S_R_A3 = 53
    JAPAN2_L_PHOTO = 54
    JAPAN_CHOU3_ENVELOPE = 55
    JAPAN_CHOU3_ENVELOPE_ROTATED = 56
    JAPAN_CHOU4_ENVELOPE = 57
    JAPAN_CHOU4_ENVELOPE_ROTATED = 58
    JAPAN_DOUBLE_HAGAKI_POSTCARD = 59
    JAPAN_DOUBLE_HAGAKI_POSTCARD_ROTATED = 60
    JAPAN_HAGAKI_POSTCARD = 61
    JAPAN_HAGAKI_POSTCARD_ROTATED = 62
    JAPAN_KAKU2_ENVELOPE = 63
    JAPAN_KAKU2_ENVELOPE_ROTATED = 64
    JAPAN_KAKU3_ENVELOPE = 65
    JAPAN_KAKU3_ENVELOPE_ROTATED = 66
    JAPAN_L_PHOTO = 67
    JAPAN_QUADRUPLE_HAGAKI_POSTCARD = 68
    JAPAN_YOU1_ENVELOPE = 69
    JAPAN_YOU2_ENVELOPE = 70
    JAPAN_YOU3_ENVELOPE = 71
    JAPAN_YOU4_ENVELOPE = 72
    JAPAN_YOU4_ENVELOPE_ROTATED = 73
    JAPAN_YOU6_ENVELOPE = 74
    JAPAN_YOU6_ENVELOPE_ROTATED = 75
    JIS_B0 = 76
    JIS_B1 = 77
    JIS_B10 = 78
    JIS_B2 = 79
    JIS_B3 = 80
    JIS_B4 = 81
    JIS_B4_ROTATED = 82
    JIS_B5 = 83
    JIS_B5_ROTATED = 84
    JIS_B6 = 85
    JIS_B6_ROTATED = 86
    JIS_B7 = 87
    JIS_B8 = 88
    JIS_B9 = 89
    NORTH_AMERICA10X11 = 90
    NORTH_AMERICA10X12 = 91
    NORTH_AMERICA10X14 = 92
    NORTH_AMERICA11X17 = 93
    NORTH_AMERICA14X17 = 94
    NORTH_AMERICA4X6 = 95
    NORTH_AMERICA4X8 = 96
    NORTH_AMERICA5X7 = 97
    NORTH_AMERICA8X10 = 98
    NORTH_AMERICA9X11 = 99
    NORTH_AMERICA_ARCHITECTURE_A_SHEET = 100
    NORTH_AMERICA_ARCHITECTURE_B_SHEET = 101
    NORTH_AMERICA_ARCHITECTURE_C_SHEET = 102
    NORTH_AMERICA_ARCHITECTURE_D_SHEET = 103
    NORTH_AMERICA_ARCHITECTURE_E_SHEET = 104
    NORTH_AMERICA_C_SHEET = 105
    NORTH_AMERICA_D_SHEET = 106
    NORTH_AMERICA_E_SHEET = 107
    NORTH_AMERICA_EXECUTIVE = 108
    NORTH_AMERICA_GERMAN_LEGAL_FANFOLD = 109
    NORTH_AMERICA_GERMAN_STANDARD_FANFOLD = 110
    NORTH_AMERICA_LEGAL = 111
    NORTH_AMERICA_LEGAL_EXTRA = 112
    NORTH_AMERICA_LETTER = 113
    NORTH_AMERICA_LETTER_EXTRA = 114
    NORTH_AMERICA_LETTER_PLUS = 115
    NORTH_AMERICA_LETTER_ROTATED = 116
    NORTH_AMERICA_MONARCH_ENVELOPE = 117
    NORTH_AMERICA_NOTE = 118
    NORTH_AMERICA_NUMBER10_ENVELOPE = 119
    NORTH_AMERICA_NUMBER10_ENVELOPE_ROTATED = 120
    NORTH_AMERICA_NUMBER11_ENVELOPE = 121
    NORTH_AMERICA_NUMBER12_ENVELOPE = 122
    NORTH_AMERICA_NUMBER14_ENVELOPE = 123
    NORTH_AMERICA_NUMBER9_ENVELOPE = 124
    NORTH_AMERICA_PERSONAL_ENVELOPE = 125
    NORTH_AMERICA_QUARTO = 126
    NORTH_AMERICA_STATEMENT = 127
    NORTH_AMERICA_SUPER_A = 128
    NORTH_AMERICA_SUPER_B = 129
    NORTH_AMERICA_TABLOID = 130
    NORTH_AMERICA_TABLOID_EXTRA = 131
    OTHER_METRIC_A3_PLUS = 132
    OTHER_METRIC_A4_PLUS = 133
    OTHER_METRIC_FOLIO = 134
    OTHER_METRIC_INVITE_ENVELOPE = 135
    OTHER_METRIC_ITALIAN_ENVELOPE = 136
    PRC10_ENVELOPE = 137
    PRC10_ENVELOPE_ROTATED = 138
    PRC16_K = 139
    PRC16_K_ROTATED = 140
    PRC1_ENVELOPE = 141
    PRC1_ENVELOPE_ROTATED = 142
    PRC2_ENVELOPE = 143
    PRC2_ENVELOPE_ROTATED = 144
    PRC32_K = 145
    PRC32_K_BIG = 146
    PRC32_K_ROTATED = 147
    PRC3_ENVELOPE = 148
    PRC3_ENVELOPE_ROTATED = 149
    PRC4_ENVELOPE = 150
    PRC4_ENVELOPE_ROTATED = 151
    PRC5_ENVELOPE = 152
    PRC5_ENVELOPE_ROTATED = 153
    PRC6_ENVELOPE = 154
    PRC6_ENVELOPE_ROTATED = 155
    PRC7_ENVELOPE = 156
    PRC7_ENVELOPE_ROTATED = 157
    PRC8_ENVELOPE = 158
    PRC8_ENVELOPE_ROTATED = 159
    PRC9_ENVELOPE = 160
    PRC9_ENVELOPE_ROTATED = 161
    ROLL04_INCH = 162
    ROLL06_INCH = 163
    ROLL08_INCH = 164
    ROLL12_INCH = 165
    ROLL15_INCH = 166
    ROLL18_INCH = 167
    ROLL22_INCH = 168
    ROLL24_INCH = 169
    ROLL30_INCH = 170
    ROLL36_INCH = 171
    ROLL54_INCH = 172

class PrintMediaType(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    AUTO_SELECT = 3
    ARCHIVAL = 4
    BACK_PRINT_FILM = 5
    BOND = 6
    CARD_STOCK = 7
    CONTINUOUS = 8
    ENVELOPE_PLAIN = 9
    ENVELOPE_WINDOW = 10
    FABRIC = 11
    HIGH_RESOLUTION = 12
    LABEL = 13
    MULTI_LAYER_FORM = 14
    MULTI_PART_FORM = 15
    PHOTOGRAPHIC = 16
    PHOTOGRAPHIC_FILM = 17
    PHOTOGRAPHIC_GLOSSY = 18
    PHOTOGRAPHIC_HIGH_GLOSS = 19
    PHOTOGRAPHIC_MATTE = 20
    PHOTOGRAPHIC_SATIN = 21
    PHOTOGRAPHIC_SEMI_GLOSS = 22
    PLAIN = 23
    SCREEN = 24
    SCREEN_PAGED = 25
    STATIONERY = 26
    TAB_STOCK_FULL = 27
    TAB_STOCK_PRE_CUT = 28
    TRANSPARENCY = 29
    T_SHIRT_TRANSFER = 30
    NONE = 31

class PrintOrientation(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    PORTRAIT = 3
    PORTRAIT_FLIPPED = 4
    LANDSCAPE = 5
    LANDSCAPE_FLIPPED = 6

class PrintQuality(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    AUTOMATIC = 3
    DRAFT = 4
    FAX = 5
    HIGH = 6
    NORMAL = 7
    PHOTOGRAPHIC = 8
    TEXT = 9

class PrintStaple(enum.IntEnum):
    DEFAULT = 0
    NOT_AVAILABLE = 1
    PRINTER_CUSTOM = 2
    NONE = 3
    STAPLE_TOP_LEFT = 4
    STAPLE_TOP_RIGHT = 5
    STAPLE_BOTTOM_LEFT = 6
    STAPLE_BOTTOM_RIGHT = 7
    STAPLE_DUAL_LEFT = 8
    STAPLE_DUAL_RIGHT = 9
    STAPLE_DUAL_TOP = 10
    STAPLE_DUAL_BOTTOM = 11
    SADDLE_STITCH = 12

class PrintTaskCompletion(enum.IntEnum):
    ABANDONED = 0
    CANCELED = 1
    FAILED = 2
    SUBMITTED = 3

class PrintPageDescription(_winrt.winrt_base):
    ...

class PrintManager(_winrt.winrt_base):
    ...
    def get_for_current_view() -> PrintManager:
        ...
    def is_supported() -> bool:
        ...
    def show_print_u_i_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def add_print_task_requested(event_handler: winrt.windows.foundation.TypedEventHandler[PrintManager, PrintTaskRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_print_task_requested(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class PrintPageInfo(_winrt.winrt_base):
    ...
    page_size: winrt.windows.foundation.Size
    orientation: PrintOrientation
    media_size: PrintMediaSize
    dpi_y: int
    dpi_x: int

class PrintPageRange(_winrt.winrt_base):
    ...
    first_page_number: int
    last_page_number: int

class PrintPageRangeOptions(_winrt.winrt_base):
    ...
    allow_custom_set_of_pages: bool
    allow_current_page: bool
    allow_all_pages: bool

class PrintTask(_winrt.winrt_base):
    ...
    options: PrintTaskOptions
    properties: winrt.windows.applicationmodel.datatransfer.DataPackagePropertySet
    source: IPrintDocumentSource
    is_preview_enabled: bool
    is_printer_target_enabled: bool
    is3_d_manufacturing_target_enabled: bool
    def add_completed(event_handler: winrt.windows.foundation.TypedEventHandler[PrintTask, PrintTaskCompletedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_completed(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_previewing(event_handler: winrt.windows.foundation.TypedEventHandler[PrintTask, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_previewing(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_progressing(event_handler: winrt.windows.foundation.TypedEventHandler[PrintTask, PrintTaskProgressingEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_progressing(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_submitting(event_handler: winrt.windows.foundation.TypedEventHandler[PrintTask, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_submitting(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class PrintTaskCompletedEventArgs(_winrt.winrt_base):
    ...
    completion: PrintTaskCompletion

class PrintTaskOptions(IPrintTaskOptionsCore, IPrintTaskOptionsCoreProperties, IPrintTaskOptionsCoreUIConfiguration, _winrt.winrt_base):
    ...
    bordering: PrintBordering
    page_range_options: PrintPageRangeOptions
    custom_page_ranges: winrt.windows.foundation.collections.IVector[PrintPageRange]
    media_type: PrintMediaType
    hole_punch: PrintHolePunch
    binding: PrintBinding
    duplex: PrintDuplex
    color_mode: PrintColorMode
    collation: PrintCollation
    staple: PrintStaple
    print_quality: PrintQuality
    orientation: PrintOrientation
    media_size: PrintMediaSize
    number_of_copies: int
    max_copies: int
    min_copies: int
    displayed_options: winrt.windows.foundation.collections.IVector[str]
    def get_page_description(job_page_number: int) -> PrintPageDescription:
        ...
    def get_page_print_ticket(print_page_info: PrintPageInfo) -> winrt.windows.storage.streams.IRandomAccessStream:
        ...

class PrintTaskProgressingEventArgs(_winrt.winrt_base):
    ...
    document_page_count: int

class PrintTaskRequest(_winrt.winrt_base):
    ...
    deadline: winrt.windows.foundation.DateTime
    def create_print_task(title: str, handler: PrintTaskSourceRequestedHandler) -> PrintTask:
        ...
    def get_deferral() -> PrintTaskRequestedDeferral:
        ...

class PrintTaskRequestedDeferral(_winrt.winrt_base):
    ...
    def complete() -> None:
        ...

class PrintTaskRequestedEventArgs(_winrt.winrt_base):
    ...
    request: PrintTaskRequest

class PrintTaskSourceRequestedArgs(_winrt.winrt_base):
    ...
    deadline: winrt.windows.foundation.DateTime
    def get_deferral() -> PrintTaskSourceRequestedDeferral:
        ...
    def set_source(source: IPrintDocumentSource) -> None:
        ...

class PrintTaskSourceRequestedDeferral(_winrt.winrt_base):
    ...
    def complete() -> None:
        ...

class StandardPrintTaskOptions(_winrt.winrt_base):
    ...
    copies: str
    binding: str
    collation: str
    color_mode: str
    media_type: str
    duplex: str
    hole_punch: str
    input_bin: str
    media_size: str
    n_up: str
    orientation: str
    print_quality: str
    staple: str
    bordering: str
    custom_page_ranges: str

class IPrintDocumentSource(_winrt.winrt_base):
    ...

class IPrintTaskOptionsCore(_winrt.winrt_base):
    ...
    def get_page_description(job_page_number: int) -> PrintPageDescription:
        ...

class IPrintTaskOptionsCoreProperties(_winrt.winrt_base):
    ...
    binding: PrintBinding
    collation: PrintCollation
    color_mode: PrintColorMode
    duplex: PrintDuplex
    hole_punch: PrintHolePunch
    max_copies: int
    media_size: PrintMediaSize
    media_type: PrintMediaType
    min_copies: int
    number_of_copies: int
    orientation: PrintOrientation
    print_quality: PrintQuality
    staple: PrintStaple

class IPrintTaskOptionsCoreUIConfiguration(_winrt.winrt_base):
    ...
    displayed_options: winrt.windows.foundation.collections.IVector[str]

PrintTaskSourceRequestedHandler = typing.Callable[[PrintTaskSourceRequestedArgs], None]

