# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Media.SpeechRecognition")

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
    import winrt.windows.storage
except Exception:
    pass

class SpeechContinuousRecognitionMode(enum.IntEnum):
    DEFAULT = 0
    PAUSE_ON_RECOGNITION = 1

class SpeechRecognitionAudioProblem(enum.IntEnum):
    NONE = 0
    TOO_NOISY = 1
    NO_SIGNAL = 2
    TOO_LOUD = 3
    TOO_QUIET = 4
    TOO_FAST = 5
    TOO_SLOW = 6

class SpeechRecognitionConfidence(enum.IntEnum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2
    REJECTED = 3

class SpeechRecognitionConstraintProbability(enum.IntEnum):
    DEFAULT = 0
    MIN = 1
    MAX = 2

class SpeechRecognitionConstraintType(enum.IntEnum):
    TOPIC = 0
    LIST = 1
    GRAMMAR = 2
    VOICE_COMMAND_DEFINITION = 3

class SpeechRecognitionResultStatus(enum.IntEnum):
    SUCCESS = 0
    TOPIC_LANGUAGE_NOT_SUPPORTED = 1
    GRAMMAR_LANGUAGE_MISMATCH = 2
    GRAMMAR_COMPILATION_FAILURE = 3
    AUDIO_QUALITY_FAILURE = 4
    USER_CANCELED = 5
    UNKNOWN = 6
    TIMEOUT_EXCEEDED = 7
    PAUSE_LIMIT_EXCEEDED = 8
    NETWORK_FAILURE = 9
    MICROPHONE_UNAVAILABLE = 10

class SpeechRecognitionScenario(enum.IntEnum):
    WEB_SEARCH = 0
    DICTATION = 1
    FORM_FILLING = 2

class SpeechRecognizerState(enum.IntEnum):
    IDLE = 0
    CAPTURING = 1
    PROCESSING = 2
    SOUND_STARTED = 3
    SOUND_ENDED = 4
    SPEECH_DETECTED = 5
    PAUSED = 6

SpeechContinuousRecognitionCompletedEventArgs = _ns_module.SpeechContinuousRecognitionCompletedEventArgs
SpeechContinuousRecognitionResultGeneratedEventArgs = _ns_module.SpeechContinuousRecognitionResultGeneratedEventArgs
SpeechContinuousRecognitionSession = _ns_module.SpeechContinuousRecognitionSession
SpeechRecognitionCompilationResult = _ns_module.SpeechRecognitionCompilationResult
SpeechRecognitionGrammarFileConstraint = _ns_module.SpeechRecognitionGrammarFileConstraint
SpeechRecognitionHypothesis = _ns_module.SpeechRecognitionHypothesis
SpeechRecognitionHypothesisGeneratedEventArgs = _ns_module.SpeechRecognitionHypothesisGeneratedEventArgs
SpeechRecognitionListConstraint = _ns_module.SpeechRecognitionListConstraint
SpeechRecognitionQualityDegradingEventArgs = _ns_module.SpeechRecognitionQualityDegradingEventArgs
SpeechRecognitionResult = _ns_module.SpeechRecognitionResult
SpeechRecognitionSemanticInterpretation = _ns_module.SpeechRecognitionSemanticInterpretation
SpeechRecognitionTopicConstraint = _ns_module.SpeechRecognitionTopicConstraint
SpeechRecognitionVoiceCommandDefinitionConstraint = _ns_module.SpeechRecognitionVoiceCommandDefinitionConstraint
SpeechRecognizer = _ns_module.SpeechRecognizer
SpeechRecognizerStateChangedEventArgs = _ns_module.SpeechRecognizerStateChangedEventArgs
SpeechRecognizerTimeouts = _ns_module.SpeechRecognizerTimeouts
SpeechRecognizerUIOptions = _ns_module.SpeechRecognizerUIOptions
ISpeechRecognitionConstraint = _ns_module.ISpeechRecognitionConstraint