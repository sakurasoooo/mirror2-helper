# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.AI.MachineLearning")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.graphics
except Exception:
    pass

try:
    import winrt.windows.graphics.directx.direct3d11
except Exception:
    pass

try:
    import winrt.windows.graphics.imaging
except Exception:
    pass

try:
    import winrt.windows.media
except Exception:
    pass

try:
    import winrt.windows.storage
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class LearningModelDeviceKind(enum.IntEnum):
    DEFAULT = 0
    CPU = 1
    DIRECT_X = 2
    DIRECT_X_HIGH_PERFORMANCE = 3
    DIRECT_X_MIN_POWER = 4

class LearningModelFeatureKind(enum.IntEnum):
    TENSOR = 0
    SEQUENCE = 1
    MAP = 2
    IMAGE = 3

class LearningModelPixelRange(enum.IntEnum):
    ZERO_TO255 = 0
    ZERO_TO_ONE = 1
    MINUS_ONE_TO_ONE = 2

class TensorKind(enum.IntEnum):
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOLEAN = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15

ImageFeatureDescriptor = _ns_module.ImageFeatureDescriptor
ImageFeatureValue = _ns_module.ImageFeatureValue
LearningModel = _ns_module.LearningModel
LearningModelBinding = _ns_module.LearningModelBinding
LearningModelDevice = _ns_module.LearningModelDevice
LearningModelEvaluationResult = _ns_module.LearningModelEvaluationResult
LearningModelSession = _ns_module.LearningModelSession
LearningModelSessionOptions = _ns_module.LearningModelSessionOptions
MapFeatureDescriptor = _ns_module.MapFeatureDescriptor
SequenceFeatureDescriptor = _ns_module.SequenceFeatureDescriptor
TensorBoolean = _ns_module.TensorBoolean
TensorDouble = _ns_module.TensorDouble
TensorFeatureDescriptor = _ns_module.TensorFeatureDescriptor
TensorFloat = _ns_module.TensorFloat
TensorFloat16Bit = _ns_module.TensorFloat16Bit
TensorInt16Bit = _ns_module.TensorInt16Bit
TensorInt32Bit = _ns_module.TensorInt32Bit
TensorInt64Bit = _ns_module.TensorInt64Bit
TensorInt8Bit = _ns_module.TensorInt8Bit
TensorString = _ns_module.TensorString
TensorUInt16Bit = _ns_module.TensorUInt16Bit
TensorUInt32Bit = _ns_module.TensorUInt32Bit
TensorUInt64Bit = _ns_module.TensorUInt64Bit
TensorUInt8Bit = _ns_module.TensorUInt8Bit
ILearningModelFeatureDescriptor = _ns_module.ILearningModelFeatureDescriptor
ILearningModelFeatureValue = _ns_module.ILearningModelFeatureValue
ILearningModelOperatorProvider = _ns_module.ILearningModelOperatorProvider
ITensor = _ns_module.ITensor