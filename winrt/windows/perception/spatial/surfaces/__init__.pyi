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
    import winrt.windows.foundation.numerics
except Exception:
    pass

try:
    import winrt.windows.graphics.directx
except Exception:
    pass

try:
    import winrt.windows.perception.spatial
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class SpatialSurfaceInfo(_winrt.winrt_base):
    ...
    id: uuid.UUID
    update_time: winrt.windows.foundation.DateTime
    def try_compute_latest_mesh_async(max_triangles_per_cubic_meter: float) -> winrt.windows.foundation.IAsyncOperation[SpatialSurfaceMesh]:
        ...
    def try_compute_latest_mesh_async(max_triangles_per_cubic_meter: float, options: SpatialSurfaceMeshOptions) -> winrt.windows.foundation.IAsyncOperation[SpatialSurfaceMesh]:
        ...
    def try_get_bounds(coordinate_system: winrt.windows.perception.spatial.SpatialCoordinateSystem) -> typing.Optional[winrt.windows.perception.spatial.SpatialBoundingOrientedBox]:
        ...

class SpatialSurfaceMesh(_winrt.winrt_base):
    ...
    coordinate_system: winrt.windows.perception.spatial.SpatialCoordinateSystem
    surface_info: SpatialSurfaceInfo
    triangle_indices: SpatialSurfaceMeshBuffer
    vertex_normals: SpatialSurfaceMeshBuffer
    vertex_position_scale: winrt.windows.foundation.numerics.Vector3
    vertex_positions: SpatialSurfaceMeshBuffer

class SpatialSurfaceMeshBuffer(_winrt.winrt_base):
    ...
    data: winrt.windows.storage.streams.IBuffer
    element_count: int
    format: winrt.windows.graphics.directx.DirectXPixelFormat
    stride: int

class SpatialSurfaceMeshOptions(_winrt.winrt_base):
    ...
    vertex_position_format: winrt.windows.graphics.directx.DirectXPixelFormat
    vertex_normal_format: winrt.windows.graphics.directx.DirectXPixelFormat
    triangle_index_format: winrt.windows.graphics.directx.DirectXPixelFormat
    include_vertex_normals: bool
    supported_triangle_index_formats: winrt.windows.foundation.collections.IVectorView[winrt.windows.graphics.directx.DirectXPixelFormat]
    supported_vertex_normal_formats: winrt.windows.foundation.collections.IVectorView[winrt.windows.graphics.directx.DirectXPixelFormat]
    supported_vertex_position_formats: winrt.windows.foundation.collections.IVectorView[winrt.windows.graphics.directx.DirectXPixelFormat]

class SpatialSurfaceObserver(_winrt.winrt_base):
    ...
    def get_observed_surfaces() -> winrt.windows.foundation.collections.IMapView[uuid.UUID, SpatialSurfaceInfo]:
        ...
    def is_supported() -> bool:
        ...
    def request_access_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.perception.spatial.SpatialPerceptionAccessStatus]:
        ...
    def set_bounding_volume(bounds: winrt.windows.perception.spatial.SpatialBoundingVolume) -> None:
        ...
    def set_bounding_volumes(bounds: typing.Iterable[winrt.windows.perception.spatial.SpatialBoundingVolume]) -> None:
        ...
    def add_observed_surfaces_changed(handler: winrt.windows.foundation.TypedEventHandler[SpatialSurfaceObserver, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_observed_surfaces_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

