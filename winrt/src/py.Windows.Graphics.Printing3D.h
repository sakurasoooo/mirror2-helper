// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#if __has_include("py.Windows.Foundation.Numerics.h")
#include "py.Windows.Foundation.Numerics.h"
#endif

#if __has_include("py.Windows.Storage.Streams.h")
#include "py.Windows.Storage.Streams.h"
#endif

#if __has_include("py.Windows.UI.h")
#include "py.Windows.UI.h"
#endif

#include <winrt/Windows.Graphics.Printing3D.h>

namespace py::proj::Windows::Graphics::Printing3D
{}

namespace py::impl::Windows::Graphics::Printing3D
{
    struct Print3DTaskSourceRequestedHandler
    {
        static winrt::Windows::Graphics::Printing3D::Print3DTaskSourceRequestedHandler get(PyObject* callable)
        {
            py::delegate_callable _delegate{ callable };

            return [delegate = std::move(_delegate)](auto param0)
            {
                winrt::handle_type<py::gil_state_traits> gil_state{ PyGILState_Ensure() };

                py::pyobj_handle py_param0{ py::convert(param0) };

                py::pyobj_handle args{ PyTuple_Pack(1, py_param0.get()) };
                py::pyobj_handle return_value{ PyObject_CallObject(delegate.callable(), args.get()) };

                if (!return_value)
                {
                    PyErr_WriteUnraisable(delegate.callable());
                    throw winrt::hresult_invalid_argument();
                }
            };
        };
    };
}

namespace py::wrapper::Windows::Graphics::Printing3D
{
    using Print3DManager = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Print3DManager>;
    using Print3DTask = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Print3DTask>;
    using Print3DTaskCompletedEventArgs = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Print3DTaskCompletedEventArgs>;
    using Print3DTaskRequest = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Print3DTaskRequest>;
    using Print3DTaskRequestedEventArgs = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Print3DTaskRequestedEventArgs>;
    using Print3DTaskSourceChangedEventArgs = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Print3DTaskSourceChangedEventArgs>;
    using Print3DTaskSourceRequestedArgs = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Print3DTaskSourceRequestedArgs>;
    using Printing3D3MFPackage = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3D3MFPackage>;
    using Printing3DBaseMaterial = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DBaseMaterial>;
    using Printing3DBaseMaterialGroup = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DBaseMaterialGroup>;
    using Printing3DColorMaterial = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DColorMaterial>;
    using Printing3DColorMaterialGroup = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DColorMaterialGroup>;
    using Printing3DComponent = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DComponent>;
    using Printing3DComponentWithMatrix = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DComponentWithMatrix>;
    using Printing3DCompositeMaterial = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DCompositeMaterial>;
    using Printing3DCompositeMaterialGroup = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DCompositeMaterialGroup>;
    using Printing3DFaceReductionOptions = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DFaceReductionOptions>;
    using Printing3DMaterial = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DMaterial>;
    using Printing3DMesh = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DMesh>;
    using Printing3DMeshVerificationResult = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DMeshVerificationResult>;
    using Printing3DModel = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DModel>;
    using Printing3DModelTexture = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DModelTexture>;
    using Printing3DMultiplePropertyMaterial = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DMultiplePropertyMaterial>;
    using Printing3DMultiplePropertyMaterialGroup = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DMultiplePropertyMaterialGroup>;
    using Printing3DTexture2CoordMaterial = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DTexture2CoordMaterial>;
    using Printing3DTexture2CoordMaterialGroup = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DTexture2CoordMaterialGroup>;
    using Printing3DTextureResource = py::winrt_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DTextureResource>;
    using Printing3DBufferDescription = py::winrt_struct_wrapper<winrt::Windows::Graphics::Printing3D::Printing3DBufferDescription>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Print3DManager>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Print3DTask>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Print3DTaskCompletedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Print3DTaskRequest>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Print3DTaskRequestedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Print3DTaskSourceChangedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Print3DTaskSourceRequestedArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3D3MFPackage>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DBaseMaterial>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DBaseMaterialGroup>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DColorMaterial>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DColorMaterialGroup>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DComponent>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DComponentWithMatrix>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DCompositeMaterial>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DCompositeMaterialGroup>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DFaceReductionOptions>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DMaterial>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DMesh>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DMeshVerificationResult>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DModel>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DModelTexture>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DMultiplePropertyMaterial>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DMultiplePropertyMaterialGroup>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DTexture2CoordMaterial>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DTexture2CoordMaterialGroup>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DTextureResource>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Graphics::Printing3D::Printing3DBufferDescription>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template <>
    struct delegate_python_type<winrt::Windows::Graphics::Printing3D::Print3DTaskSourceRequestedHandler>
    {
        using type = py::impl::Windows::Graphics::Printing3D::Print3DTaskSourceRequestedHandler;
    };

    template<>
    struct converter<winrt::Windows::Graphics::Printing3D::Printing3DBufferDescription>
    {
        static PyObject* convert(winrt::Windows::Graphics::Printing3D::Printing3DBufferDescription instance) noexcept;
        static winrt::Windows::Graphics::Printing3D::Printing3DBufferDescription convert_to(PyObject* obj);
    };

}
