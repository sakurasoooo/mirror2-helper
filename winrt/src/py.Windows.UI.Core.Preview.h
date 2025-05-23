// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.UI.WindowManagement.h")
#include "py.Windows.UI.WindowManagement.h"
#endif

#include <winrt/Windows.UI.Core.Preview.h>

namespace py::proj::Windows::UI::Core::Preview
{}

namespace py::impl::Windows::UI::Core::Preview
{}

namespace py::wrapper::Windows::UI::Core::Preview
{
    using CoreAppWindowPreview = py::winrt_wrapper<winrt::Windows::UI::Core::Preview::CoreAppWindowPreview>;
    using SystemNavigationCloseRequestedPreviewEventArgs = py::winrt_wrapper<winrt::Windows::UI::Core::Preview::SystemNavigationCloseRequestedPreviewEventArgs>;
    using SystemNavigationManagerPreview = py::winrt_wrapper<winrt::Windows::UI::Core::Preview::SystemNavigationManagerPreview>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::UI::Core::Preview::CoreAppWindowPreview>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::Core::Preview::SystemNavigationCloseRequestedPreviewEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::Core::Preview::SystemNavigationManagerPreview>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}
