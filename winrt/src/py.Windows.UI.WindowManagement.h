// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#if __has_include("py.Windows.System.h")
#include "py.Windows.System.h"
#endif

#if __has_include("py.Windows.UI.h")
#include "py.Windows.UI.h"
#endif

#if __has_include("py.Windows.UI.Composition.h")
#include "py.Windows.UI.Composition.h"
#endif

#include <winrt/Windows.UI.WindowManagement.h>

namespace py::proj::Windows::UI::WindowManagement
{}

namespace py::impl::Windows::UI::WindowManagement
{}

namespace py::wrapper::Windows::UI::WindowManagement
{
    using AppWindow = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::AppWindow>;
    using AppWindowChangedEventArgs = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::AppWindowChangedEventArgs>;
    using AppWindowCloseRequestedEventArgs = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::AppWindowCloseRequestedEventArgs>;
    using AppWindowClosedEventArgs = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::AppWindowClosedEventArgs>;
    using AppWindowFrame = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::AppWindowFrame>;
    using AppWindowPlacement = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::AppWindowPlacement>;
    using AppWindowPresentationConfiguration = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::AppWindowPresentationConfiguration>;
    using AppWindowPresenter = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::AppWindowPresenter>;
    using AppWindowTitleBar = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::AppWindowTitleBar>;
    using AppWindowTitleBarOcclusion = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::AppWindowTitleBarOcclusion>;
    using CompactOverlayPresentationConfiguration = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::CompactOverlayPresentationConfiguration>;
    using DefaultPresentationConfiguration = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::DefaultPresentationConfiguration>;
    using DisplayRegion = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::DisplayRegion>;
    using FullScreenPresentationConfiguration = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::FullScreenPresentationConfiguration>;
    using WindowServices = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::WindowServices>;
    using WindowingEnvironment = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::WindowingEnvironment>;
    using WindowingEnvironmentAddedEventArgs = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::WindowingEnvironmentAddedEventArgs>;
    using WindowingEnvironmentChangedEventArgs = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::WindowingEnvironmentChangedEventArgs>;
    using WindowingEnvironmentRemovedEventArgs = py::winrt_wrapper<winrt::Windows::UI::WindowManagement::WindowingEnvironmentRemovedEventArgs>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::AppWindow>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::AppWindowChangedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::AppWindowCloseRequestedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::AppWindowClosedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::AppWindowFrame>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::AppWindowPlacement>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::AppWindowPresentationConfiguration>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::AppWindowPresenter>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::AppWindowTitleBar>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::AppWindowTitleBarOcclusion>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::CompactOverlayPresentationConfiguration>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::DefaultPresentationConfiguration>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::DisplayRegion>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::FullScreenPresentationConfiguration>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::WindowServices>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::WindowingEnvironment>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::WindowingEnvironmentAddedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::WindowingEnvironmentChangedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::WindowManagement::WindowingEnvironmentRemovedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}
