// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.UI.Input.h")
#include "py.Windows.UI.Input.h"
#endif

#if __has_include("py.Windows.UI.WindowManagement.h")
#include "py.Windows.UI.WindowManagement.h"
#endif

#include <winrt/Windows.UI.Input.Preview.h>

namespace py::proj::Windows::UI::Input::Preview
{}

namespace py::impl::Windows::UI::Input::Preview
{}

namespace py::wrapper::Windows::UI::Input::Preview
{
    using InputActivationListenerPreview = py::winrt_wrapper<winrt::Windows::UI::Input::Preview::InputActivationListenerPreview>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::UI::Input::Preview::InputActivationListenerPreview>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}