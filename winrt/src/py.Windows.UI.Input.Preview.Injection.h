// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#if __has_include("py.Windows.Gaming.Input.h")
#include "py.Windows.Gaming.Input.h"
#endif

#include <winrt/Windows.UI.Input.Preview.Injection.h>

namespace py::proj::Windows::UI::Input::Preview::Injection
{}

namespace py::impl::Windows::UI::Input::Preview::Injection
{}

namespace py::wrapper::Windows::UI::Input::Preview::Injection
{
    using InjectedInputGamepadInfo = py::winrt_wrapper<winrt::Windows::UI::Input::Preview::Injection::InjectedInputGamepadInfo>;
    using InjectedInputKeyboardInfo = py::winrt_wrapper<winrt::Windows::UI::Input::Preview::Injection::InjectedInputKeyboardInfo>;
    using InjectedInputMouseInfo = py::winrt_wrapper<winrt::Windows::UI::Input::Preview::Injection::InjectedInputMouseInfo>;
    using InjectedInputPenInfo = py::winrt_wrapper<winrt::Windows::UI::Input::Preview::Injection::InjectedInputPenInfo>;
    using InjectedInputTouchInfo = py::winrt_wrapper<winrt::Windows::UI::Input::Preview::Injection::InjectedInputTouchInfo>;
    using InputInjector = py::winrt_wrapper<winrt::Windows::UI::Input::Preview::Injection::InputInjector>;
    using InjectedInputPoint = py::winrt_struct_wrapper<winrt::Windows::UI::Input::Preview::Injection::InjectedInputPoint>;
    using InjectedInputPointerInfo = py::winrt_struct_wrapper<winrt::Windows::UI::Input::Preview::Injection::InjectedInputPointerInfo>;
    using InjectedInputRectangle = py::winrt_struct_wrapper<winrt::Windows::UI::Input::Preview::Injection::InjectedInputRectangle>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::UI::Input::Preview::Injection::InjectedInputGamepadInfo>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::Input::Preview::Injection::InjectedInputKeyboardInfo>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::Input::Preview::Injection::InjectedInputMouseInfo>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::Input::Preview::Injection::InjectedInputPenInfo>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::Input::Preview::Injection::InjectedInputTouchInfo>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::Input::Preview::Injection::InputInjector>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::Input::Preview::Injection::InjectedInputPoint>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::Input::Preview::Injection::InjectedInputPointerInfo>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::Input::Preview::Injection::InjectedInputRectangle>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct converter<winrt::Windows::UI::Input::Preview::Injection::InjectedInputPoint>
    {
        static PyObject* convert(winrt::Windows::UI::Input::Preview::Injection::InjectedInputPoint instance) noexcept;
        static winrt::Windows::UI::Input::Preview::Injection::InjectedInputPoint convert_to(PyObject* obj);
    };

    template<>
    struct converter<winrt::Windows::UI::Input::Preview::Injection::InjectedInputPointerInfo>
    {
        static PyObject* convert(winrt::Windows::UI::Input::Preview::Injection::InjectedInputPointerInfo instance) noexcept;
        static winrt::Windows::UI::Input::Preview::Injection::InjectedInputPointerInfo convert_to(PyObject* obj);
    };

    template<>
    struct converter<winrt::Windows::UI::Input::Preview::Injection::InjectedInputRectangle>
    {
        static PyObject* convert(winrt::Windows::UI::Input::Preview::Injection::InjectedInputRectangle instance) noexcept;
        static winrt::Windows::UI::Input::Preview::Injection::InjectedInputRectangle convert_to(PyObject* obj);
    };

}