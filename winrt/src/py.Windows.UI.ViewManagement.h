// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Devices.Enumeration.h")
#include "py.Windows.Devices.Enumeration.h"
#endif

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#if __has_include("py.Windows.UI.h")
#include "py.Windows.UI.h"
#endif

#if __has_include("py.Windows.UI.Core.h")
#include "py.Windows.UI.Core.h"
#endif

#if __has_include("py.Windows.UI.Popups.h")
#include "py.Windows.UI.Popups.h"
#endif

#if __has_include("py.Windows.UI.WindowManagement.h")
#include "py.Windows.UI.WindowManagement.h"
#endif

#include <winrt/Windows.UI.ViewManagement.h>

namespace py::proj::Windows::UI::ViewManagement
{}

namespace py::impl::Windows::UI::ViewManagement
{}

namespace py::wrapper::Windows::UI::ViewManagement
{
    using AccessibilitySettings = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::AccessibilitySettings>;
    using ActivationViewSwitcher = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::ActivationViewSwitcher>;
    using ApplicationView = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::ApplicationView>;
    using ApplicationViewConsolidatedEventArgs = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::ApplicationViewConsolidatedEventArgs>;
    using ApplicationViewScaling = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::ApplicationViewScaling>;
    using ApplicationViewSwitcher = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::ApplicationViewSwitcher>;
    using ApplicationViewTitleBar = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::ApplicationViewTitleBar>;
    using ApplicationViewTransferContext = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::ApplicationViewTransferContext>;
    using InputPane = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::InputPane>;
    using InputPaneVisibilityEventArgs = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::InputPaneVisibilityEventArgs>;
    using ProjectionManager = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::ProjectionManager>;
    using UISettings = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::UISettings>;
    using UISettingsAnimationsEnabledChangedEventArgs = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::UISettingsAnimationsEnabledChangedEventArgs>;
    using UISettingsAutoHideScrollBarsChangedEventArgs = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::UISettingsAutoHideScrollBarsChangedEventArgs>;
    using UISettingsMessageDurationChangedEventArgs = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::UISettingsMessageDurationChangedEventArgs>;
    using UIViewSettings = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::UIViewSettings>;
    using ViewModePreferences = py::winrt_wrapper<winrt::Windows::UI::ViewManagement::ViewModePreferences>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::AccessibilitySettings>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::ActivationViewSwitcher>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::ApplicationView>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::ApplicationViewConsolidatedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::ApplicationViewScaling>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::ApplicationViewSwitcher>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::ApplicationViewTitleBar>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::ApplicationViewTransferContext>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::InputPane>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::InputPaneVisibilityEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::ProjectionManager>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::UISettings>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::UISettingsAnimationsEnabledChangedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::UISettingsAutoHideScrollBarsChangedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::UISettingsMessageDurationChangedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::UIViewSettings>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::UI::ViewManagement::ViewModePreferences>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}
