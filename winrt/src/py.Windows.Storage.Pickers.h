// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#if __has_include("py.Windows.Storage.h")
#include "py.Windows.Storage.h"
#endif

#if __has_include("py.Windows.System.h")
#include "py.Windows.System.h"
#endif

#include <winrt/Windows.Storage.Pickers.h>

namespace py::proj::Windows::Storage::Pickers
{}

namespace py::impl::Windows::Storage::Pickers
{}

namespace py::wrapper::Windows::Storage::Pickers
{
    using FileExtensionVector = py::winrt_wrapper<winrt::Windows::Storage::Pickers::FileExtensionVector>;
    using FileOpenPicker = py::winrt_wrapper<winrt::Windows::Storage::Pickers::FileOpenPicker>;
    using FilePickerFileTypesOrderedMap = py::winrt_wrapper<winrt::Windows::Storage::Pickers::FilePickerFileTypesOrderedMap>;
    using FilePickerSelectedFilesArray = py::winrt_wrapper<winrt::Windows::Storage::Pickers::FilePickerSelectedFilesArray>;
    using FileSavePicker = py::winrt_wrapper<winrt::Windows::Storage::Pickers::FileSavePicker>;
    using FolderPicker = py::winrt_wrapper<winrt::Windows::Storage::Pickers::FolderPicker>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::Storage::Pickers::FileExtensionVector>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Storage::Pickers::FileOpenPicker>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Storage::Pickers::FilePickerFileTypesOrderedMap>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Storage::Pickers::FilePickerSelectedFilesArray>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Storage::Pickers::FileSavePicker>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Storage::Pickers::FolderPicker>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}