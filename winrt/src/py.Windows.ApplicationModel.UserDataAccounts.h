// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.ApplicationModel.Appointments.h")
#include "py.Windows.ApplicationModel.Appointments.h"
#endif

#if __has_include("py.Windows.ApplicationModel.Contacts.h")
#include "py.Windows.ApplicationModel.Contacts.h"
#endif

#if __has_include("py.Windows.ApplicationModel.Email.h")
#include "py.Windows.ApplicationModel.Email.h"
#endif

#if __has_include("py.Windows.ApplicationModel.UserDataTasks.h")
#include "py.Windows.ApplicationModel.UserDataTasks.h"
#endif

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#if __has_include("py.Windows.Storage.Streams.h")
#include "py.Windows.Storage.Streams.h"
#endif

#if __has_include("py.Windows.System.h")
#include "py.Windows.System.h"
#endif

#include <winrt/Windows.ApplicationModel.UserDataAccounts.h>

namespace py::proj::Windows::ApplicationModel::UserDataAccounts
{}

namespace py::impl::Windows::ApplicationModel::UserDataAccounts
{}

namespace py::wrapper::Windows::ApplicationModel::UserDataAccounts
{
    using UserDataAccount = py::winrt_wrapper<winrt::Windows::ApplicationModel::UserDataAccounts::UserDataAccount>;
    using UserDataAccountManager = py::winrt_wrapper<winrt::Windows::ApplicationModel::UserDataAccounts::UserDataAccountManager>;
    using UserDataAccountManagerForUser = py::winrt_wrapper<winrt::Windows::ApplicationModel::UserDataAccounts::UserDataAccountManagerForUser>;
    using UserDataAccountStore = py::winrt_wrapper<winrt::Windows::ApplicationModel::UserDataAccounts::UserDataAccountStore>;
    using UserDataAccountStoreChangedEventArgs = py::winrt_wrapper<winrt::Windows::ApplicationModel::UserDataAccounts::UserDataAccountStoreChangedEventArgs>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::ApplicationModel::UserDataAccounts::UserDataAccount>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::ApplicationModel::UserDataAccounts::UserDataAccountManager>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::ApplicationModel::UserDataAccounts::UserDataAccountManagerForUser>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::ApplicationModel::UserDataAccounts::UserDataAccountStore>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::ApplicationModel::UserDataAccounts::UserDataAccountStoreChangedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}