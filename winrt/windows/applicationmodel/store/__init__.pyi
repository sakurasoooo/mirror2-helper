# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
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
    import winrt.windows.storage
except Exception:
    pass

class FulfillmentResult(enum.IntEnum):
    SUCCEEDED = 0
    NOTHING_TO_FULFILL = 1
    PURCHASE_PENDING = 2
    PURCHASE_REVERTED = 3
    SERVER_ERROR = 4

class ProductPurchaseStatus(enum.IntEnum):
    SUCCEEDED = 0
    ALREADY_PURCHASED = 1
    NOT_FULFILLED = 2
    NOT_PURCHASED = 3

class ProductType(enum.IntEnum):
    UNKNOWN = 0
    DURABLE = 1
    CONSUMABLE = 2

class CurrentApp(_winrt.winrt_base):
    ...
    app_id: uuid.UUID
    license_information: LicenseInformation
    link_uri: winrt.windows.foundation.Uri
    def get_app_purchase_campaign_id_async() -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def get_app_receipt_async() -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def get_customer_collections_id_async(service_ticket: str, publisher_user_id: str) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def get_customer_purchase_id_async(service_ticket: str, publisher_user_id: str) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def get_product_receipt_async(product_id: str) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def get_unfulfilled_consumables_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[UnfulfilledConsumable]]:
        ...
    def load_listing_information_async() -> winrt.windows.foundation.IAsyncOperation[ListingInformation]:
        ...
    def load_listing_information_by_keywords_async(keywords: typing.Iterable[str]) -> winrt.windows.foundation.IAsyncOperation[ListingInformation]:
        ...
    def load_listing_information_by_product_ids_async(product_ids: typing.Iterable[str]) -> winrt.windows.foundation.IAsyncOperation[ListingInformation]:
        ...
    def report_consumable_fulfillment_async(product_id: str, transaction_id: uuid.UUID) -> winrt.windows.foundation.IAsyncOperation[FulfillmentResult]:
        ...
    def report_product_fulfillment(product_id: str) -> None:
        ...
    def request_app_purchase_async(include_receipt: bool) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def request_product_purchase_async(product_id: str) -> winrt.windows.foundation.IAsyncOperation[PurchaseResults]:
        ...
    def request_product_purchase_async(product_id: str, include_receipt: bool) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def request_product_purchase_async(product_id: str, offer_id: str, display_properties: ProductPurchaseDisplayProperties) -> winrt.windows.foundation.IAsyncOperation[PurchaseResults]:
        ...

class CurrentAppSimulator(_winrt.winrt_base):
    ...
    app_id: uuid.UUID
    license_information: LicenseInformation
    link_uri: winrt.windows.foundation.Uri
    def get_app_purchase_campaign_id_async() -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def get_app_receipt_async() -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def get_product_receipt_async(product_id: str) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def get_unfulfilled_consumables_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[UnfulfilledConsumable]]:
        ...
    def load_listing_information_async() -> winrt.windows.foundation.IAsyncOperation[ListingInformation]:
        ...
    def load_listing_information_by_keywords_async(keywords: typing.Iterable[str]) -> winrt.windows.foundation.IAsyncOperation[ListingInformation]:
        ...
    def load_listing_information_by_product_ids_async(product_ids: typing.Iterable[str]) -> winrt.windows.foundation.IAsyncOperation[ListingInformation]:
        ...
    def reload_simulator_async(simulator_settings_file: winrt.windows.storage.StorageFile) -> winrt.windows.foundation.IAsyncAction:
        ...
    def report_consumable_fulfillment_async(product_id: str, transaction_id: uuid.UUID) -> winrt.windows.foundation.IAsyncOperation[FulfillmentResult]:
        ...
    def request_app_purchase_async(include_receipt: bool) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def request_product_purchase_async(product_id: str) -> winrt.windows.foundation.IAsyncOperation[PurchaseResults]:
        ...
    def request_product_purchase_async(product_id: str, include_receipt: bool) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def request_product_purchase_async(product_id: str, offer_id: str, display_properties: ProductPurchaseDisplayProperties) -> winrt.windows.foundation.IAsyncOperation[PurchaseResults]:
        ...

class LicenseInformation(_winrt.winrt_base):
    ...
    expiration_date: winrt.windows.foundation.DateTime
    is_active: bool
    is_trial: bool
    product_licenses: winrt.windows.foundation.collections.IMapView[str, ProductLicense]
    def add_license_changed(handler: LicenseChangedEventHandler) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_license_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ListingInformation(_winrt.winrt_base):
    ...
    age_rating: int
    current_market: str
    description: str
    formatted_price: str
    name: str
    product_listings: winrt.windows.foundation.collections.IMapView[str, ProductListing]
    currency_code: str
    formatted_base_price: str
    is_on_sale: bool
    sale_end_date: winrt.windows.foundation.DateTime

class ProductLicense(_winrt.winrt_base):
    ...
    expiration_date: winrt.windows.foundation.DateTime
    is_active: bool
    product_id: str
    is_consumable: bool

class ProductListing(_winrt.winrt_base):
    ...
    formatted_price: str
    name: str
    product_id: str
    formatted_base_price: str
    is_on_sale: bool
    sale_end_date: winrt.windows.foundation.DateTime
    currency_code: str
    description: str
    image_uri: winrt.windows.foundation.Uri
    keywords: winrt.windows.foundation.collections.IIterable[str]
    tag: str
    product_type: ProductType

class ProductPurchaseDisplayProperties(_winrt.winrt_base):
    ...
    name: str
    image: winrt.windows.foundation.Uri
    description: str

class PurchaseResults(_winrt.winrt_base):
    ...
    offer_id: str
    receipt_xml: str
    status: ProductPurchaseStatus
    transaction_id: uuid.UUID

class UnfulfilledConsumable(_winrt.winrt_base):
    ...
    offer_id: str
    product_id: str
    transaction_id: uuid.UUID

LicenseChangedEventHandler = typing.Callable[[], None]
