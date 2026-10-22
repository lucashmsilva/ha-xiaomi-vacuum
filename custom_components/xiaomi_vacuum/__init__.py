"""Xiaomi Vacuum integration for Home Assistant."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from homeassistant.const import Platform
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.loader import async_get_loaded_integration

from .api import XiaomiVacuumApiClient, XiaomiVacuumApiClientCommunicationError
from .cached_device_info import CachedDeviceInfo
from .cloud import XiaomiCloud, XiaomiCloudConnectionError, XiaomiCloudError
from .const import (
    CONF_CLOUD_COUNTRY,
    CONF_CLOUD_SERVICE_TOKEN,
    CONF_CLOUD_SSECURITY,
    CONF_CLOUD_USER_ID,
    CONF_DEVICE_INFO,
    CONF_HOST,
    CONF_TOKEN,
    LOGGER,
)
from .coordinator import XiaomiVacuumDataUpdateCoordinator
from .data import XiaomiVacuumData
from .map_coordinator import XiaomiVacuumMapCoordinator
from .repairs import (
    async_clear_cannot_connect,
    async_clear_unsupported_model,
    async_raise_cannot_connect,
    async_raise_unsupported_model,
)
from .spec import DEFAULT_MODEL, SUPPORTED_MODELS, get_spec

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .data import DeviceInfoLike, JsonObject, XiaomiVacuumConfigEntry

PLATFORMS: list[Platform] = [
    Platform.VACUUM,
    Platform.SELECT,
    Platform.BUTTON,
    Platform.IMAGE,
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
]


async def _setup_cloud(
    hass: HomeAssistant,
    entry: XiaomiVacuumConfigEntry,
    coordinator: XiaomiVacuumDataUpdateCoordinator,
) -> XiaomiVacuumMapCoordinator:
    """
    Resolve the cloud session and wire it into the map/fault coordinators.

    Always returns a map coordinator so the cached map and local entities keep
    working without Xiaomi cloud. When the cloud is unreachable a background
    retry restores map updates once it comes back; an invalid session starts
    reauth while the cached map stays available.
    """
    map_coordinator = XiaomiVacuumMapCoordinator(
        hass, cloud=None, state_coordinator=coordinator
    )
    cloud_country = entry.data.get(CONF_CLOUD_COUNTRY)
    ssecurity = entry.data.get(CONF_CLOUD_SSECURITY)
    service_token = entry.data.get(CONF_CLOUD_SERVICE_TOKEN)
    cloud_user_id = entry.data.get(CONF_CLOUD_USER_ID)
    if not (cloud_country and ssecurity and service_token and cloud_user_id):
        return map_coordinator
    cloud = XiaomiCloud.from_session(
        hass,
        country=cloud_country,
        ssecurity=ssecurity,
        service_token=service_token,
        user_id=cloud_user_id,
    )
    try:
        await cloud.async_resolve_device(entry.data[CONF_TOKEN])
    except XiaomiCloudConnectionError as exception:
        LOGGER.warning(
            "Xiaomi cloud unreachable at setup; local control and the cached "
            "map remain available: %s",
            exception,
        )
        # Retry in the background so map updates resume without a HA restart.
        entry.async_create_background_task(
            hass,
            _retry_cloud_connection(hass, entry, coordinator, map_coordinator, cloud),
            f"{DOMAIN}_cloud_retry",
        )
        return map_coordinator
    except XiaomiCloudError as exception:
        LOGGER.warning(
            "Cloud session invalid; starting reauth to refresh: %s", exception
        )
        # Surface a repair/reauth prompt instead of dropping the map; the
        # cached PNG and all local entities stay available meanwhile.
        entry.async_start_reauth(hass)
        return map_coordinator
    coordinator.cloud = cloud
    map_coordinator.set_cloud(cloud)
    return map_coordinator


async def _retry_cloud_connection(
    hass: HomeAssistant,
    entry: XiaomiVacuumConfigEntry,
    coordinator: XiaomiVacuumDataUpdateCoordinator,
    map_coordinator: XiaomiVacuumMapCoordinator,
    cloud: XiaomiCloud,
) -> None:
    """Retry the cloud handshake until it succeeds, backing off up to 15 min."""
    delay = 60
    while True:
        await asyncio.sleep(delay)
        try:
            await cloud.async_resolve_device(entry.data[CONF_TOKEN])
        except XiaomiCloudConnectionError as exception:
            LOGGER.debug(
                "Xiaomi cloud still unreachable (retrying in %ss): %s",
                delay,
                exception,
            )
            delay = min(delay * 2, 900)
            continue
        except XiaomiCloudError as exception:
            LOGGER.warning(
                "Cloud session invalid while retrying; starting reauth: %s", exception
            )
            entry.async_start_reauth(hass)
            return
        coordinator.cloud = cloud
        map_coordinator.set_cloud(cloud)
        await map_coordinator.async_refresh()
        LOGGER.info("Xiaomi cloud reachable again; map updates resumed")
        return


async def async_setup_entry(
    hass: HomeAssistant,
    entry: XiaomiVacuumConfigEntry,
) -> bool:
    """Set up Xiaomi Vacuum from a config entry."""
    coordinator = XiaomiVacuumDataUpdateCoordinator(hass=hass, config_entry=entry)
    # Model is unknown until the local handshake succeeds; provisionally pick
    # the cached/default model so the client can be built. Once the handshake
    # resolves a different model we rebuild the client with the right spec.
    stored_info = entry.data.get(CONF_DEVICE_INFO)
    cached_model = stored_info.get("model") if isinstance(stored_info, dict) else None
    provisional_spec = get_spec(cached_model or DEFAULT_MODEL)
    client = XiaomiVacuumApiClient(
        host=entry.data[CONF_HOST],
        token=entry.data[CONF_TOKEN],
        spec=provisional_spec,
    )
    offline = False
    info: DeviceInfoLike
    try:
        info = await client.async_get_info()
    except XiaomiVacuumApiClientCommunicationError as exception:
        if not stored_info:
            # First setup ever — nothing cached to build entities from.
            raise ConfigEntryNotReady(exception) from exception
        LOGGER.warning(
            "Vacuum unreachable at setup; continuing with cached device info: %s",
            exception,
        )
        info = CachedDeviceInfo.from_stored(cast("JsonObject", stored_info))
        offline = True
    else:
        stored_info_obj = CachedDeviceInfo.to_stored(info)
        if entry.data.get(CONF_DEVICE_INFO) != stored_info_obj:
            hass.config_entries.async_update_entry(
                entry, data={**entry.data, CONF_DEVICE_INFO: stored_info_obj}
            )
    LOGGER.debug(
        "Device info: model=%s firmware=%s hardware=%s",
        getattr(info, "model", None),
        getattr(info, "firmware_version", None),
        getattr(info, "hardware_version", None),
    )
    # Resolve the real spec from the (now-known) model and rebuild the client if
    # the handshake revealed a different model than the cached/default one.
    model = getattr(info, "model", None)
    spec = get_spec(model)
    if model and model not in SUPPORTED_MODELS:
        async_raise_unsupported_model(hass, entry, model)
    else:
        async_clear_unsupported_model(hass, entry)
    if spec is not provisional_spec:
        client = XiaomiVacuumApiClient(
            host=entry.data[CONF_HOST],
            token=entry.data[CONF_TOKEN],
            spec=spec,
        )
    map_coordinator = await _setup_cloud(hass, entry, coordinator)

    entry.runtime_data = XiaomiVacuumData(
        client=client,
        integration=async_get_loaded_integration(hass, entry.domain),
        coordinator=coordinator,
        info=info,
        spec=spec,
        map_coordinator=map_coordinator,
    )

    if offline:
        async_raise_cannot_connect(hass, entry)
        # Tolerant refresh: entities are created and marked unavailable by the
        # coordinator instead of holding the whole entry in SETUP_RETRY.
        await coordinator.async_refresh()
    else:
        await coordinator.async_config_entry_first_refresh()
    # The map is best-effort and must never block entity creation: restore the
    # cached PNG first, then try the cloud (no-op without a session).
    await map_coordinator.async_load_cached()
    await map_coordinator.async_refresh()

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(
    hass: HomeAssistant,
    entry: XiaomiVacuumConfigEntry,
) -> bool:
    """Handle removal of an entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(
    hass: HomeAssistant,
    entry: XiaomiVacuumConfigEntry,
) -> None:
    """Delete the per-entry repair issues when the entry itself is deleted."""
    async_clear_cannot_connect(hass, entry)
    async_clear_unsupported_model(hass, entry)
