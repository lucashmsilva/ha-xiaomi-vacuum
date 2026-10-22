"""Coordinator that periodically downloads & renders the vacuum map (cloud-backed)."""

from __future__ import annotations

import base64
import binascii
import contextlib
import io
import json
import struct
import zlib
from datetime import timedelta
from typing import TYPE_CHECKING, NotRequired, TypedDict

from PIL import Image, ImageDraw
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from xiaomi_vacuum_sdk import (
    CoordinateSystem,
    MapParseError,
    MapPoint,
    MapRenderer,
    RenderOptions,
)

from .cloud import XiaomiCloudAuthError
from .const import DOMAIN, LOGGER

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .cloud import XiaomiCloud
    from .coordinator import XiaomiVacuumDataUpdateCoordinator
    from .data import MapCalibration, RenderedMap, XiaomiVacuumConfigEntry

MAP_UPDATE_INTERVAL = timedelta(seconds=30)
MAP_STORAGE_VERSION = 1

# Trajectory blobs are base64+zlib streams of 9-byte records:
# one flag byte (2 = travel, 3 = cleaning) followed by little-endian int32 x/y.
_TRAJECTORY_POINT_SIZE = 9
_PATH_COLORS = {
    2: (0, 122, 255, 220),
    3: (255, 149, 0, 235),
}
_PATH_COLOR_DEFAULT = (120, 120, 120, 200)
_PATH_UNDERLAY = (20, 34, 20, 110)


class _StoredMap(TypedDict):
    """On-disk shape of the cached map."""

    png_b64: str
    calibration: NotRequired[MapCalibration]
    path: NotRequired[list[list[int]]]


class XiaomiVacuumMapCoordinator(DataUpdateCoordinator["RenderedMap | None"]):
    """Polls the cloud for the latest map binary and renders it to PNG bytes."""

    config_entry: XiaomiVacuumConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        cloud: XiaomiCloud | None,
        state_coordinator: XiaomiVacuumDataUpdateCoordinator,
    ) -> None:
        """Initialize."""
        super().__init__(
            hass=hass,
            logger=LOGGER,
            name=f"{DOMAIN}_map",
            update_interval=MAP_UPDATE_INTERVAL,
            config_entry=state_coordinator.config_entry,
        )
        self._cloud = cloud
        self._state_coordinator = state_coordinator
        self._last_raw: bytes | None = None
        self._last_trajectory_raw: bytes | None = None
        self._base_png: bytes | None = None
        self._base_calibration: MapCalibration | None = None
        self._trajectory: list[tuple[int, int, int]] = []
        self._store: Store[_StoredMap] = Store(
            hass,
            MAP_STORAGE_VERSION,
            f"{DOMAIN}.map_{state_coordinator.config_entry.entry_id}",
        )
        # The SDK's default RenderOptions carry this integration's palette,
        # scale and element sizes; only the 12 px breathing border is added
        # on top of the legacy render.
        self._renderer = MapRenderer(RenderOptions())

    @property
    def has_cloud(self) -> bool:
        """Whether map updates can be fetched from the Xiaomi cloud."""
        return self._cloud is not None

    def set_cloud(self, cloud: XiaomiCloud | None) -> None:
        """Attach (or detach) the cloud client used for map updates."""
        self._cloud = cloud
        # Force the next poll to fetch and render even if the blobs are unchanged.
        self._last_raw = None
        self._last_trajectory_raw = None

    async def async_load_cached(self) -> None:
        """Restore the last rendered map PNG from disk so it survives restarts."""
        stored = await self._store.async_load()
        if not stored:
            return
        try:
            png = base64.b64decode(stored["png_b64"], validate=True)
        except (KeyError, ValueError) as exception:
            LOGGER.warning("Discarding corrupt cached map: %s", exception)
            return
        calibration = stored.get("calibration")
        path = _parse_stored_path(stored.get("path"))
        self._base_png = png
        self._base_calibration = calibration
        self._trajectory = path
        if calibration is not None and path:
            png = await self.hass.async_add_executor_job(
                _render_path, png, calibration, path
            )
        LOGGER.debug("Restored cached map PNG: %s bytes", len(png))
        self.async_set_updated_data({"png": png, "calibration": calibration})

    async def _async_update_data(self) -> RenderedMap | None:
        # Without a cloud session the coordinator is cache-only: it keeps
        # serving the PNG restored from disk and never blocks local entities.
        if self._cloud is None:
            return self.data
        # State is None while the robot has been offline since startup; keep
        # serving whatever we have (possibly the disk-restored PNG).
        state = self._state_coordinator.data
        map_obj = self._extract_obj_name(state.get("map_obj_name")) if state else None
        trajectory_obj = (
            self._extract_obj_name(state.get("trajectory_obj_name")) if state else None
        )
        device = self._cloud.device
        if not map_obj or device is None:
            LOGGER.debug("No map obj_name or cloud device; skipping map update")
            return self.data

        cache_dirty = False
        try:
            raw = await self._cloud.async_get_map_bytes(map_obj)
            LOGGER.debug("Map blob: obj=%s bytes=%s", map_obj, len(raw) if raw else 0)
            if raw and raw != self._last_raw:
                rendered = await self.hass.async_add_executor_job(
                    self._render_blob, raw, device.model, str(device.device_id)
                )
                if rendered is not None:
                    # Marked as seen only after a successful render, so a
                    # transient parse failure is retried on the next poll.
                    self._last_raw = raw
                    self._base_png = rendered["png"]
                    self._base_calibration = rendered["calibration"]
                    cache_dirty = True
                else:
                    LOGGER.debug("Renderer returned no image for obj=%s", map_obj)

            if self._base_png is None or self._base_calibration is None:
                return self.data

            if trajectory_obj:
                try:
                    trajectory_raw = await self._cloud.async_get_map_bytes(
                        trajectory_obj
                    )
                except XiaomiCloudAuthError:
                    raise
                except Exception as exception:
                    # The path is a nice-to-have: a failed trajectory download
                    # must not discard the freshly rendered floor plan.
                    LOGGER.debug(
                        "Trajectory fetch failed; keeping last path: %s", exception
                    )
                    trajectory_raw = None
                if trajectory_raw and trajectory_raw != self._last_trajectory_raw:
                    path = await self.hass.async_add_executor_job(
                        _parse_trajectory, trajectory_raw
                    )
                    if path is not None:
                        self._last_trajectory_raw = trajectory_raw
                        self._trajectory = path
                        cache_dirty = True
        except XiaomiCloudAuthError as exception:
            # The stored session expired: prompt reauth (HA dedupes concurrent
            # flows) and keep serving the last rendered map meanwhile.
            LOGGER.warning("Cloud session rejected; starting reauth: %s", exception)
            self.config_entry.async_start_reauth(self.hass)
            return self.data
        except Exception as exception:
            raise UpdateFailed(exception) from exception

        if cache_dirty:
            await self._save_cache()

        png = self._base_png
        if self._trajectory:
            try:
                png = await self.hass.async_add_executor_job(
                    _render_path, png, self._base_calibration, self._trajectory
                )
            except Exception as exception:  # pylint: disable=broad-except
                LOGGER.warning("Could not draw the cleaning path: %s", exception)
                png = self._base_png
        if (
            self.data is not None
            and png == self.data["png"]
            and self._base_calibration == self.data["calibration"]
        ):
            return self.data
        return {"png": png, "calibration": self._base_calibration}

    async def _save_cache(self) -> None:
        """Persist the base map and last trajectory (never the position overlay)."""
        if self._base_png is None:
            return
        stored: _StoredMap = {"png_b64": base64.b64encode(self._base_png).decode()}
        if self._base_calibration is not None:
            stored["calibration"] = self._base_calibration
        if self._trajectory:
            stored["path"] = [list(point) for point in self._trajectory]
        await self._store.async_save(stored)

    def _render_blob(
        self, raw: bytes, model: str, device_id: str
    ) -> RenderedMap | None:
        """
        Decrypt, parse and render the map blob (CPU-bound, runs in the executor).

        ``render_map`` parses the blob once and returns the projection it drew
        with, so the stored calibration describes the served PNG exactly.
        """
        try:
            rendered = self._renderer.render_map(raw, model=model, device_id=device_id)
        except MapParseError as exception:
            # The device published a payload without a drawable map (e.g. a
            # fresh map still being built) — keep serving the previous image.
            LOGGER.debug("Map payload not drawable: %s", exception)
            return None
        coordinates = rendered.coordinates
        calibration: MapCalibration = {
            "origin_x": coordinates.origin_x,
            "origin_y": coordinates.origin_y,
            "resolution": coordinates.resolution,
            "width": rendered.map_data.width,
            "height": coordinates.grid_height,
            "scale": coordinates.scale,
            # RenderOptions.border is an int; CoordinateSystem widens it.
            "border": int(coordinates.offset),
        }
        return {"png": rendered.png, "calibration": calibration}

    @staticmethod
    def _extract_obj_name(raw_field: str | None) -> str | None:
        """Extract the inner obj_name from the MIoT property's JSON envelope."""
        if not raw_field:
            return None
        with contextlib.suppress(json.JSONDecodeError, ValueError, TypeError):
            payload = json.loads(raw_field)
            if isinstance(payload, dict) and (name := payload.get("obj_name")):
                return str(name)
        return raw_field if isinstance(raw_field, str) else None


def _parse_stored_path(raw: object) -> list[tuple[int, int, int]]:
    """Validate a trajectory restored from disk."""
    if not isinstance(raw, list):
        return []
    path: list[tuple[int, int, int]] = []
    for point in raw:
        if (
            isinstance(point, list)
            and len(point) == 3
            and all(isinstance(value, int) for value in point)
        ):
            path.append((point[0], point[1], point[2]))
    return path


def _parse_trajectory(raw: bytes) -> list[tuple[int, int, int]] | None:
    """Decode a trajectory blob into (flag, x, y) points in millimetres."""
    try:
        data = zlib.decompress(base64.b64decode(raw))
    except (ValueError, binascii.Error, zlib.error) as exception:
        LOGGER.debug("Could not decode trajectory blob: %s", exception)
        return None
    if not data or len(data) % _TRAJECTORY_POINT_SIZE:
        LOGGER.debug("Unexpected trajectory size: %s bytes", len(data))
        return None
    return [
        struct.unpack_from("<Bii", data, offset)
        for offset in range(0, len(data), _TRAJECTORY_POINT_SIZE)
    ]


def _render_path(
    png: bytes,
    calibration: MapCalibration,
    points: list[tuple[int, int, int]],
) -> bytes:
    """Draw the trajectory on top of the floor plan."""
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    coordinates = _coordinate_system(calibration)
    pixels = [coordinates.to_image(MapPoint(x=x, y=y)) for _, x, y in points]
    if len(pixels) < 2:
        return png
    # A translucent dark underlay keeps the coloured path readable on the
    # muted room palette.
    draw.line(pixels, fill=_PATH_UNDERLAY, width=6, joint="curve")
    for flag, run in _path_runs(points):
        segment = [coordinates.to_image(MapPoint(x=x, y=y)) for _, x, y in run]
        if len(segment) >= 2:
            draw.line(
                segment,
                fill=_PATH_COLORS.get(flag, _PATH_COLOR_DEFAULT),
                width=3,
                joint="curve",
            )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _path_runs(
    points: list[tuple[int, int, int]],
) -> list[tuple[int, list[tuple[int, int, int]]]]:
    """Split the trajectory into contiguous runs of the same flag."""
    runs: list[tuple[int, list[tuple[int, int, int]]]] = []
    start = 0
    current = points[0][0]
    for index, (flag, _, _) in enumerate(points[1:], 1):
        if flag != current:
            runs.append((current, points[start:index]))
            start = index
            current = flag
    runs.append((current, points[start:]))
    return runs


def _coordinate_system(calibration: MapCalibration) -> CoordinateSystem:
    """Rebuild the projection the renderer used for this PNG."""
    return CoordinateSystem(
        origin_x=calibration["origin_x"],
        origin_y=calibration["origin_y"],
        resolution=calibration["resolution"],
        grid_height=calibration["height"],
        scale=calibration["scale"],
        offset=calibration["border"],
    )
