"""Map image entity for xiaomi_vacuum (cloud-backed)."""

from __future__ import annotations

import io
import json
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from PIL import Image, ImageDraw
from homeassistant.components.image import ImageEntity
from homeassistant.core import callback
from xiaomi_vacuum_sdk import CoordinateSystem, MapPoint

from ..entity import XiaomiVacuumEntity  # noqa: TID252

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from ..coordinator import XiaomiVacuumDataUpdateCoordinator  # noqa: TID252
    from ..data import MapCalibration  # noqa: TID252
    from ..map_coordinator import XiaomiVacuumMapCoordinator  # noqa: TID252


class _Point(TypedDict):
    """A 2-D point, in image pixels or device millimetres depending on context."""

    x: int
    y: int


class CalibrationPoint(TypedDict):
    """One pixel ↔ device-coordinate pair, in the Xiaomi Vacuum Map Card's format."""

    vacuum: _Point
    map: _Point


class _Position(TypedDict):
    """The robot's live position in device map millimetres."""

    x: float
    y: float
    angle: float | None


class _MapAttributes(TypedDict):
    """Attributes letting a map card translate clicks into device coordinates."""

    calibration_points: list[CalibrationPoint] | None
    calibration: MapCalibration | None
    vacuum_position: _Position | None


class XiaomiVacuumMap(XiaomiVacuumEntity, ImageEntity):
    """Renders the vacuum map produced by XiaomiVacuumMapCoordinator."""

    _attr_translation_key = "map"
    _attr_content_type = "image/png"
    # Map geometry and the frequent position ping are not history worth keeping.
    _unrecorded_attributes = frozenset(
        {"calibration", "calibration_points", "vacuum_position"}
    )

    def __init__(
        self,
        hass: HomeAssistant,
        state_coordinator: XiaomiVacuumDataUpdateCoordinator,
        map_coordinator: XiaomiVacuumMapCoordinator,
    ) -> None:
        """Initialize."""
        XiaomiVacuumEntity.__init__(self, state_coordinator)
        ImageEntity.__init__(self, hass)
        self._map_coordinator = map_coordinator
        self._last_image: bytes | None = None
        self._last_calibration: MapCalibration | None = None
        self._last_position: MapPoint | None = None
        self._attr_image_last_updated = datetime.now(UTC)

    @property
    def unique_id(self) -> str:
        """Return a stable unique id for this entity."""
        return f"{self.coordinator.config_entry.entry_id}_map"

    @property
    def available(self) -> bool:
        """
        Available whenever a map image exists, even if the robot is offline.

        Decoupled from the state coordinator on purpose: an unavailable image
        entity makes the frontend request the map with ``token=undefined``,
        which Home Assistant logs as an invalid-authentication attempt.
        """
        return self._map_coordinator.data is not None or self._last_image is not None

    @property
    def extra_state_attributes(self) -> _MapAttributes:
        """Expose map calibration and the live robot position."""
        calibration = self._last_calibration
        position = self._last_position
        return {
            "calibration_points": (
                calibration_points(calibration) if calibration is not None else None
            ),
            "calibration": calibration,
            "vacuum_position": (
                {
                    "x": position.x,
                    "y": position.y,
                    "angle": position.angle,
                }
                if position is not None
                else None
            ),
        }

    async def async_added_to_hass(self) -> None:
        """Subscribe to the map coordinator for refresh on new map data."""
        await super().async_added_to_hass()
        self._last_position = _parse_vacuum_position(
            (self.coordinator.data or {}).get("vacuum_position")
        )
        self.async_on_remove(
            self._map_coordinator.async_add_listener(self._handle_new_map)
        )
        # Surface a disk-restored map immediately after a restart.
        if self._map_coordinator.data is not None:
            self._handle_new_map()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Refresh the served image whenever the robot position changes."""
        position = _parse_vacuum_position(
            (self.coordinator.data or {}).get("vacuum_position")
        )
        if position != self._last_position:
            self._last_position = position
            self._attr_image_last_updated = datetime.now(UTC)
            # Rotate the image token so the frontend re-fetches the PNG.
            self.async_update_token()
        super()._handle_coordinator_update()

    def _handle_new_map(self) -> None:
        rendered = self._map_coordinator.data
        # The calibration is compared too: a cache written before calibration
        # existed restores it as None, and the first poll can re-render the
        # very same PNG — comparing only bytes would drop the calibration
        # until the map itself next changes, which can take hours docked.
        if rendered is None or (
            rendered["png"] == self._last_image
            and rendered["calibration"] == self._last_calibration
        ):
            return
        self._last_image = rendered["png"]
        self._last_calibration = rendered["calibration"]
        self._attr_image_last_updated = datetime.now(UTC)
        self.async_update_token()
        self.async_write_ha_state()

    async def async_image(self) -> bytes | None:
        """Serve the freshest rendered PNG with the live position overlaid."""
        rendered = self._map_coordinator.data
        if rendered is not None:
            self._last_image = rendered["png"]
            self._last_calibration = rendered["calibration"]
        if self._last_image is None:
            return None
        position = _parse_vacuum_position(
            (self.coordinator.data or {}).get("vacuum_position")
        )
        calibration = self._last_calibration
        if position is None or calibration is None:
            return self._last_image
        return await self.hass.async_add_executor_job(
            _render_position, self._last_image, calibration, position
        )


def calibration_points(calibration: MapCalibration) -> list[CalibrationPoint]:
    """
    Three pixel ↔ millimetre pairs describing the served PNG.

    The projection is the SDK's own ``CoordinateSystem``, rebuilt from the
    stored calibration so the pairs stay the exact inverse of what the
    renderer drew. The pairs are the floor image's top-left, top-right and
    bottom-left corners, which is the three-point form the Xiaomi Vacuum Map
    Card accepts as a ``calibration_source``.
    """
    coordinates = _coordinate_system(calibration)
    scale = calibration["scale"]
    border = calibration["border"]
    floor_width = calibration["width"] * scale
    floor_height = calibration["height"] * scale
    return [
        _calibration_point(coordinates, border, border),
        _calibration_point(coordinates, border + floor_width, border),
        _calibration_point(coordinates, border, border + floor_height),
    ]


def _calibration_point(
    coordinates: CoordinateSystem, pixel_x: float, pixel_y: float
) -> CalibrationPoint:
    """Project one PNG pixel back onto the device's millimetre frame."""
    point = coordinates.to_device(pixel_x, pixel_y)
    return {
        "map": {"x": round(pixel_x), "y": round(pixel_y)},
        "vacuum": {"x": round(point.x), "y": round(point.y)},
    }


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


def _render_position(
    png: bytes, calibration: MapCalibration, position: MapPoint
) -> bytes:
    """Draw the robot marker on top of the rendered map PNG."""
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    x, y = _coordinate_system(calibration).to_image(position)
    radius = 18
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(255, 59, 48, 235),
        outline=(255, 255, 255, 255),
        width=3,
    )
    if position.angle is not None:
        heading = math.radians(position.angle)
        tip = (
            x + (radius + 7) * math.cos(heading),
            y - (radius + 7) * math.sin(heading),
        )
        draw.line((x, y, *tip), fill=(255, 255, 255, 255), width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _parse_vacuum_position(raw: str | None) -> MapPoint | None:
    """Parse the `vacuum-position` property into a map point."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError, TypeError:
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("position")
    if isinstance(value, dict):
        x, y, yaw = value.get("x"), value.get("y"), value.get("yaw")
    elif isinstance(value, list) and len(value) >= 2:
        x, y = value[0], value[1]
        yaw = value[2] if len(value) > 2 else None
    else:
        x, y, yaw = data.get("x"), data.get("y"), data.get("yaw")
    if not isinstance(x, int | float) or not isinstance(y, int | float):
        return None
    angle = _yaw_to_degrees(yaw) if isinstance(yaw, int | float) else None
    return MapPoint(x=float(x), y=float(y), angle=angle)


def _yaw_to_degrees(yaw: float) -> float:
    """Normalize the property's yaw to degrees (radians/centi-degrees/degrees)."""
    if abs(yaw) <= 2 * math.pi + 0.001:
        return yaw * 180.0 / math.pi
    if abs(yaw) > 180.0:  # noqa: PLR2004 -- unit heuristic, same as the SDK parser
        return (yaw / 100.0) % 180.0
    return yaw % 180.0
