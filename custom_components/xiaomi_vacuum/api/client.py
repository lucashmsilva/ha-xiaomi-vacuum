"""Local MIoT client for xiaomi_vacuum."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from xiaomi_vacuum_sdk import (
    ActionAddress,
    MiotAckTimeoutError,
    MiotClient,
    MiotError,
    PropertyAddress,
    PropertyValue,
)

from ..const import LOGGER  # noqa: TID252
from ..spec import Property  # noqa: TID252
from .custom_cleaning import CustomCleaning
from .errors import (
    XiaomiVacuumApiClientCommunicationError,
    XiaomiVacuumApiClientError,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from ..data import (  # noqa: TID252
        DeviceInfoLike,
        JsonObject,
        JsonValue,
        VacuumState,
    )
    from ..spec import ModelSpec  # noqa: TID252
    from ..spec.addresses import MiotActionAddress  # noqa: TID252
    from .cleaning_zone import CleaningZone

# Xiaomi vacuums routinely take >5 s to ack action commands (start / pause /
# room-sweep) — the device is busy spinning up and misses the default read
# window, surfacing as `-9999 user ack timeout`. 10 s is a forgiving middle
# ground that avoids most spurious failures without making a genuinely-offline
# device feel unresponsive.
_DEVICE_TIMEOUT_SECONDS = 10.0

# The vacuums silently drop any reply that does not fit in a single UDP
# datagram. The room list grows with the number and names of the rooms (non-ASCII
# names are escaped), so batching it with other properties can push the reply
# past that limit and leave every poll unanswered.
_STANDALONE_PROPERTIES = frozenset({Property.ROOM_INFORMATION})


class XiaomiVacuumApiClient:
    """Local MIoT client backed by ``xiaomi-vacuum-sdk`` for one model."""

    def __init__(self, host: str, token: str, spec: ModelSpec) -> None:
        """Initialize with the model's spec (property + action mapping)."""
        self._spec = spec
        self._client = MiotClient(host, token, timeout=_DEVICE_TIMEOUT_SECONDS)
        self._properties: dict[str, PropertyAddress] = {
            name: PropertyAddress(siid=address["siid"], piid=address["piid"])
            for name, address in spec.property_mapping.items()
        }
        self._property_requests = _group_property_requests(self._properties)

    async def _run[T](self, operation: Awaitable[T]) -> T:
        """Await one SDK call, normalizing errors to the integration types."""
        try:
            return await operation
        except MiotError as exception:
            msg = f"Device error: {exception}"
            raise XiaomiVacuumApiClientCommunicationError(msg) from exception
        except Exception as exception:  # pylint: disable=broad-except
            msg = f"Unexpected error: {exception}"
            raise XiaomiVacuumApiClientError(msg) from exception

    async def _run_action(self, operation: Awaitable[JsonValue]) -> None:
        """
        Run an MIoT action, tolerating the flaky ``-9999 user ack timeout``.

        Xiaomi vacuums often accept a command (start / pause / room-sweep) and
        then never send the ack because the robot is already busy moving. The
        command usually *did* reach the device, so we treat an ack-timeout as
        accepted instead of bubbling a communication error up to Home Assistant:
        the optimistic UI patch + the scheduled ~5 s refresh will reconcile the
        real state. Any other device error still raises.
        """
        try:
            await operation
        except MiotAckTimeoutError:
            LOGGER.warning(
                "Device did not ack the action in time (-9999); assuming it "
                "was accepted. State will be confirmed on the next refresh."
            )
        except MiotError as exception:
            msg = f"Device error: {exception}"
            raise XiaomiVacuumApiClientCommunicationError(msg) from exception
        except Exception as exception:  # pylint: disable=broad-except
            msg = f"Unexpected error: {exception}"
            raise XiaomiVacuumApiClientError(msg) from exception

    async def async_get_info(self) -> DeviceInfoLike:
        """Handshake — returns the device identity (model, mac, fw)."""
        return await self._run(self._client.info())

    async def async_get_state(self) -> VacuumState:
        """Read all mapped properties, keyed by the spec's property names."""
        values: dict[str, PropertyValue] = {}
        for properties in self._property_requests:
            values |= await self._run(self._client.get_properties(properties))
        LOGGER.debug("Parsed state: %s", values)
        return cast("VacuumState", values)

    async def async_start(self) -> None:
        """Start sweeping."""
        await self._run_action(
            self._client.call_action(_address(self._spec.actions.start_sweep))
        )

    async def async_continue(self) -> None:
        """Resume a paused job (keeps the current task instead of restarting)."""
        await self._run_action(
            self._client.call_action(_address(self._spec.actions.continue_sweep))
        )

    async def async_pause(self) -> None:
        """Pause current job."""
        await self._run_action(
            self._client.call_action(_address(self._spec.actions.pause_sweeping))
        )

    async def async_stop(self) -> None:
        """Stop current job."""
        await self._run_action(
            self._client.call_action(_address(self._spec.actions.stop_sweeping))
        )

    async def async_return_home(self) -> None:
        """Stop and return to dock."""
        await self._run_action(
            self._client.call_action(_address(self._spec.actions.return_home))
        )

    async def async_locate(self) -> None:
        """Identify (beep + light)."""
        await self._run_action(
            self._client.call_action(_address(self._spec.actions.identify))
        )

    async def async_call_action(self, siid: int, aiid: int) -> None:
        """Invoke an arbitrary MIoT action (backs the send_command service)."""
        await self._run_action(
            self._client.call_action(ActionAddress(siid=siid, aiid=aiid))
        )

    async def async_start_dust_arrest(self) -> None:
        """Trigger dock to empty the vacuum's dust bin (X20 Max only)."""
        action = self._spec.actions.start_dust_arrest
        if action is None:
            msg = "This model has no dust-arrest dock"
            raise XiaomiVacuumApiClientError(msg)
        await self._run_action(self._client.call_action(_address(action)))

    async def async_set_fan_speed(self, fan_speed: str) -> None:
        """Set fan speed by label (Silent/Basic/Strong/Full Speed)."""
        speeds = self._spec.fan_speeds
        if fan_speed not in speeds:
            msg = f"Unknown fan speed: {fan_speed}"
            raise XiaomiVacuumApiClientError(msg)
        await self.async_set_property(Property.FAN_SPEED, speeds[fan_speed])

    async def async_set_sweep_mop_type(self, mode: str) -> None:
        """Set sweep/mop mode (sweep / mop / sweep_mop / sweep_before_mopping)."""
        types = self._spec.sweep_mop_types
        if mode not in types:
            msg = f"Unknown sweep_mop_type: {mode}"
            raise XiaomiVacuumApiClientError(msg)
        await self.async_set_property(Property.SWEEP_MOP_TYPE, types[mode])

    async def async_set_property(self, name: Property, value: int) -> None:
        """Set a MIoT property by mapping name (raw integer value)."""
        address = self._properties.get(name)
        if address is None:
            msg = f"Unknown property: {name}"
            raise XiaomiVacuumApiClientError(msg)
        await self._run(self._client.set_property(address, value))

    async def async_clean_segments(
        self, segment_ids: list[str], room_information: str | None = None
    ) -> None:
        """
        Start cleaning specific rooms by their MIoT room IDs.

        Two strategies, picked by the active model's spec:

        * ``direct`` (X20 Max) — fire ``start-vacuum-room-sweep`` with the room
          ids as a comma-separated string.
        * ``config_then_custom`` (S20+) — captured from the Mi Home app: rebuild
          the room config from ``room_information`` with the target rooms'
          ``on`` flag set to True (others False), push it via
          ``set-room-clean-configs``, then fire ``start-custom-sweep``. The
          spec's ``start-vacuum-room-sweep`` (aiid 13) exists but the robot
          ignores it and returns to the dock.
        """
        strategy = self._spec.room_clean_strategy
        actions = self._spec.actions
        if strategy == "direct":
            if actions.start_room_sweep is None:
                msg = "direct room-clean strategy but no start_room_sweep action"
                raise XiaomiVacuumApiClientError(msg)
            action = actions.start_room_sweep
            payload: list[JsonValue] = [
                {
                    "piid": action["in_piid"],
                    "value": ",".join(str(r) for r in segment_ids),
                }
            ]
            LOGGER.debug("Calling start-vacuum-room-sweep with payload: %s", payload)
            await self._run_action(self._client.call_action(_address(action), payload))
            return

        # config_then_custom (S20+)
        if actions.set_room_clean_configs is None or actions.start_custom_sweep is None:
            msg = (
                "config_then_custom room-clean strategy requires both "
                "set_room_clean_configs and start_custom_sweep actions"
            )
            raise XiaomiVacuumApiClientError(msg)
        config = _build_room_clean_config(room_information, segment_ids)
        if not config:
            msg = (
                "Cannot start room clean: the device published no room "
                "information. Configure rooms in the Mi Home app first."
            )
            raise XiaomiVacuumApiClientError(msg)
        await self._async_start_config_room_clean(config)

    async def async_clean_rooms(
        self,
        rooms: Sequence[dict[str, JsonValue]],
        room_information: str | None = None,
    ) -> None:
        """
        Start a room clean with per-room fan/water/repeat overrides.

        Like :meth:`async_clean_segments`, but each entry may also carry
        ``fan_level``, ``water_level`` and ``clean_times`` values (already
        mapped to the device's raw integers). The requested order is the
        cleaning order. Only fields present in an entry are overridden; every
        other per-room setting is preserved from ``room_information``.
        """
        actions = self._spec.actions
        if actions.set_room_clean_configs is None or actions.start_custom_sweep is None:
            msg = "This model does not support per-room cleaning settings"
            raise XiaomiVacuumApiClientError(msg)
        wanted_order = [str(room["id"]) for room in rooms]
        overrides = {
            str(room["id"]): {
                key: value
                for key, value in room.items()
                if key in _ROOM_OVERRIDE_KEYS and value is not None
            }
            for room in rooms
        }
        config = _build_room_clean_config(
            room_information, wanted_order, overrides=overrides
        )
        if not config:
            msg = (
                "Cannot start room clean: the device published no room "
                "information. Configure rooms in the Mi Home app first."
            )
            raise XiaomiVacuumApiClientError(msg)
        selected = {str(room.get("id")) for room in config if room.get("on")}
        missing = [room_id for room_id in wanted_order if room_id not in selected]
        if missing:
            msg = f"Unknown room id(s): {', '.join(missing)}"
            raise XiaomiVacuumApiClientError(msg)
        await self._async_start_config_room_clean(config)

    async def _async_start_config_room_clean(
        self, config: list[JsonObject]
    ) -> None:
        """Push the S20+ room config and start the saved custom sweep."""
        actions = self._spec.actions
        if actions.set_room_clean_configs is None or actions.start_custom_sweep is None:
            msg = (
                "config_then_custom room-clean strategy requires both "
                "set_room_clean_configs and start_custom_sweep actions"
            )
            raise XiaomiVacuumApiClientError(msg)
        set_action = actions.set_room_clean_configs
        config_payload = [json.dumps({"room_attrs": config})]
        LOGGER.debug(
            "Calling set-room-clean-configs (%s/%s) with %d room(s), cleaning %d",
            set_action["siid"],
            set_action["aiid"],
            len(config),
            sum(1 for r in config if r.get("on")),
        )
        await self._run_action(
            self._client.call_action(_address(set_action), config_payload)
        )
        start_action = actions.start_custom_sweep
        LOGGER.debug(
            "Calling start-custom-sweep (%s/%s)",
            start_action["siid"],
            start_action["aiid"],
        )
        await self._run_action(self._client.call_action(_address(start_action), []))

    async def async_clean_zones(
        self, zones: Sequence[CleaningZone], repeats: int | None = None
    ) -> None:
        """
        Start cleaning rectangular zones, optionally setting the repeat count.

        The device acknowledges any ``start-zone-sweep`` payload with ``code: 0``
        but only moves for the block list the Mi Home plugin sends; repeats are
        not part of that payload, so they go to the clean-times property first.
        """
        action = self._spec.actions.start_zone_sweep
        if action is None:
            msg = "Failed to clean zones: this model has no zone-sweep action"
            raise XiaomiVacuumApiClientError(msg)
        if repeats is not None:
            await self.async_set_property(Property.CLEAN_TIMES, repeats)
        blocks = json.dumps([zone.to_block() for zone in zones], separators=(",", ":"))
        payload: list[JsonValue] = [{"piid": action["in_piid"], "value": blocks}]
        LOGGER.debug("Calling start-zone-sweep with payload: %s", payload)
        await self._run_action(self._client.call_action(_address(action), payload))

    async def async_get_custom_cleanings(self) -> list[CustomCleaning]:
        """Read the custom cleanings saved in the Mi Home app."""
        address = self._spec.user_define_sweep_config
        if address is None:
            msg = "Failed to read custom cleanings: this model has no custom cleanings"
            raise XiaomiVacuumApiClientError(msg)
        values = await self._run(
            self._client.get_properties(
                {"config": PropertyAddress(siid=address["siid"], piid=address["piid"])}
            )
        )
        return CustomCleaning.list_from_config(values.get("config"))

    async def async_start_custom_cleaning(self, cleaning: CustomCleaning) -> None:
        """Start a custom cleaning saved in the Mi Home app."""
        action = self._spec.actions.start_user_define_sweep
        if action is None:
            msg = "Failed to start custom cleaning: this model has no custom cleanings"
            raise XiaomiVacuumApiClientError(msg)
        payload: list[JsonValue] = [
            {"piid": action["in_piid"], "value": cleaning.identifier}
        ]
        LOGGER.debug("Calling start-user-define-sweep with payload: %s", payload)
        await self._run_action(self._client.call_action(_address(action), payload))

def _address(action: MiotActionAddress) -> ActionAddress:
    """Convert a spec action mapping entry to the SDK's address type."""
    return ActionAddress(siid=action["siid"], aiid=action["aiid"])


def _group_property_requests(
    properties: dict[str, PropertyAddress],
) -> list[dict[str, PropertyAddress]]:
    """Group the properties so each standalone one is read in its own request."""
    shared: dict[str, PropertyAddress] = {}
    standalone: list[dict[str, PropertyAddress]] = []
    for name, address in properties.items():
        if name in _STANDALONE_PROPERTIES:
            standalone.append({name: address})
        else:
            shared[name] = address
    return [shared, *standalone]


# Columns published by the S20+ `room-info` property (SIID 6 / piid 10) and
# accepted by `set-room-clean-configs` (SIID 2 / aiid 10). Order matters: the
# device publishes a table `room_attrs` whose first row is exactly this header.
_ROOM_ATTR_HEADERS = (
    "id",
    "room_name",
    "fan_level",
    "water_level",
    "clean_mode",
    "clean_times",
    "mop_mode",
    "on",
)

# Per-room fields a caller may override when starting a room clean.
_ROOM_OVERRIDE_KEYS = frozenset(
    {"fan_level", "water_level", "clean_mode", "clean_times", "mop_mode"}
)


def _build_room_clean_config(
    room_information: str | None,
    segment_ids: list[str],
    overrides: dict[str, dict[str, JsonValue]] | None = None,
) -> list[JsonObject]:
    """
    Build the S20+ ``set-room-clean-configs`` payload.

    Reads the current room config from ``room_information`` (the device's
    table-shaped ``room_attrs`` JSON), converts it to the object-array write
    format the action expects, and flips the ``on`` flag to True for every
    requested room id (False for the rest). Returns an empty list when the
    device published no rooms (caller should refuse to start in that case).

    The order of selected rooms is preserved from ``segment_ids``: the Mi Home
    app stores the cleaning sequence as the position of each room in
    ``room_attrs``, so we emit the requested rooms first (in the order the
    caller asked for), followed by the unselected ones in their original
    relative order. A ``clean_area`` call with ``[kitchen, living_room]`` thus
    cleans kitchen before living_room.

    ``overrides`` maps room id → per-room fields to replace on selected rooms
    (``fan_level``, ``water_level``, ``clean_mode``, ``clean_times``,
    ``mop_mode``). Unselected rooms keep the values published by the device.
    """
    if not room_information:
        return []
    try:
        data = json.loads(room_information)
    except ValueError, TypeError:
        return []
    matrix = data.get("room_attrs") if isinstance(data, dict) else None
    if not isinstance(matrix, list) or not matrix:
        return []
    header = matrix[0] if isinstance(matrix[0], list) else list(_ROOM_ATTR_HEADERS)
    # Normalise header cells to lower-case, matching ``_rows_to_attrs`` in
    # ``vacuum/cleaner.py``: both parsers read the same ``room_attrs`` payload,
    # so a device publishing ``Id`` / ``Room_Name`` instead of the lower-case
    # spelling resolves identically in both places. Without this the per-room
    # ``id`` lookup below fails on every row and the config comes back empty.
    header = [str(c).casefold() if c is not None else "" for c in header]
    # Requested sequence as an ordered lookup: position decides emit order.
    wanted_order = [str(s) for s in segment_ids]
    wanted = set(wanted_order)
    selected: list[JsonObject] = []
    others: list[JsonObject] = []
    for row in matrix[1:]:
        if not isinstance(row, list):
            continue
        room = dict(zip(header, row, strict=False))
        room_id = room.get("id")
        if room_id is None:
            continue
        room_id_str = str(room_id)
        room["on"] = room_id_str in wanted
        if room["on"] and overrides and room_id_str in overrides:
            room.update(overrides[room_id_str])
        if room["on"]:
            selected.append(room)
        else:
            others.append(room)
    # Reorder selected rooms to match the requested sequence (stable for any
    # ids the device knows but the caller didn't list, which shouldn't happen).
    selected.sort(key=lambda r: _ordered_index(str(r.get("id")), wanted_order))
    return selected + others


def _ordered_index(room_id: str, wanted_order: list[str]) -> int:
    """Position of ``room_id`` in ``wanted_order``, or a large sentinel if absent."""
    try:
        return wanted_order.index(room_id)
    except ValueError:
        return len(wanted_order)
