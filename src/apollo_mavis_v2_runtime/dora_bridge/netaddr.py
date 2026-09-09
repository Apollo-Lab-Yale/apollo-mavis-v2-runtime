"""``dora.bind_host`` resolution and vetting (14-dora §9 "Bind address").

``bind_host`` is either an IPv4 literal or a Linux interface name (``wlp38s0``,
``tailscale0``) whose CURRENT IPv4 is looked up with the ``SIOCGIFADDR`` ioctl
(pure stdlib; the lab Wi-Fi is DHCP so the address changes between boots and
the bridge re-resolves before every dataflow restart). The vet refuses
``0.0.0.0`` and any address that lies inside a control-box subnet — the
host's own addresses on the two arm links (192.168.1.11 / 192.168.2.12,
derived from the ``workcells.hardware`` arm IPs as /24 networks). The xArm
SDK path over those NICs is untouched; this only decides where dora LISTENS.
"""

from __future__ import annotations

import ipaddress
import socket
import struct
from collections.abc import Iterable

LOOPBACK_NAMES = ("localhost", "lo")


class BindHostError(ValueError):
    """``bind_host`` cannot be used (message is the ``ExternalStatus.detail``)."""


def is_loopback(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return ip in LOOPBACK_NAMES


def interface_ipv4(name: str) -> str | None:
    """Current IPv4 of interface ``name`` (Linux ``SIOCGIFADDR``); ``None`` when the
    interface does not exist or has no IPv4 (Wi-Fi not associated yet)."""
    try:
        import fcntl  # Linux only; the runtime targets Ubuntu 22.04
    except ImportError:  # pragma: no cover - non-Linux dev box
        return None
    if not name or len(name) > 15:
        return None
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", name.encode("ascii")[:15])
        res = fcntl.ioctl(s.fileno(), 0x8915, packed)  # SIOCGIFADDR
        return socket.inet_ntoa(res[20:24])
    except (OSError, UnicodeEncodeError):
        return None
    finally:
        s.close()


def resolve_bind_host(value: str) -> str:
    """IPv4 literal -> itself; ``localhost``/``lo`` -> 127.0.0.1; interface name ->
    its current IPv4. Raises :class:`BindHostError` when nothing resolves."""
    v = value.strip()
    if v in LOOPBACK_NAMES:
        return "127.0.0.1"
    try:
        return str(ipaddress.IPv4Address(v))
    except ValueError:
        pass
    ip = interface_ipv4(v)
    if ip is None:
        raise BindHostError(
            f"dora.bind_host {value!r} is neither an IPv4 address nor an interface with an "
            "IPv4 address (Wi-Fi not associated? interface renamed?)"
        )
    return ip


def control_box_subnets(arm_ips: Iterable[str | None]) -> list[ipaddress.IPv4Network]:
    """The /24 networks of the configured control-box IPs (192.168.1.0/24, 192.168.2.0/24)."""
    nets: list[ipaddress.IPv4Network] = []
    for ip in arm_ips:
        if not ip:
            continue
        try:
            net = ipaddress.ip_network(f"{ip}/24", strict=False)
        except ValueError:
            continue
        if isinstance(net, ipaddress.IPv4Network) and net not in nets:
            nets.append(net)
    return nets


def vet_bind_ip(ip: str, arm_ips: Iterable[str | None]) -> None:
    """Refuse 0.0.0.0 / unspecified and any address inside a control-box subnet."""
    addr = ipaddress.ip_address(ip)
    if addr.is_unspecified:
        raise BindHostError(
            "dora.bind_host 0.0.0.0 is refused: the control plane binds ONE interface "
            "(14-dora §9); use 127.0.0.1, the lab Wi-Fi address or its interface name"
        )
    if addr.is_multicast:
        raise BindHostError(f"dora.bind_host {ip} is a multicast address")
    for net in control_box_subnets(arm_ips):
        if addr in net:
            raise BindHostError(
                f"dora.bind_host {ip} lies in the control-box subnet {net} (an arm link NIC); "
                "dora must never listen on the arm links (14-dora §9)"
            )


def resolve_and_vet(value: str, arm_ips: Iterable[str | None]) -> str:
    ip = resolve_bind_host(value)
    vet_bind_ip(ip, arm_ips)
    return ip


__all__ = [
    "BindHostError",
    "control_box_subnets",
    "interface_ipv4",
    "is_loopback",
    "resolve_and_vet",
    "resolve_bind_host",
    "vet_bind_ip",
]
