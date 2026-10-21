"""
Leitura da bateria do UPS HAT (DFRobot FIT0992, família Suptronics X120x).

- Fuel gauge no I2C 1, endereço 0x36: tensão no registrador 0x02 e carga (%) no
  0x04, ambos em words big-endian (por isso o swap de bytes).
- Falta de energia da tomada no GPIO 6 (nível alto = adaptador ligado), lido com o
  `pinctrl` do Raspberry Pi OS.

Tudo é opcional: sem o smbus2, sem o barramento ou sem o chip (ex: no PC de
desenvolvimento) a leitura devolve None e o app segue sem o ícone de bateria.
"""

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

I2C_BUS = 1
I2C_ADDRESS = 0x36
REG_VOLTAGE = 0x02
REG_CAPACITY = 0x04
AC_GPIO = 6

# Entre leituras (a tela consulta a cada ~20 s) e depois de uma falha, para não
# martelar um barramento que não responde.
_CACHE_SECONDS = 3.0
_RETRY_AFTER_FAILURE_SECONDS = 30.0


@dataclass(frozen=True)
class BatteryStatus:
    percent: float
    voltage: float
    ac_power: Optional[bool]  # None = não foi possível ler o GPIO

    @property
    def charging(self) -> Optional[bool]:
        if self.ac_power is None:
            return None
        return self.ac_power and self.percent < 100


def _swap16(word: int) -> int:
    return ((word & 0xFF) << 8) | (word >> 8)


def decode_voltage(word: int) -> float:
    """Tensão da bateria em volts (78,125 µV por LSB, valor nos 12 bits altos)."""
    return _swap16(word) * 1.25 / 1000 / 16


def decode_percent(word: int) -> float:
    """Carga em % (1/256 % por LSB), limitada a 0-100."""
    return min(100.0, max(0.0, _swap16(word) / 256))


def read_ac_power_with_pinctrl(gpio: int = AC_GPIO) -> Optional[bool]:
    """True se o adaptador está ligado (GPIO em nível alto); None se não deu para ler."""
    try:
        # Depois do boot o pino fica sem função e o `get` mostra "--": é preciso
        # configurá-lo como entrada antes (idempotente, não muda o pull).
        subprocess.run(
            ["pinctrl", "set", str(gpio), "ip"],
            capture_output=True,
            timeout=2,
            check=False,
        )
        out = subprocess.run(
            ["pinctrl", "get", str(gpio)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\|\s*(hi|lo)\b", out)
    if not match:
        return None
    return match.group(1) == "hi"


def _open_smbus(bus: int):
    from smbus2 import SMBus  # import tardio: só existe no dispositivo

    return SMBus(bus)


class BatteryMonitor:
    """Lê a bateria com cache; thread-safe e nunca levanta exceção."""

    def __init__(
        self,
        bus: int = I2C_BUS,
        address: int = I2C_ADDRESS,
        bus_factory: Callable[[int], object] = _open_smbus,
        ac_reader: Callable[[], Optional[bool]] = read_ac_power_with_pinctrl,
        cache_seconds: float = _CACHE_SECONDS,
        retry_after_failure: float = _RETRY_AFTER_FAILURE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bus_number = bus
        self._address = address
        self._bus_factory = bus_factory
        self._ac_reader = ac_reader
        self._cache_seconds = cache_seconds
        self._retry_after_failure = retry_after_failure
        self._clock = clock
        self._lock = threading.Lock()
        self._bus = None
        self._cached: Optional[BatteryStatus] = None
        self._next_attempt = 0.0

    def read(self) -> Optional[BatteryStatus]:
        with self._lock:
            now = self._clock()
            if now < self._next_attempt:
                return self._cached
            try:
                status = self._read_device()
            except Exception as exc:
                self._close_bus()
                self._cached = None
                self._next_attempt = now + self._retry_after_failure
                logger.info("Bateria indisponível (%s: %s)", type(exc).__name__, exc)
                return None
            self._cached = status
            self._next_attempt = now + self._cache_seconds
            return status

    def _read_device(self) -> BatteryStatus:
        if self._bus is None:
            self._bus = self._bus_factory(self._bus_number)
        voltage_word = self._bus.read_word_data(self._address, REG_VOLTAGE)
        capacity_word = self._bus.read_word_data(self._address, REG_CAPACITY)
        return BatteryStatus(
            percent=round(decode_percent(capacity_word), 1),
            voltage=round(decode_voltage(voltage_word), 3),
            ac_power=self._ac_reader(),
        )

    def _close_bus(self) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass
