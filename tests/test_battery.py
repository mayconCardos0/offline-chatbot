"""
Tests for core/battery.py e GET /battery — leitura da bateria do UPS HAT.
"""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.routes import router  # noqa: E402
from core.battery import (  # noqa: E402
    BatteryMonitor,
    BatteryStatus,
    decode_percent,
    decode_voltage,
    read_ac_power_with_pinctrl,
)
from core.config import Settings  # noqa: E402


class FakeBus:
    """Simula o smbus2: read_word_data devolve a word com o 1º byte lido no byte baixo."""

    def __init__(self, voltage_word=0x00D2, capacity_word=0x0032, fail=False):
        self.words = {0x02: voltage_word, 0x04: capacity_word}
        self.fail = fail
        self.closed = False
        self.reads = 0

    def read_word_data(self, address, register):
        self.reads += 1
        if self.fail:
            raise OSError("Remote I/O error")
        assert address == 0x36
        return self.words[register]

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class TestDecode:
    def test_voltage_of_a_full_cell(self):
        # registrador 0xD200 (big-endian) → o smbus devolve 0x00D2
        assert decode_voltage(0x00D2) == pytest.approx(4.2)

    def test_voltage_of_a_low_cell(self):
        assert decode_voltage(0x00A0) == pytest.approx(3.2)

    def test_percent_half(self):
        assert decode_percent(0x0032) == pytest.approx(50.0)

    def test_percent_full(self):
        assert decode_percent(0x0064) == pytest.approx(100.0)

    def test_percent_is_clamped(self):
        assert decode_percent(0x0065) == 100.0  # 101,x %
        assert decode_percent(0x0000) == 0.0

    def test_percent_keeps_the_fraction(self):
        # 0x3280 → 50,5 %
        assert decode_percent(0x8032) == pytest.approx(50.5)


class TestBatteryStatus:
    def test_charging_when_on_mains_and_not_full(self):
        assert BatteryStatus(80, 4.0, True).charging is True

    def test_not_charging_when_full(self):
        assert BatteryStatus(100, 4.2, True).charging is False

    def test_not_charging_on_battery(self):
        assert BatteryStatus(80, 4.0, False).charging is False

    def test_unknown_when_mains_state_is_unknown(self):
        assert BatteryStatus(80, 4.0, None).charging is None


class TestBatteryMonitor:
    def _monitor(self, bus, clock=None, ac=True, **kwargs):
        return BatteryMonitor(
            bus_factory=lambda number: bus,
            ac_reader=lambda: ac,
            clock=clock or Clock(),
            **kwargs,
        )

    def test_reads_percent_voltage_and_mains(self):
        status = self._monitor(FakeBus()).read()
        assert status == BatteryStatus(percent=50.0, voltage=4.2, ac_power=True)

    def test_reading_is_cached_for_a_few_seconds(self):
        bus, clock = FakeBus(), Clock()
        monitor = self._monitor(bus, clock)
        monitor.read()
        reads_after_first = bus.reads
        clock.now += 1.0
        monitor.read()
        assert bus.reads == reads_after_first
        clock.now += 5.0
        monitor.read()
        assert bus.reads == reads_after_first * 2

    def test_unavailable_hardware_returns_none_without_raising(self):
        def no_device(number):
            raise FileNotFoundError("/dev/i2c-1")

        monitor = BatteryMonitor(bus_factory=no_device, ac_reader=lambda: True)
        assert monitor.read() is None

    def test_missing_smbus2_returns_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "smbus2", None)  # import falha
        monitor = BatteryMonitor(ac_reader=lambda: True)
        assert monitor.read() is None

    def test_failure_is_not_retried_until_the_backoff_passes(self):
        clock, calls = Clock(), []

        def flaky(number):
            calls.append(number)
            raise OSError("sem resposta")

        monitor = BatteryMonitor(
            bus_factory=flaky,
            ac_reader=lambda: True,
            clock=clock,
            retry_after_failure=30,
        )
        assert monitor.read() is None
        clock.now += 10
        assert monitor.read() is None
        assert len(calls) == 1
        clock.now += 30
        monitor.read()
        assert len(calls) == 2

    def test_recovers_after_a_failure(self):
        clock = Clock()
        bus = FakeBus(fail=True)
        monitor = self._monitor(bus, clock)
        assert monitor.read() is None
        assert bus.closed is True  # fecha o barramento quebrado
        bus.fail, bus.closed = False, False
        clock.now += 31
        assert monitor.read() is not None

    def test_unknown_mains_state_still_reports_the_battery(self):
        status = self._monitor(FakeBus(), ac=None).read()
        assert status.percent == 50.0
        assert status.ac_power is None


class TestReadAcPower:
    def _run(self, monkeypatch, stdout=None, error=None):
        def fake_run(*args, **kwargs):
            if error:
                raise error
            return SimpleNamespace(stdout=stdout)

        monkeypatch.setattr(subprocess, "run", fake_run)

    def test_high_means_adapter_connected(self, monkeypatch):
        self._run(monkeypatch, "6: ip    pu | hi // GPIO6 = input\n")
        assert read_ac_power_with_pinctrl() is True

    def test_low_means_power_loss(self, monkeypatch):
        self._run(monkeypatch, "6: ip    pu | lo // GPIO6 = input\n")
        assert read_ac_power_with_pinctrl() is False

    def test_unparseable_output_is_unknown(self, monkeypatch):
        self._run(monkeypatch, "erro inesperado")
        assert read_ac_power_with_pinctrl() is None

    def test_missing_pinctrl_is_unknown(self, monkeypatch):
        self._run(monkeypatch, error=FileNotFoundError("pinctrl"))
        assert read_ac_power_with_pinctrl() is None

    def test_timeout_is_unknown(self, monkeypatch):
        self._run(monkeypatch, error=subprocess.TimeoutExpired("pinctrl", 2))
        assert read_ac_power_with_pinctrl() is None


def _app(battery=...):
    app = FastAPI()
    app.include_router(router)

    @app.on_event("startup")
    async def set_state():
        if battery is not ...:
            app.state.battery = battery

    return app


class TestBatteryEndpoint:
    def test_reports_the_battery_when_available(self):
        monitor = MagicMock()
        monitor.read.return_value = BatteryStatus(
            percent=73.5, voltage=3.91, ac_power=True
        )
        with TestClient(_app(monitor)) as client:
            body = client.get("/battery").json()
        assert body == {
            "available": True,
            "percent": 73.5,
            "voltage": 3.91,
            "ac_power": True,
            "charging": True,
        }

    def test_on_battery_is_not_charging(self):
        monitor = MagicMock()
        monitor.read.return_value = BatteryStatus(
            percent=40.0, voltage=3.7, ac_power=False
        )
        with TestClient(_app(monitor)) as client:
            body = client.get("/battery").json()
        assert body["ac_power"] is False
        assert body["charging"] is False

    def test_unavailable_when_the_ups_does_not_answer(self):
        monitor = MagicMock()
        monitor.read.return_value = None
        with TestClient(_app(monitor)) as client:
            body = client.get("/battery").json()
        assert body["available"] is False
        assert body["percent"] is None

    def test_unavailable_when_disabled(self):
        with TestClient(_app(None)) as client:
            assert client.get("/battery").json()["available"] is False

    def test_unavailable_when_not_configured(self):
        with TestClient(_app()) as client:
            assert client.get("/battery").json()["available"] is False


class TestBatterySetting:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("BATTERY_ENABLED", raising=False)
        assert Settings().battery_enabled is True

    def test_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("BATTERY_ENABLED", "false")
        assert Settings().battery_enabled is False
