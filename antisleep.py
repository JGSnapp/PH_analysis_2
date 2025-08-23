
"""
antisleep.py — tiny cross‑platform "don't sleep" context manager.
- Windows: uses SetThreadExecutionState to prevent system/display sleep.
- macOS: runs `caffeinate -dimsu` while in context.
- Linux (systemd): runs `systemd-inhibit --what=shutdown:sleep:idle:handle-lid-switch`.
No external deps; safe to vendor.
"""

from __future__ import annotations
import platform
import subprocess
import ctypes
from typing import Optional

class _KeepAwake:
    def __init__(self, reason: str = "Long-running task is active", logger=None) -> None:
        self.reason = reason
        self.logger = logger
        self.os = platform.system()
        self._proc: Optional[subprocess.Popen] = None

        # Windows constants
        self._ES_CONTINUOUS = 0x80000000
        self._ES_SYSTEM_REQUIRED = 0x00000001
        self._ES_DISPLAY_REQUIRED = 0x00000002
        self._ES_AWAYMODE_REQUIRED = 0x00000040

    # --------- logging helper ---------
    def _log(self, msg: str) -> None:
        if self.logger:
            try:
                self.logger.info(msg)
                return
            except Exception:
                pass
        # fallback
        print(msg)

    # --------- platform impls ---------
    def _enter_windows(self) -> None:
        kernel32 = ctypes.windll.kernel32
        # prevent system + display sleep (away mode allows work during "sleep" on desktops)
        ok = kernel32.SetThreadExecutionState(
            self._ES_CONTINUOUS
            | self._ES_SYSTEM_REQUIRED
            | self._ES_DISPLAY_REQUIRED
            | self._ES_AWAYMODE_REQUIRED
        )
        if not ok:
            raise OSError("SetThreadExecutionState failed")
        self._kernel32 = kernel32
        self._log("antisleep: Windows execution state set (system+display required).")

    def _exit_windows(self) -> None:
        try:
            self._kernel32.SetThreadExecutionState(self._ES_CONTINUOUS)
            self._log("antisleep: Windows execution state restored.")
        except Exception:
            pass

    def _enter_macos(self) -> None:
        # -d keep display on, -i prevent idle sleep, -m prevent disk sleep, -s system sleep, -u declare user active
        self._proc = subprocess.Popen(["caffeinate", "-dimsu"])
        self._log("antisleep: macOS caffeinate started.")

    def _enter_linux(self) -> None:
        # Holds an inhibitor lock until the child exits.
        # Works on systemd-based distros; otherwise it's a no-op if binary missing.
        try:
            self._proc = subprocess.Popen([
                "systemd-inhibit",
                "--what=shutdown:sleep:idle:handle-lid-switch",
                f"--why={self.reason}",
                "--mode=block",
                "bash", "-c", "while :; do sleep 3600; done"
            ])
            self._log("antisleep: Linux systemd-inhibit lock acquired.")
        except Exception as e:
            self._log(f"antisleep: systemd-inhibit not available ({e}); continuing without Linux inhibit.")

    def _terminate_proc(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            finally:
                self._proc = None

    # --------- context protocol ---------
    def __enter__(self) -> "_KeepAwake":
        try:
            if self.os == "Windows":
                self._enter_windows()
            elif self.os == "Darwin":
                self._enter_macos()
            else:
                self._enter_linux()
        except Exception as e:
            self._log(f"antisleep: init failed: {e}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.os == "Windows":
                self._exit_windows()
            elif self.os in ("Darwin", "Linux"):
                self._terminate_proc()
                self._log("antisleep: inhibitor process stopped.")
        except Exception as e:
            self._log(f"antisleep: release failed: {e}")

def keep_awake(reason: str = "Long-running task is active", logger=None) -> _KeepAwake:
    """
    Usage:
        from antisleep import keep_awake
        with keep_awake("Fetching Product Hunt data", logger=my_logger):
            ... do work ...
    """
    return _KeepAwake(reason=reason, logger=logger)
