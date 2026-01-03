"""Logger sencillo (opcional).

Para un MVP, usamos prints controlados. Si luego quieres logging a archivo,
puedes expandir esta clase.
"""

from dataclasses import dataclass

@dataclass
class Logger:
    enabled: bool = True

    def info(self, msg: str):
        if self.enabled:
            print(f"[INFO] {msg}")

    def warn(self, msg: str):
        if self.enabled:
            print(f"[WARN] {msg}")

    def error(self, msg: str):
        if self.enabled:
            print(f"[ERROR] {msg}")
