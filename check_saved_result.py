"""Пример проверки допуска по сохранённому результату, без повторного решения."""
import json
import math
from pathlib import Path

result = json.loads((Path(__file__).parent / "heat-results.json").read_text(encoding="utf-8"))
history = result["history"]
errors = [row["max_error"] for row in history]
if not errors or any(not math.isfinite(e) or e <= 0 for e in errors):
    raise AssertionError("Некорректная история погрешности")
if any(fine >= coarse for coarse, fine in zip(errors, errors[1:])):
    raise AssertionError("Погрешность не уменьшается при уточнении шага")
if errors[-1] >= 0.001:
    raise AssertionError("Не выполнен допуск по времени")
if max(abs(v) for v in result["boundary_residuals"].values()) > 1e-9:
    raise AssertionError("Граничные невязки выше допуска")
print(f"Временная погрешность: {errors[-1]:.9f} < 0.001")
print(f"Наблюдаемый порядок: {history[-1]['observed_order']:.4f}")
print("Граничные условия и убывание погрешности прошли проверку.")
