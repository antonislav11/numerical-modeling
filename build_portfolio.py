"""Расчёт температурного поля, графики и проверка точности."""
import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from heat_model import Parameters, boundary_residuals, informative_theta_limit, runge_control

ROOT = Path(__file__).resolve().parent


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=160, facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "model-parameters.json")
    parser.add_argument("--output", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    p, solution, history = runge_control(Parameters(**config["parameters"]),
                                         tolerance=config["tolerance"])
    residuals = boundary_residuals(solution, p)
    if max(residuals.values()) > 1e-8:
        raise AssertionError(f"Граничные условия не выполнены: {residuals}")
    if history[-1]["max_error"] > config["tolerance"]:
        raise AssertionError("Не достигнут допуск по времени")
    if not np.isfinite(solution.temperature).all():
        raise AssertionError("Поле содержит нечисловые значения")
    reference = json.loads((ROOT / "heat-results.json").read_text())
    reference_error = None
    if asdict(p) == reference["parameters"]:
        reference_error = float(np.max(np.abs(solution.temperature[0] - reference["axis_temperature"])))
        if reference_error > 1e-9:
            raise AssertionError(f"Расчёт исходного набора не совпал с сохранённым: {reference_error}")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"parameters": asdict(p), "history": history, "boundary_residuals": residuals,
              "tolerance": config["tolerance"], "reference_axis_max_error": reference_error,
              "temperature_min": float(solution.temperature.min()),
              "temperature_max": float(solution.temperature.max())}
    (args.output / "model-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "temperature-field.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["radius", "theta", "temperature"])
        for i, r in enumerate(solution.radius):
            for j, theta in enumerate(solution.theta):
                writer.writerow([r, theta, solution.temperature[i, j]])
    with (args.output / "convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "heat-portfolio", "svg.fonttype": "none"})
    theta_limit = informative_theta_limit(solution)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3), layout="constrained")
    fig.suptitle("Теплообмен: поле, профиль и проверка точности", fontsize=18, weight="bold")
    image = axes[0].pcolormesh(solution.theta, solution.radius, solution.temperature,
                              shading="auto", cmap="Blues")
    axes[0].set(xlim=(0, theta_limit), xlabel="θ, рад", ylabel="r / R", title="Температурное поле")
    fig.colorbar(image, ax=axes[0], label="Безразмерная температура")
    axes[1].plot(solution.theta, solution.temperature[0], color="#141414", lw=2)
    axes[1].set(xlim=(0, theta_limit), xlabel="θ, рад", ylabel="Безразмерная температура", title="Профиль на оси трубы")
    axes[1].grid(alpha=.2)
    axes[2].loglog([r["fine_step"] for r in history], [r["max_error"] for r in history], "o-", color="#2965a3")
    axes[2].axhline(config["tolerance"], color="#141414", linestyle="--", label="Допуск")
    axes[2].set(xlabel="Шаг по времени", ylabel="Оценка Рунге", title="Временная сходимость")
    axes[2].grid(alpha=.2)
    axes[2].legend()
    save(fig, args.output, "heat-overview")
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.plot(solution.theta, solution.temperature[0], color="#141414", lw=2)
    ax.set(xlim=(0, theta_limit), xlabel="θ, рад", ylabel="Безразмерная температура",
           title=f"Профиль на оси трубы · τ = {p.final_time:g}")
    ax.grid(alpha=.2)
    save(fig, args.output, "heat-axis")
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.loglog([r["fine_step"] for r in history], [r["max_error"] for r in history], "o-", color="#2965a3")
    ax.axhline(config["tolerance"], color="#141414", linestyle="--", label="Допуск")
    ax.set(xlabel="Шаг по времени", ylabel="Оценка Рунге", title="Контроль временной погрешности")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, args.output, "heat-convergence")
    report = "# Последняя сборка численной модели\n\n"
    report += "Температурное поле рассчитываю в `heat_model.py`; здесь показываю профиль температуры и контроль временной погрешности.\n\n"
    report += f"Сетка: **{p.intervals+1} × {p.intervals+1}**; конечное время: **{p.final_time:g}**.\n\n"
    report += f"Принятый шаг: **{p.time_step:.8g}**; оценка Рунге: **{history[-1]['max_error']:.8g}**; допуск: **{config['tolerance']:g}**.\n\n"
    report += f"Максимальная граничная невязка: **{max(residuals.values()):.3g}**.\n\n"
    report += "![Поле, профиль и сходимость](heat-overview.png)\n\n"
    report += "[PNG](heat-overview.png) · [SVG](heat-overview.svg)\n\n"
    report += "![Профиль температуры](heat-axis.png)\n\n[SVG профиля](heat-axis.svg)\n\n"
    report += "![Временная сходимость](heat-convergence.png)\n\n[SVG сходимости](heat-convergence.svg)\n\n"
    report += "[Полное поле CSV](temperature-field.csv) · [История сходимости](convergence.csv) · [Результаты и параметры](model-results.json) · [Разбор проекта](../README.md)\n\n"
    report += "Проверка относится к временному шагу при фиксированной пространственной сетке. Адекватность физической модели и пространственную сходимость нужно оценивать отдельно.\n"
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    files = ["model-results.json", "temperature-field.csv", "convergence.csv", "REPORT.md"]
    files += [f"{name}.{ext}" for name in ("heat-overview", "heat-axis", "heat-convergence") for ext in ("png", "svg")]
    manifest = {"checks": "passed", "chart_count": 3,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("heat_model.py", "build_portfolio.py", "requirements-rebuild.txt")},
                "files": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in files}}
    (args.output / "build-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Проверено: шаг {p.time_step:g}; погрешность {history[-1]['max_error']:.8g}; 3 графика PNG/SVG.")


if __name__ == "__main__":
    main()
