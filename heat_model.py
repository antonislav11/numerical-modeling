from __future__ import annotations
 
import argparse
import csv
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
 
import numpy as np
@dataclass(frozen=True)
class Parameters:
 
    heat_transfer_coefficient: float = 0.1
    pipe_radius: float = 5.0
    thermal_conductivity: float = 0.683
    bend_radius: float = 1.0
    density: float = 1000.0
    heat_capacity: float = 4200.0
    velocity: float = 0.001
 
    volumetric_heat_source: float = 0.0
    temperature_difference: float = 1.0
 
    intervals: int = 100
    time_step: float = 0.001
    final_time: float = 0.1
 
    @property
    def thermal_diffusivity(self) -> float:
 
        return self.thermal_conductivity / (self.density * self.heat_capacity)
 
    @property
    def curvature_ratio(self) -> float:
 
        return self.bend_radius / self.pipe_radius
 
    @property
    def fourier_parameter(self) -> float:
 
        return self.thermal_diffusivity / (self.pipe_radius * self.velocity)
 
    @property
    def biot_number(self) -> float:
 
        return (
            self.heat_transfer_coefficient
            * self.pipe_radius
            / self.thermal_conductivity
        )
 
    @property
    def source_parameter(self) -> float:
 
        return (
            self.volumetric_heat_source
            * self.pipe_radius
            / (
                self.density
                * self.heat_capacity
                * self.velocity
                * self.temperature_difference
            )
        )
@dataclass
class Solution:
    radius: np.ndarray
    theta: np.ndarray
    temperature: np.ndarray
    snapshots: dict[float, np.ndarray]
def validate_parameters(p: Parameters, check_stability: bool = True) -> None:
 
    if not isinstance(p.intervals, int) or p.intervals < 3:
        raise ValueError("Число интервалов N должно быть не меньше 3.")
    if not all(np.isfinite(value) for value in asdict(p).values()):
        raise ValueError("Все параметры должны быть конечными числами.")
    if p.pipe_radius <= 0 or p.bend_radius <= 0:
        raise ValueError("Радиусы трубы и изгиба должны быть положительными.")
    if p.velocity <= 0 or p.time_step <= 0 or p.final_time <= 0:
        raise ValueError(
            "Скорость, шаг и конечное время должны быть положительными."
        )
    if p.temperature_difference == 0:
        raise ValueError("Для обезразмеривания требуется T1 - T0 != 0.")
    if p.density <= 0 or p.heat_capacity <= 0:
        raise ValueError(
            "Плотность и теплоёмкость должны быть положительными."
        )
    if p.thermal_conductivity <= 0 or p.heat_transfer_coefficient < 0:
        raise ValueError(
            "Теплопроводность должна быть положительной, теплоотдача — "
            "неотрицательной."
        )
    if not check_stability:
        return
 
    dr = 1.0 / p.intervals
    dtheta = np.pi / p.intervals
    smallest_radius = dr
    smallest_metric = p.curvature_ratio + smallest_radius
    radial_coefficient = (
        p.curvature_ratio + 2.0 * smallest_radius
    ) / (smallest_radius * smallest_metric)
 
    cfl = (
        p.time_step / (smallest_metric * dtheta)
        + 2.0 * p.fourier_parameter * p.time_step / dr**2
        + 2.0
        * p.fourier_parameter
        * p.time_step
        / (smallest_metric**2 * dtheta**2)
        + p.fourier_parameter * p.time_step * radial_coefficient / dr
    )
    if cfl > 1.0:
        raise ValueError(
            "Явная схема может быть неустойчивой: оценка CFL = "
            f"{cfl:.3f} > 1. Уменьшите time_step."
        )
def solve(
    p: Parameters,
    snapshot_times: tuple[float, ...] = (0.0, 0.025, 0.05, 0.075, 0.1),
) -> Solution:
 
    validate_parameters(p)
 
    n = p.intervals
    dr = 1.0 / n
    dtheta = np.pi / n
    radius = np.linspace(0.0, 1.0, n + 1)
    theta = np.linspace(0.0, np.pi, n + 1)
 
    temperature = np.zeros((n + 1, n + 1), dtype=float)
    temperature[:, 0] = 1.0
 
    number_of_steps = int(round(p.final_time / p.time_step))
    if number_of_steps < 1 or not np.isclose(
        p.final_time / p.time_step, number_of_steps, rtol=0.0, atol=1e-8
    ):
        raise ValueError("final_time должен быть кратен time_step.")
 
    requested_steps = {
        int(round(time / p.time_step)): time
        for time in snapshot_times
        if 0.0 <= time <= p.final_time
    }
    snapshots: dict[float, np.ndarray] = {}
    if 0 in requested_steps:
        snapshots[0.0] = temperature.copy()
 
    internal_radius = radius[1:-1, np.newaxis]
    metric = p.curvature_ratio + internal_radius
    radial_coefficient = (
        p.curvature_ratio + 2.0 * internal_radius
    ) / (internal_radius * metric)
 
    for step in range(1, number_of_steps + 1):
        center = temperature[1:-1, 1:-1]
        radial_plus = temperature[2:, 1:-1]
        radial_minus = temperature[:-2, 1:-1]
        theta_plus = temperature[1:-1, 2:]
        theta_minus = temperature[1:-1, :-2]
 
        d_temperature_dtheta = (center - theta_minus) / dtheta
        d_temperature_dr = (radial_plus - center) / dr
        d2_temperature_dr2 = (
            radial_plus - 2.0 * center + radial_minus
        ) / dr**2
        d2_temperature_dtheta2 = (
            theta_plus - 2.0 * center + theta_minus
        ) / dtheta**2
 
        right_hand_side = (
            -d_temperature_dtheta / metric
            + p.fourier_parameter
            * (
                radial_coefficient * d_temperature_dr
                + d2_temperature_dr2
                + d2_temperature_dtheta2 / metric**2
            )
            + p.source_parameter
        )
 
        new_temperature = np.zeros_like(temperature)
        new_temperature[1:-1, 1:-1] = center + p.time_step * right_hand_side
 
        new_temperature[:, 0] = 1.0
        new_temperature[:, -1] = 0.0
        new_temperature[0, 1:-1] = new_temperature[1, 1:-1]
        new_temperature[-1, 1:-1] = (
            new_temperature[-2, 1:-1] / (1.0 + p.biot_number * dr)
        )
 
        if not np.all(np.isfinite(new_temperature)):
            raise FloatingPointError(f"Расчёт разошёлся на шаге {step}.")
 
        temperature = new_temperature
        if step in requested_steps:
            actual_time = step * p.time_step
            snapshots[actual_time] = temperature.copy()
 
    if p.final_time not in snapshots:
        snapshots[p.final_time] = temperature.copy()
 
    return Solution(radius, theta, temperature, snapshots)
def stability_limit(p: Parameters) -> float:
    validate_parameters(p, check_stability=False)
    dr = 1.0 / p.intervals
    dtheta = np.pi / p.intervals
    radius = np.arange(1, p.intervals, dtype=float) * dr
    metric = p.curvature_ratio + radius
    radial = (p.curvature_ratio + 2 * radius) / (radius * metric)
    rates = 1 / (metric * dtheta) + p.fourier_parameter * (
        radial / dr + 2 / dr**2 + 2 / (metric * dtheta)**2
    )
    return float(1 / np.max(rates))
def runge_control(
    p: Parameters, tolerance: float = 1e-3, max_halvings: int = 12
):
    validate_parameters(p, check_stability=False)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Допуск должен быть положительным конечным числом.")
    if not isinstance(max_halvings, int) or max_halvings < 0:
        raise ValueError(
            "Число уменьшений шага должно быть неотрицательным целым."
        )
    count = p.final_time / p.time_step
    n = int(round(count))
    if n < 1 or not np.isclose(count, n, rtol=0, atol=1e-8):
        raise ValueError("Конечное время должно быть кратно исходному шагу.")
    if n % 2:
        n *= 2
    limit = stability_limit(p)
    while 2 * p.final_time / n > limit:
        n *= 2
    while n % 4 == 0 and 4 * p.final_time / n <= limit:
        n //= 2
    coarse = solve(
        replace(p, time_step=p.final_time / (n // 2)),
        snapshot_times=(),
    )
    history = []
    previous_error = None
    for _ in range(max_halvings + 1):
        fine_p = replace(p, time_step=p.final_time / n)
        fine = solve(fine_p, snapshot_times=())
 
        error = float(np.max(np.abs(fine.temperature - coarse.temperature)))
        order = None
        if previous_error is not None and previous_error > 0 and error > 0:
            order = float(np.log2(previous_error / error))
        history.append({
            "coarse_step": 2 * fine_p.time_step,
            "fine_step": fine_p.time_step,
            "max_error": error,
            "observed_order": order,
        })
        if error <= tolerance:
            return fine_p, fine, history
        previous_error = error
        coarse = fine
        n *= 2
    raise RuntimeError(
        "Заданный допуск не достигнут за разрешённое число уменьшений шага."
    )
def boundary_residuals(solution: Solution, p: Parameters) -> dict[str, float]:
    t = solution.temperature
    return {
        "inlet": float(np.max(np.abs(t[:, 0] - 1))),
        "outlet": float(np.max(np.abs(t[:, -1]))),
        "axis": float(np.max(np.abs(t[0, 1:-1] - t[1, 1:-1]))),
        "wall": float(np.max(np.abs(
            (t[-1, 1:-1] - t[-2, 1:-1]) * p.intervals
            + p.biot_number * t[-1, 1:-1]
        ))),
    }
def save_axis_table(solution: Solution, output_directory: Path) -> Path:
 
    path = output_directory / "k_dtheta_T1_0_k.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["k", "k_dtheta_rad", "k_dtheta_deg", "T1_0_k"])
        for k, (theta, value) in enumerate(
            zip(solution.theta, solution.temperature[0, :], strict=True)
        ):
            writer.writerow(
                [
                    k, f"{theta:.10f}",
                    f"{np.degrees(theta):.6f}", f"{value:.10f}",
                ]
            )
    return path
def informative_theta_limit(
    solution: Solution, relative_tolerance: float = 1e-3
) -> float:
 
    profile = np.abs(solution.temperature[0, :])
    scale = float(np.max(profile))
    if scale == 0.0:
        return float(solution.theta[-1])
 
    nonzero_indices = np.flatnonzero(profile > relative_tolerance * scale)
    if nonzero_indices.size == 0:
        return float(solution.theta[-1])
 
    last_index = min(int(nonzero_indices[-1]) + 2, solution.theta.size - 1)
 
    return min(
        float(solution.theta[-1]),
        float(np.ceil(solution.theta[last_index] * 10) / 10),
    )
def make_plots(
    solution: Solution,
    p: Parameters,
    output_directory: Path,
) -> list[Path]:
 
    import matplotlib.pyplot as plt
 
    created_files: list[Path] = []
 
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        solution.theta,
        solution.temperature[0, :],
        linewidth=1.6,
        label=fr"$\tau={p.final_time:g}$",
    )
    ax.set(
        xlabel=r"$\theta$, рад",
        ylabel=r"$\overline{T}(0,\theta)$",
        title=(
            r"Зависимость температуры $\overline{T}$ "
            r"от угла $\theta$ на оси трубы"
        ),
        xlim=(0.0, informative_theta_limit(solution)),
        ylim=(0.0, 1.03) if p.source_parameter == 0 else (None, None),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_directory / "T1_0_k_vs_k_dtheta.png"
    fig.savefig(path, dpi=180)
    created_files.append(path)
 
    return created_files
def make_convergence_plots(
    history, tolerance: float, output_directory: Path
) -> Path:
    import matplotlib.pyplot as plt
 
    rows = sorted(history, key=lambda row: row["fine_step"])
    steps = np.array([row["fine_step"] for row in rows])
    errors = np.array([row["max_error"] for row in rows])
    fig, (error_ax, order_ax) = plt.subplots(2, 1, figsize=(8.4, 7.8))
 
    positive = errors > 0
    if np.any(positive):
        error_ax.loglog(
            steps[positive], errors[positive], "o-",
            label="Оценка Рунге",
        )
        first = np.flatnonzero(positive)[0]
        error_ax.loglog(steps, errors[first] * steps / steps[first], "--",
                       color="0.45", label=r"$C h$ (первый порядок)")
    else:
        error_ax.text(0.5, 0.5, "Разность решений равна нулю",
                      transform=error_ax.transAxes, ha="center")
        error_ax.set_xscale("log")
    error_ax.axhline(tolerance, color="#a33b27", linestyle=":",
                     label=fr"$\varepsilon_{{\mathrm{{зад}}}}={tolerance:g}$")
    error_ax.set(xlabel=r"Временной шаг $h=\Delta\tau$",
                 ylabel=r"$\varepsilon_{\max}(h)$",
                 title="а) Погрешность по времени")
    error_ax.legend(fontsize=9)
    order_rows = [row for row in rows if row["observed_order"] is not None]
    if order_rows:
        order_ax.semilogx([row["fine_step"] for row in order_rows],
                          [row["observed_order"] for row in order_rows], "o-",
                          label="Наблюдаемый порядок")
    else:
        order_ax.set_xscale("log")
        order_ax.text(0.5, 0.65, "Недостаточно сравнений для оценки порядка",
                      transform=order_ax.transAxes, ha="center")
    order_ax.axhline(
        1, color="0.45", linestyle="--",
        label="Теоретический порядок 1",
    )
    order_ax.set(xlabel=r"Временной шаг $h=\Delta\tau$",
                 ylabel=r"$p_{\mathrm{набл}}(h)$",
                 title="б) Порядок сходимости по времени")
    order_ax.legend(fontsize=9)
    for ax in (error_ax, order_ax):
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout(h_pad=2)
    path = output_directory / "time_convergence.png"
    fig.savefig(path, dpi=200)
    return path
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Расчёт теплообмена в изогнутой трубе по явной схеме."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_final",
        help="Каталог итоговых результатов (по умолчанию: results_final).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Сохранить графики, не открывая окно matplotlib.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-3,
        help=(
            "Допуск временной погрешности по правилу Рунге "
            "(по умолчанию: 0.001)."
        ),
    )
    return parser.parse_args()
def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
 
    if arguments.no_show:
        import matplotlib
 
        matplotlib.use("Agg")
 
    initial_parameters = Parameters()
    parameters, solution, history = runge_control(
        initial_parameters, tolerance=arguments.tolerance
    )
    table = save_axis_table(solution, arguments.output_dir)
    plots = make_plots(
        solution,
        parameters,
        arguments.output_dir,
    )
    plots.append(
        make_convergence_plots(
            history, arguments.tolerance, arguments.output_dir
        )
    )
    runge_path = arguments.output_dir / "runge_check.csv"
    with runge_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(history[0]), delimiter=";"
        )
        writer.writeheader()
        writer.writerows(history)
    residuals = boundary_residuals(solution, parameters)
    summary = {
        "initial_parameters": asdict(initial_parameters),
        "accepted_parameters": asdict(parameters),
        "tolerance": arguments.tolerance,
        "history": history,
        "stability_limit": stability_limit(parameters),
        "accepted_cfl": parameters.time_step / stability_limit(parameters),
        "time_steps": int(
            round(parameters.final_time / parameters.time_step)
        ),
        "dimensionless_temperature_min": float(solution.temperature.min()),
        "dimensionless_temperature_max": float(solution.temperature.max()),
        "boundary_residuals": residuals,
        "plot_theta_limit": informative_theta_limit(solution),
        "axis_monotone": bool(
            np.all(np.diff(solution.temperature[0]) <= 1e-12)
        ),
    }
    summary_path = arguments.output_dir / "calculation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
 
    print("Расчёт завершён.")
    print(f"a  = {parameters.thermal_diffusivity:.6e} м²/с")
    print(f"R0 = {parameters.curvature_ratio:.6g}")
    print(f"A  = {parameters.fourier_parameter:.6e}")
    print(f"B  = {parameters.biot_number:.6g}")
    print(f"Q  = {parameters.source_parameter:.6g}")
    print(
        f"Шаг по времени: {parameters.time_step:.8g}; "
        f"число шагов: {summary['time_steps']}"
    )
    print(
        f"Оценка Рунге: {history[-1]['max_error']:.8g}; "
        f"допуск: {arguments.tolerance:g}"
    )
    print(f"Наблюдаемый порядок: {history[-1]['observed_order']}")
    print(f"Невязки граничных условий: {residuals}")
    print(
        "Диапазон температуры: "
        f"{solution.temperature.min():.6f} <= T_bar <= "
        f"{solution.temperature.max():.6f}"
    )
    print("Созданы файлы:")
    for path in [table, runge_path, summary_path, *plots]:
        print(f"  {path}")
 
    if not arguments.no_show:
        import matplotlib.pyplot as plt
 
        plt.show()
if __name__ == "__main__":
    main()