#!/usr/bin/env python3
"""Quasi-static pantograph contact on two 3D wire profiles (SI units).

At each distance the strip is horizontal and perpendicular to travel.
An eligible wire exerts a vertical force k * max(0, head_z - wire_z).
Wire heights are unloaded; the springs represent effective compliance.
"""

import argparse
from bisect import bisect_right
import csv
import math
from pathlib import Path


WIRES = ("side", "main")
INPUT_COLUMNS = ("s_m", "side_y_m", "side_z_m", "main_y_m", "main_z_m")
RESULT_COLUMNS = ("time_s", "s_m", "head_z_m", "contact_state") + tuple(
    f"{wire}_{field}"
    for wire in WIRES
    for field in ("y_m", "unloaded_z_m", "contact_y_m", "contact_z_m", "force_n")
)


def read_profiles(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not set(INPUT_COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError("Expected columns: " + ", ".join(INPUT_COLUMNS))
        rows = []
        for row in reader:
            try:
                values = tuple(float(row[column]) for column in INPUT_COLUMNS)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid number on CSV line {reader.line_num}") from exc
            if not all(map(math.isfinite, values)):
                raise ValueError(f"Non-finite number on CSV line {reader.line_num}")
            rows.append(values)
    distances = [row[0] for row in rows]
    if len(rows) < 2 or any(a >= b for a, b in zip(distances, distances[1:])):
        raise ValueError("At least two rows with strictly increasing s_m are required")
    return distances, rows


def interpolate(distances, rows, distance):
    right = min(bisect_right(distances, distance), len(rows) - 1)
    left = right - 1
    fraction = (distance - distances[left]) / (distances[right] - distances[left])
    return tuple(
        a + fraction * (b - a) for a, b in zip(rows[left][1:], rows[right][1:])
    )


def equilibrium(heights, push_force, stiffness):
    # Start with the lowest eligible wire. Include the next wire only
    # if the first equilibrium would press the head above its height.
    if not heights:
        return None
    ordered = sorted(heights)
    head = ordered[0] + push_force / stiffness
    if len(ordered) == 2 and head > ordered[1]:
        head = ordered[0] + (push_force / stiffness + ordered[1] - ordered[0]) / 2
    return head


def simulate(distances, rows, speed_kmh, step_m, half_width_m, push_force, stiffness):
    start, end = distances[0], distances[-1]
    intervals = math.ceil((end - start) / step_m)
    for index in range(intervals + 1):
        distance = end if index == intervals else start + index * step_m
        side_y, side_z, main_y, main_z = interpolate(distances, rows, distance)
        positions = {"side": (side_y, side_z), "main": (main_y, main_z)}
        eligible = {wire: z for wire, (y, z) in positions.items() if abs(y) <= half_width_m}
        head = equilibrium(list(eligible.values()), push_force, stiffness)
        result = {
            "time_s": (distance - start) / (speed_kmh / 3.6),
            "s_m": distance,
            "head_z_m": head,
        }
        active = []
        for wire, (y, z) in positions.items():
            force = stiffness * max(0.0, head - z) if wire in eligible else 0.0
            contacting = force > 1e-8
            if contacting:
                active.append(wire)
            result.update({
                f"{wire}_y_m": y,
                f"{wire}_unloaded_z_m": z,
                f"{wire}_contact_y_m": y if contacting else None,
                f"{wire}_contact_z_m": head if contacting else None,
                f"{wire}_force_n": force if contacting else 0.0,
            })
        result["contact_state"] = "both" if len(active) == 2 else next(iter(active), "none")
        yield result


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive finite number")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Wire profile CSV")
    parser.add_argument("--output", type=Path, default=Path("contact_results.csv"))
    parser.add_argument("--speed-kmh", type=positive_float, default=30.0)
    parser.add_argument("--step-m", type=positive_float, default=0.05)
    parser.add_argument("--half-width-m", type=positive_float, default=0.65)
    parser.add_argument("--push-force-n", type=positive_float, default=70.0)
    parser.add_argument("--stiffness-n-per-m", type=positive_float, default=10000.0)
    args = parser.parse_args()

    previous_state = None
    try:
        if args.input.resolve() == args.output.resolve() or (
            args.output.exists() and args.input.samefile(args.output)
        ):
            raise ValueError("Input and output must be different files")
        distances, rows = read_profiles(args.input)
        results = simulate(
            distances, rows, args.speed_kmh, args.step_m, args.half_width_m,
            args.push_force_n, args.stiffness_n_per_m,
        )
        with args.output.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
                state = result["contact_state"]
                if previous_state != state:
                    print(f"{result['s_m']:.3f} m | {result['time_s']:.3f} s | {state}")
                    previous_state = state
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
