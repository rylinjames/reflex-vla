#!/usr/bin/env python3
"""Patch LIBERO's __init__.py to skip interactive prompts in containers.

LIBERO's libero/libero/__init__.py has 3 input() calls that prompt the user
for a dataset directory. In non-interactive environments (Docker, Modal),
these cause EOFError. This script replaces them with defaults.
"""
import pathlib
import sys

target = pathlib.Path("/opt/LIBERO/libero/libero/__init__.py")
if not target.exists():
    print(f"ERROR: {target} not found")
    sys.exit(1)

lines = target.read_text().splitlines()
patched = 0

new_lines = []
for i, line in enumerate(lines):
    # Line ~50: answer = input("Do you want to specify...").lower()
    if "input(" in line and "Do you want to specify" in line:
        new_lines.append(line.replace('input("Do you want to specify a custom path for the dataset folder? (Y/N): ").lower()', '"n"'))
        patched += 1
    # Line ~60: custom_dataset_path = input("Enter the path...")
    elif "input(" in line and "Enter the path" in line:
        new_lines.append(line.replace('input("Enter the path where you want to store the datasets: ")', '"/tmp/libero_data"'))
        patched += 1
    # Line ~68: confirm_answer = input().lower()
    elif "input()" in line and "confirm" in line.lower():
        new_lines.append(line.replace("input().lower()", '"y"'))
        patched += 1
    # Catch any other bare input() calls
    elif line.strip().startswith("input(") or "= input(" in line:
        new_lines.append(line.replace("input()", '"n"'))
        patched += 1
    else:
        new_lines.append(line)

target.write_text("\n".join(new_lines) + "\n")
print(f"Patched {patched} input() calls in {target}")
