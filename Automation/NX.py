import json
import sys
import os
from pathlib import Path

# --- CONFIGURATION ---
INPUT_FOLDER = r"D:\Automtion\Merge_Output"
# Where to save the generated NX Python script
NX_SCRIPT_OUTPUT_FOLDER = r"D:\Automtion\Merge_Output_NX"
# Where the final NX CAD Part (.prt) will be saved
CAD_OUTPUT_FOLDER = r"D:\Automtion\CAD_Output"

INPUT_UNIT = "mm"


def parse_tokens_to_nx_script(tokens: list, input_filename: str) -> str:
    lines = []
    lines.append("import nx_easy as nx")
    lines.append("import os")
    lines.append("")

    lines.append(f"# Input data in {INPUT_UNIT}, converting to NX (mm)")
    if INPUT_UNIT == "mm":
        lines.append("SCALE = 1.0")
    elif INPUT_UNIT == "cm":
        lines.append("SCALE = 10.0")
    else:
        lines.append("SCALE = 1.0")

    lines.append("")
    lines.append("def main():")
    lines.append(f'    print("Generating NX model from: {input_filename}")')
    lines.append("")

    # Buffer now holds both circles and lines
    pending_sketch = {
        "active": False,
        "plane": "XY",
        "circles": [],
        "lines": []
    }

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # ---------------------------------------------------------
        # 1. Handle Sketch Start
        # ---------------------------------------------------------
        if token.startswith("plane="):
            plane_name_raw = token.split("=")[1]
            if i + 1 < len(tokens) and tokens[i + 1] == "ENTITY_START__Sketch":
                nx_plane = "XY"
                p_lower = plane_name_raw.lower()
                if p_lower in ["top", "xy"]:
                    nx_plane = "XY"
                elif p_lower in ["front", "yz"]:
                    nx_plane = "XZ"
                elif p_lower in ["right", "xz"]:
                    nx_plane = "YZ"

                pending_sketch = {
                    "active": True,
                    "plane": nx_plane,
                    "circles": [],
                    "lines": []
                }
                i += 1

        # ---------------------------------------------------------
        # 2. Handle Circles
        # ---------------------------------------------------------
        elif token == "CURVE_START__Circle":
            params = {}
            j = i + 1
            while j < len(tokens) and tokens[j] != "CURVE_END__Circle":
                if "=" in tokens[j]:
                    try:
                        key, val = tokens[j].split("=")
                        params[key] = float(val)
                    except ValueError:
                        pass
                j += 1

            if pending_sketch["active"]:
                pending_sketch["circles"].append(params)
            i = j

        # ---------------------------------------------------------
        # 3. Handle Lines (NEW)
        # ---------------------------------------------------------
        elif token == "CURVE_START__Line":
            params = {}
            j = i + 1
            while j < len(tokens) and tokens[j] != "CURVE_END__Line":
                if "=" in tokens[j]:
                    try:
                        key, val = tokens[j].split("=")
                        params[key] = float(val)
                    except ValueError:
                        pass
                j += 1

            if pending_sketch["active"]:
                pending_sketch["lines"].append(params)
            i = j

        # ---------------------------------------------------------
        # 4. Handle Extrusion
        # ---------------------------------------------------------
        elif token == "ENTITY_START__Extrude":
            params = {}
            j = i + 1
            while j < len(tokens) and tokens[j] != "ENTITY_END__Extrude":
                if "=" in tokens[j]:
                    key, val = tokens[j].split("=")
                    params[key] = val
                j += 1

            if pending_sketch["active"] and "distance" in params:
                try:
                    dist_val = float(params["distance"])
                    op_type = params.get("operation_type", "NewBody")

                    circles = pending_sketch["circles"]
                    lines_data = pending_sketch["lines"]
                    plane = pending_sketch["plane"]
                    is_creation = (op_type in ["NewBody", "Join"])

                    # --- LOGIC BRANCHING ---

                    # CASE A: We have LINES (L-Clamp, Rectangles, Profiles)
                    if lines_data:
                        lines.append(f'    # --- Profile Feature (Lines) ---')
                        lines.append(f'    nx.create_plane("{plane}")')

                        # Draw all lines
                        for l in lines_data:
                            lines.append(
                                f'    nx.sketch_line({l["start_x"]} * SCALE, {l["start_y"]} * SCALE, {l["end_x"]} * SCALE, {l["end_y"]} * SCALE)')

                        # Extrude the profile (Body)
                        nx_op = "Create" if is_creation else "Subtract"
                        lines.append(f'    nx.extrude({dist_val} * SCALE, operation="{nx_op}")')

                        # If there are ALSO circles in this sketch block, they are likely holes inside that profile
                        if circles:
                            lines.append(f'    # --- Holes inside Profile ---')
                            lines.append(f'    nx.create_plane("{plane}")')
                            for c in circles:
                                lines.append(
                                    f'    nx.sketch_circle({c["center_x"]} * SCALE, {c["center_y"]} * SCALE, {c["radius"]} * SCALE)')
                            lines.append(f'    nx.extrude({dist_val} * SCALE, operation="Subtract")')

                    # CASE B: ONLY CIRCLES (Cylinders, Holes)
                    elif circles:
                        if is_creation and len(circles) > 1:
                            # Smart Split: Largest = Body, Rest = Holes
                            largest = max(circles, key=lambda c: c['radius'])
                            holes = [c for c in circles if c != largest]

                            lines.append(f'    # --- Smart Split: Base Body (Circle) ---')
                            lines.append(f'    nx.create_plane("{plane}")')
                            lines.append(
                                f'    nx.sketch_circle({largest["center_x"]} * SCALE, {largest["center_y"]} * SCALE, {largest["radius"]} * SCALE)')
                            lines.append(f'    nx.extrude({dist_val} * SCALE, operation="Create")')

                            if holes:
                                lines.append(f'    # --- Smart Split: Holes ---')
                                lines.append(f'    nx.create_plane("{plane}")')
                                for h in holes:
                                    lines.append(
                                        f'    nx.sketch_circle({h["center_x"]} * SCALE, {h["center_y"]} * SCALE, {h["radius"]} * SCALE)')
                                lines.append(f'    nx.extrude({dist_val} * SCALE, operation="Subtract")')
                        else:
                            # Standard Single Circle or Explicit Cut
                            nx_op = "Create"
                            if op_type == "Cut":
                                nx_op = "Subtract"
                            elif op_type == "Join":
                                nx_op = "Unite"

                            lines.append(f'    # --- Standard Feature (Circle) ---')
                            lines.append(f'    nx.create_plane("{plane}")')
                            for c in circles:
                                lines.append(
                                    f'    nx.sketch_circle({c["center_x"]} * SCALE, {c["center_y"]} * SCALE, {c["radius"]} * SCALE)')
                            lines.append(f'    nx.extrude({dist_val} * SCALE, operation="{nx_op}")')

                except ValueError:
                    pass

            # Reset Buffer
            pending_sketch = {"active": False, "plane": "XY", "circles": [], "lines": []}
            i = j

        elif token == "<eos>":
            lines.append("")
            lines.append(f'    # Export result to CAD Output')
            export_name = Path(input_filename).stem + "_nx.prt"
            clean_cad_path = CAD_OUTPUT_FOLDER.replace("\\", "\\\\")
            lines.append(f'    export_path = r"{clean_cad_path}\\{export_name}"')
            lines.append(f'    nx.export(export_path)')
            lines.append(f'    print(f"Saved NX Part to: {{export_path}}")')

        i += 1

    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")

    return "\n".join(lines)


def get_latest_file(folder_path: Path) -> Path:
    files = list(folder_path.glob('merged_*.json'))
    if not files:
        raise FileNotFoundError(f"No merged JSON files found in {folder_path}")
    return max(files, key=lambda p: p.stat().st_mtime)


def main():
    # 1. Check/Create Output Dir for NX Scripts
    nx_out_path = Path(NX_SCRIPT_OUTPUT_FOLDER)
    if not nx_out_path.exists():
        os.makedirs(nx_out_path)

    # Check/Create Output Dir for CAD Parts
    cad_out_path = Path(CAD_OUTPUT_FOLDER)
    if not cad_out_path.exists():
        os.makedirs(cad_out_path)

    try:
        json_path = get_latest_file(Path(INPUT_FOLDER))
    except Exception as e:
        print(f"Error: {e}")
        return

    # Save the .py file to D:\Automtion\Merge_Output_NX
    output_py_path = nx_out_path / (json_path.stem + "_nx_journal.py")

    print(f"Loading tokens from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    tokens = data.get("generated_tokens", [])
    if not tokens:
        print("Error: No 'generated_tokens' found in JSON.")
        return

    print("Converting tokens to NX Python script...")
    script_content = parse_tokens_to_nx_script(tokens, json_path.name)

    print(f"Saving script to: {output_py_path}")
    with open(output_py_path, "w") as f:
        f.write(script_content)

    print("-" * 30)
    print("Success! NX Journal Created.")
    print(f"Run this in NX: run_journal \"{output_py_path}\"")
    print("-" * 30)


if __name__ == "__main__":
    main()