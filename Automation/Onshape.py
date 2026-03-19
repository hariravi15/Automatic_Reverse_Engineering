import json
import sys
from pathlib import Path

INPUT_FOLDER = r"Path to your folder"
INPUT_UNIT = "mm"

NEW_ACCESS_KEY = "Your access Key"
NEW_SECRET_KEY = "Your screat key"


def parse_tokens_to_script(tokens: list, input_filename: str) -> str:
    lines = []

    lines.append("import os")
    lines.append("import sys")
    lines.append("import time")

    lines.append(f"ACCESS_KEY = '{NEW_ACCESS_KEY}'")
    lines.append(f"SECRET_KEY = '{NEW_SECRET_KEY}'")

    lines.append('config_content = f"""')
    lines.append('base_url: https://cad.onshape.com')
    lines.append('access_key: {ACCESS_KEY}')
    lines.append('secret_key: {SECRET_KEY}')
    lines.append('"""')

    lines.append('config_path = "temp_onshape_config.yaml"')
    lines.append('with open(config_path, "w") as f:')
    lines.append('    f.write(config_content)')


    lines.append('os.environ["ONSHAPE_CLIENT_CONFIG_FILE"] = config_path')


    lines.append("import onpy")
    lines.append("")

    lines.append(f"# Input data was in {INPUT_UNIT}, converting to inches for Onshape")
    if INPUT_UNIT == "mm":
        lines.append("SCALE = 1 / 25.4")
    elif INPUT_UNIT == "cm":
        lines.append("SCALE = 1 / 2.54")
    else:
        lines.append("SCALE = 1.0")

    lines.append("")
    lines.append("def main():")
    lines.append(f'    print("Generating model from: {input_filename}")')


    lines.append(f'    print(f"Using Access Key: {{ACCESS_KEY[:5]}}... (Should be your NEW key)")')
    lines.append(f'    # Create document and get default partstudio')
    doc_name = Path(input_filename).stem
    lines.append(f'    document = onpy.create_document("{doc_name}")')
    lines.append(f'    partstudio = document.get_partstudio()')
    lines.append("")
    lines.append(f'    main_part = None')
    lines.append("")

    pending_sketch = {
        "active": False,
        "plane": None,
        "name": "",
        "circles": [],
        "lines": []
    }

    sketch_counter = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]


        if token.startswith("plane="):
            plane_name_raw = token.split("=")[1]
            if i + 1 < len(tokens) and tokens[i + 1] == "ENTITY_START__Sketch":
                plane_code = "partstudio.features.top_plane"
                if plane_name_raw.lower() in ["top", "xy"]:
                    plane_code = "partstudio.features.top_plane"
                elif plane_name_raw.lower() in ["front", "yz"]:
                    plane_code = "partstudio.features.front_plane"
                elif plane_name_raw.lower() in ["right", "xz"]:
                    plane_code = "partstudio.features.right_plane"

                pending_sketch = {
                    "active": True,
                    "plane": plane_code,
                    "plane_name": plane_name_raw,
                    "name": f"Sketch_{sketch_counter}",
                    "circles": [],
                    "lines": []
                }
                sketch_counter += 1
                i += 1


        elif token == "CURVE_START__Circle":
            params = {}
            j = i + 1
            while j < len(tokens) and tokens[j] != "CURVE_END__Circle":
                try:
                    if "=" in tokens[j]:
                        key, val = tokens[j].split("=")
                        params[key] = float(val)
                except (ValueError, IndexError):
                    pass
                j += 1
            if pending_sketch["active"] and "center_x" in params and "radius" in params:
                pending_sketch["circles"].append(params)
            i = j


        elif token == "CURVE_START__Line":
            params = {}
            j = i + 1
            while j < len(tokens) and tokens[j] != "CURVE_END__Line":
                try:
                    if "=" in tokens[j]:
                        key, val = tokens[j].split("=")
                        params[key] = float(val)
                except (ValueError, IndexError):
                    pass
                j += 1
            if pending_sketch["active"] and all(k in params for k in ["start_x", "start_y", "end_x", "end_y"]):
                pending_sketch["lines"].append(params)
            i = j


        elif token == "ENTITY_START__Extrude":
            params = {}
            j = i + 1
            while j < len(tokens) and tokens[j] != "ENTITY_END__Extrude":
                try:
                    if "=" in tokens[j]:
                        key, val = tokens[j].split("=")
                        params[key] = val
                except (ValueError, IndexError):
                    pass
                j += 1

            if pending_sketch["active"] and "distance" in params:
                try:
                    distance_val = float(params["distance"])
                    op_type_raw = params.get("operation_type", "NewBody")

                    circles = pending_sketch["circles"]
                    lines_data = pending_sketch["lines"]

                    is_creation_event = (op_type_raw in ["NewBody", "Join"])

                    def write_geometry(sketch_var, lines_list, circles_list):
                        if lines_list:
                            for l in lines_list:
                                lines.append(
                                    f'    {sketch_var}.add_line(start=({l["start_x"]} * SCALE, {l["start_y"]} * SCALE), end=({l["end_x"]} * SCALE, {l["end_y"]} * SCALE))')
                                lines.append(f'    time.sleep(0.1)')  # <--- TINY PAUSE PER LINE
                        if circles_list:
                            for c in circles_list:
                                lines.append(
                                    f'    {sketch_var}.add_circle(center=({c["center_x"]} * SCALE, {c["center_y"]} * SCALE), radius={c["radius"]} * SCALE)')
                                lines.append(f'    time.sleep(0.1)')  # <--- TINY PAUSE PER CIRCLE


                    if lines_data:
                        s_name = pending_sketch["name"]
                        lines.append(f'    # --- Profile Feature (Lines) ---')
                        lines.append(f'    {s_name} = partstudio.add_sketch(')
                        lines.append(f'        plane={pending_sketch["plane"]},')
                        lines.append(f'        name="{s_name}"')
                        lines.append(f'    )')

                        for l in lines_data:
                            lines.append(
                                f'    {s_name}.add_line(start=({l["start_x"]} * SCALE, {l["start_y"]} * SCALE), end=({l["end_x"]} * SCALE, {l["end_y"]} * SCALE))')

                        if op_type_raw == "Cut":
                            lines.append(f'    if main_part:')
                            lines.append(f'        partstudio.add_extrude(')
                            lines.append(f'            faces={s_name},')
                            lines.append(f'            distance={distance_val} * SCALE,')
                            lines.append(f'            subtract_from=main_part')
                            lines.append(f'        )')

                            lines.append(f'        time.sleep(1.0)')
                        else:
                            lines.append(f'    ext_feat = partstudio.add_extrude(')
                            lines.append(f'        faces={s_name},')
                            lines.append(f'        distance={distance_val} * SCALE')
                            lines.append(f'    )')
                            lines.append(f'    time.sleep(1.0)')  

                            lines.append(f'    try:')
                            lines.append(f'        parts = ext_feat.get_created_parts()')
                            lines.append(f'        if parts: main_part = parts[0]')
                            lines.append(f'    except: pass')


                        if circles:
                            hole_sketch_name = f"{s_name}_holes"
                            lines.append(f'    # --- Holes inside Profile ---')
                            lines.append(f'    {hole_sketch_name} = partstudio.add_sketch(')
                            lines.append(f'        plane={pending_sketch["plane"]},')
                            lines.append(f'        name="{hole_sketch_name}"')
                            lines.append(f'    )')

                            for c in circles:
                                lines.append(
                                    f'    {hole_sketch_name}.add_circle(center=({c["center_x"]} * SCALE, {c["center_y"]} * SCALE), radius={c["radius"]} * SCALE)')

                            lines.append(f'    if main_part:')
                            lines.append(f'        partstudio.add_extrude(')
                            lines.append(f'            faces={hole_sketch_name},')
                            lines.append(f'            distance={distance_val} * SCALE,')
                            lines.append(f'            subtract_from=main_part')
                            lines.append(f'        )')

                            lines.append(f'        time.sleep(1.0)')


                    elif circles:
                        if is_creation_event and len(circles) > 1:

                            largest_circle = max(circles, key=lambda c: c['radius'])
                            holes = [c for c in circles if c != largest_circle]


                            base_sketch_name = pending_sketch["name"]
                            lines.append(f'    # Smart Generation: Separating Body from Holes')
                            lines.append(f'    {base_sketch_name} = partstudio.add_sketch(')
                            lines.append(f'        plane={pending_sketch["plane"]},')
                            lines.append(f'        name="{base_sketch_name}"')
                            lines.append(f'    )')
                            lines.append(
                                f'    {base_sketch_name}.add_circle(center=({largest_circle["center_x"]} * SCALE, {largest_circle["center_y"]} * SCALE), radius={largest_circle["radius"]} * SCALE)')

                            lines.append(f'    extrude_base = partstudio.add_extrude(')
                            lines.append(f'        faces={base_sketch_name},')
                            lines.append(f'        distance={distance_val} * SCALE')
                            lines.append(f'    )')

                            lines.append(f'    time.sleep(1.0)')

                            lines.append(f'    try:')
                            lines.append(f'        parts = extrude_base.get_created_parts()')
                            lines.append(f'        if parts: main_part = parts[0]')
                            lines.append(f'    except: pass')


                            if holes:
                                hole_sketch_name = f"{base_sketch_name}_holes"
                                lines.append(f'    {hole_sketch_name} = partstudio.add_sketch(')
                                lines.append(f'        plane={pending_sketch["plane"]},')
                                lines.append(f'        name="{hole_sketch_name}"')
                                lines.append(f'    )')
                                for h in holes:
                                    lines.append(
                                        f'    {hole_sketch_name}.add_circle(center=({h["center_x"]} * SCALE, {h["center_y"]} * SCALE), radius={h["radius"]} * SCALE)')

                                lines.append(f'    if main_part:')
                                lines.append(f'        partstudio.add_extrude(')
                                lines.append(f'            faces={hole_sketch_name},')
                                lines.append(f'            distance={distance_val} * SCALE,')
                                lines.append(f'            subtract_from=main_part')
                                lines.append(f'        )')

                                lines.append(f'        time.sleep(1.0)')


                        else:
                            s_name = pending_sketch["name"]
                            lines.append(f'    # Standard Sketch Generation')
                            lines.append(f'    {s_name} = partstudio.add_sketch(')
                            lines.append(f'        plane={pending_sketch["plane"]},')
                            lines.append(f'        name="{s_name}"')
                            lines.append(f'    )')

                            for c in circles:
                                lines.append(
                                    f'    {s_name}.add_circle(center=({c["center_x"]} * SCALE, {c["center_y"]} * SCALE), radius={c["radius"]} * SCALE)')

                            if op_type_raw == "Cut":
                                lines.append(f'    if main_part:')
                                lines.append(f'        partstudio.add_extrude(')
                                lines.append(f'            faces={s_name},')
                                lines.append(f'            distance={distance_val} * SCALE,')
                                lines.append(f'            subtract_from=main_part')
                                lines.append(f'        )')

                                lines.append(f'        time.sleep(1.0)')
                            else:
                                lines.append(f'    ext_feat = partstudio.add_extrude(')
                                lines.append(f'        faces={s_name},')
                                lines.append(f'        distance={distance_val} * SCALE')
                                lines.append(f'    )')

                                lines.append(f'    time.sleep(1.0)')

                                lines.append(f'    try:')
                                lines.append(f'        parts = ext_feat.get_created_parts()')
                                lines.append(f'        if parts: main_part = parts[0]')
                                lines.append(f'    except: pass')

                except ValueError:
                    pass

            pending_sketch = {"active": False, "plane": None, "circles": [], "lines": []}
            i = j


        elif token == "<eos>":
            lines.append("")
            lines.append('    print("Model generation instructions sent to Onshape.")')

            lines.append('    try:')
            lines.append('        if os.path.exists(config_path): os.remove(config_path)')
            lines.append('    except: pass')
        i += 1

    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    try:")
    lines.append("        main()")
    lines.append("    except Exception as e:")
    lines.append("        print(f'An error occurred: {e}')")
    lines.append("        sys.exit(1)")

    return "\n".join(lines)


def get_latest_file(folder_path: Path) -> Path:
    files = list(folder_path.glob('merged_*.json'))
    if not files:
        raise FileNotFoundError(f"No merged JSON files found in {folder_path}")
    return max(files, key=lambda p: p.stat().st_mtime)


def main():
    try:
        json_path = get_latest_file(Path(INPUT_FOLDER))
    except Exception as e:
        print(f"Error: {e}")
        return

    output_py_path = json_path.with_suffix(".py")
    print(f"Loading tokens from: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    tokens = data.get("generated_tokens", [])
    if not tokens:
        print("Error: No 'generated_tokens' found in JSON file.")
        return

    print("Parsing tokens and generating Python script...")
    script_content = parse_tokens_to_script(tokens, json_path.name)

    print(f"Saving new script to: {output_py_path}")
    with open(output_py_path, "w") as f:
        f.write(script_content)
    print("-" * 30)
    print(f"Success! To build the model, run:\npython {output_py_path}")
    print("-" * 30)


if __name__ == "__main__":
    main()
