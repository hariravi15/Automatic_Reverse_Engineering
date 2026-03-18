import os
import json
import time
from pathlib import Path

ML_SEQUENCE_FOLDER = "D:\Automtion\ML_output"
DIMENSION_FOLDER = "D:\Automtion\Mes_output"
OUTPUT_FOLDER = "D:\Automtion\Merge_Output"
TIME_INTERVAL = 0 # 0 is no looop


def get_latest_file(folder_path: Path) -> Path:
    """Finds the most recently modified .json file, searching through subfolders."""
    # The rglob('*.json') searches all subdirectories recursively
    files = list(folder_path.rglob('*.json'))

    if not files:
        raise FileNotFoundError(f"No .json files found in {folder_path} or its subfolders")

    # Sort by modification time to get the absolute newest file
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    return latest_file


def merge_files(dim_file_path: Path, seq_file_path: Path) -> dict:
    with open(dim_file_path) as f:
        dim_data = json.load(f)
    with open(seq_file_path) as f:
        seq_data = json.load(f)
    template_tokens = seq_data.get('generated_tokens', [])
    template_op_types = []
    for token in template_tokens:
        if token.startswith("operation_type="):
            template_op_types.append(token.split("=")[1])
    output_tokens = []
    main_extrude_dist = None
    extrude_op_index = 0

    for operation in dim_data.get('operations', []):
        op_type = operation.get('type')

        # --- NEW BLOCK: Handle High-Level Features (L-Clamp) ---
        if op_type == "Feature":
            feat_name = operation.get('name')
            params = operation.get('parameters', {})

            if feat_name == "L-Clamp":
                # Extract Dimensions
                L = params.get('length', 50)
                W = params.get('width', 50)
                H = params.get('height', 20)
                T = params.get('thickness', 5)

                # 1. Start Sketch
                output_tokens.append("plane=XY")
                output_tokens.append("ENTITY_START__Sketch")

                # 2. Generate L-Shape Geometry (6 Lines)
                # We start at (0,0) and draw counter-clockwise
                # P1(0,0) -> P2(L,0) -> P3(L,T) -> P4(T,T) -> P5(T,W) -> P6(0,W) -> Close

                points = [
                    (0, 0),
                    (L, 0),
                    (L, T),
                    (T, T),
                    (T, W),
                    (0, W)
                ]

                for i in range(len(points)):
                    p_start = points[i]
                    p_end = points[(i + 1) % len(points)]  # Wrap around to 0 for last line

                    output_tokens.append("CURVE_START__Line")
                    output_tokens.append(f"start_x={p_start[0]}")
                    output_tokens.append(f"start_y={p_start[1]}")
                    output_tokens.append(f"end_x={p_end[0]}")
                    output_tokens.append(f"end_y={p_end[1]}")
                    output_tokens.append("CURVE_END__Line")

                output_tokens.append("ENTITY_END__Sketch")

                # 3. Extrude
                output_tokens.append("ENTITY_START__Extrude")
                output_tokens.append("operation_type=NewBody")
                output_tokens.append(f"distance={H}")
                output_tokens.append("ENTITY_END__Extrude")

        if op_type == "Sketch":
            output_tokens.append("plane=XY")
            output_tokens.append("ENTITY_START__Sketch")

            curves = operation.get('parameters', {}).get('curves', [])
            for curve in curves:
                curve_type = curve.get('type')
                center = curve.get('center_xy', [0, 0])
                radius = curve.get('radius')

                if curve_type == "Circle" and radius is not None:
                    output_tokens.append("CURVE_START__Circle")
                    output_tokens.append(f"center_x={center[0]}")
                    output_tokens.append(f"center_y={center[1]}")
                    output_tokens.append(f"radius={radius}")
                    output_tokens.append("CURVE_END__Circle")
            output_tokens.append("ENTITY_END__Sketch")

        elif op_type == "Extrude":
            output_tokens.append("ENTITY_START__Extrude")
            params = operation.get('parameters', {})
            distance = params.get('distance')
            op_type_param = params.get('operation_type')
            if not op_type_param:
                if extrude_op_index < len(template_op_types):
                    op_type_param = template_op_types[extrude_op_index]
                else:
                    op_type_param = "NewBody"  # Default if template runs out
            extrude_op_index += 1
            if main_extrude_dist is None:
                main_extrude_dist = abs(distance)
            final_distance = main_extrude_dist

            output_tokens.append(f"operation_type={op_type_param}")
            output_tokens.append(f"distance={final_distance}")
            output_tokens.append("ENTITY_END__Extrude")

    output_tokens.append("<eos>")
    return {"status": "success", "generated_tokens": output_tokens}


def main():
    ml_path = Path(ML_SEQUENCE_FOLDER)
    dim_path = Path(DIMENSION_FOLDER)
    out_path = Path(OUTPUT_FOLDER)
    out_path.mkdir(parents=True, exist_ok=True)
    processed_files = set()
    print(f"Monitoring folders...")
    print(f"  ML Sequence Folder: {ml_path.resolve()}")
    print(f"  Dimension Folder:   {dim_path.resolve()}")
    print(f"  Output Folder:      {out_path.resolve()}")
    print(f"  Check Interval:     {TIME_INTERVAL}s")

    while True:
        try:
            latest_ml_file = get_latest_file(ml_path)
            latest_dim_file = get_latest_file(dim_path)
            file_pair = (latest_ml_file.name, latest_dim_file.name)

            if file_pair in processed_files:
                print(f"Latest files {file_pair} already processed. Waiting...")
            else:
                print(f"New file pair found:")
                print(f"  ML File: {latest_ml_file.name}")
                print(f"  Dim File: {latest_dim_file.name}")
                try:
                    merged_data = merge_files(latest_dim_file, latest_ml_file)
                    output_filename = f"merged_{latest_dim_file.stem}_{latest_ml_file.stem}.json"
                    output_filepath = out_path / output_filename
                    with open(output_filepath, 'w') as f:
                        json.dump(merged_data, f, indent=2)
                    print(f"Successfully merged and saved to {output_filepath}")
                    processed_files.add(file_pair)

                except Exception as e:
                    print(f"Error processing files {file_pair}: {e}")

        except FileNotFoundError as e:
            print(f"Warning: {e}. Waiting for files...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        if TIME_INTERVAL == 0:
            print("Time interval is 0. Exiting after one run.")
            break
        else:
            time.sleep(TIME_INTERVAL)


if __name__ == "__main__":
    main()