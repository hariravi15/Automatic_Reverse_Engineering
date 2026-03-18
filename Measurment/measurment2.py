# automatic_measurement.py
import cv2
import numpy as np
import json
import os
from datetime import datetime
import argparse
import sys
INPUT_FOLDER = r'D:\Automtion\Measurement_Input\test4'
OUTPUT_FOLDER = r'D:\Automtion\Mes_output'

#INPUT_FOLDER = r'D:\pic\mes2'
#OUTPUT_FOLDER = r'D:\pic'

PPM_TOP = 3.5250
PPM_SIDE = 16.9000

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def get_top_view_contours(image_path, debug=False):
    import cv2, numpy as np, os

    image = cv2.imread(image_path)
    if image is None:
        print(f"  -> Error: Could not read image {image_path}")
        return None, None, None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Shadow / illumination correction ---
    gray_float = gray.astype(np.float32)
    blur_background = cv2.GaussianBlur(gray_float, (55, 55), 0)
    normalized = cv2.divide(gray_float, blur_background, scale=255)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(normalized)

    # --- CHANGE 1: Calibrated Gaussian Blur ---
    # Changed from (7, 7) to (7, 7) based on your 'MY_BL' value
    blurred = cv2.GaussianBlur(normalized, (7, 7), 0)

    # --- CHANGE 2: Calibrated Paper Boundary Detection ---
    # Changed Canny from (0, 184) to (7, 185)
    paper_edges = cv2.Canny(blurred, 7, 185)
    paper_contours, _ = cv2.findContours(paper_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not paper_contours:
        print(f"  -> Error: Could not find paper background in {os.path.basename(image_path)}")
        return image, None, None

    paper_contour = max(paper_contours, key=cv2.contourArea)

    # Create mask for object region
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [paper_contour], -1, 255, -1)
    kernel = np.ones((3, 3), np.uint8)

    # --- CHANGE 3: Calibrated Adaptive Thresholding ---
    # Block size remains 33; Constant C changed from 2 to 0
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 33, 0)

    _, otsu_thresh_unclean = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine and Apply mask
    combined_thresh = cv2.bitwise_or(adaptive_thresh, otsu_thresh_unclean)
    object_thresh_uncleaned = cv2.bitwise_and(combined_thresh, combined_thresh, mask=mask)

    # Clean noise
    object_thresh = cv2.morphologyEx(object_thresh_uncleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    object_thresh = cv2.morphologyEx(object_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- CHANGE 4: Calibrated Contour Enhancement ---
    # Changed Canny from (0, 184) to (7, 185)
    edges = cv2.Canny(object_thresh, 7, 185)
    object_thresh = cv2.bitwise_or(object_thresh, edges)

    contours, hierarchy = cv2.findContours(object_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # (Debug visualization code remains the same...)
    if debug:
        print(f'  -> Total contours found: {len(contours)}')
        if len(contours) > 0:
            debug_img = image.copy()
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)
            cv2.imshow("Top View Debug Contours", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if not contours or hierarchy is None:
        print(f"  -> Could not find any contours or hierarchy in the top view.")
        return image, None, None

    return image, contours, hierarchy



def measure_height_from_side_view(side_image_path, ppm):
    image = cv2.imread(side_image_path)
    if image is None: return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    main_contour = max(contours, key=cv2.contourArea)
    _, _, _, h_px = cv2.boundingRect(main_contour)
    return h_px / ppm

def measure_cylinder(main_contour, children, ppm):
    measurements = {}
    (x, y), od_radius_px = cv2.minEnclosingCircle(main_contour)
    measurements['outer_diameter_mm'] = (od_radius_px * 2) / ppm
    center_xy_px = (x, y)

    if not children:
        pass

    elif children:
        children_sorted = sorted(children, key=cv2.contourArea, reverse=True)
        id_contour = children_sorted[0]
        (_, id_radius_px) = cv2.minEnclosingCircle(id_contour)
        measurements['inner_diameter_mm'] = (id_radius_px * 2) / ppm
        bolt_hole_contours = children_sorted[1:]

        if bolt_hole_contours:
            measurements['bolt_holes'] = []
            for hole_contour in bolt_hole_contours:
                (hx, hy), h_radius_px = cv2.minEnclosingCircle(hole_contour)
                measurements['bolt_holes'].append({
                    'diameter_mm': (h_radius_px * 2) / ppm,
                    'position_xy_mm': ((hx - center_xy_px[0]) / ppm, (hy - center_xy_px[1]) / ppm)
                })
    return measurements


def measure_cuboid(main_contour, ppm):
    rect = cv2.minAreaRect(main_contour)
    (w_px, h_px) = rect[1]
    return {'length_mm': max(w_px, h_px) / ppm, 'width_mm': min(w_px, h_px) / ppm}


def measure_l_clamp(main_contour, ppm):
    x, y, w, h = cv2.boundingRect(main_contour)
    area = cv2.contourArea(main_contour)
    denominator = (w + h - (area / (w + h))) if (w + h) > 0 else 1
    if denominator == 0:
        estimated_thickness_px = 0
    else:
        estimated_thickness_px = area / denominator

    return {'overall_length_mm': w / ppm,
            'overall_width_mm': h / ppm,
            'estimated_thickness_mm': estimated_thickness_px / ppm
            }

def classify_shape(contours):
    if not contours: return "unknown"
    main_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(main_contour, True)
    if perimeter == 0: return "unknown"
    area = cv2.contourArea(main_contour)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    if circularity > 0.8:
        return "cylinder"

    corners = cv2.approxPolyDP(main_contour, 0.02 * perimeter, True)
    if len(corners) == 4:
        return "cuboid"
    elif 5 <= len(corners) <= 8:  # L-clamps often have 6 or 8 corners
        return "l_clamp"
    if 0.6 < circularity <= 0.8:
        return "cylinder"
    return "unknown"

def generate_cad_json(shape_type, measurements, source_filename):
    operations = []
    if shape_type == "cylinder":
        od = measurements.get('outer_diameter_mm', 0)
        id_val = measurements.get('inner_diameter_mm', 0)
        height = measurements.get('height_mm', 0)
        curves = [{"type": "Circle", "center_xy": [0, 0], "radius": od / 2}]
        if id_val > 0:
            curves.append({"type": "Circle", "center_xy": [0, 0], "radius": id_val / 2})
        operations.append({"type": "Sketch", "name": "BaseSketch", "parameters": {"curves": curves}})
        operations.append({"type": "Extrude", "name": "BaseExtrude", "parameters": {"distance": height}})
        if "bolt_holes" in measurements:
            hole_curves = [{"type": "Circle", "center_xy": list(h['position_xy_mm']), "radius": h['diameter_mm'] / 2}
                           for h in measurements['bolt_holes']]
            operations.append({"type": "Sketch", "name": "HoleSketch", "parameters": {"curves": hole_curves}})
            operations.append(
                {"type": "Extrude", "name": "HoleCut", "parameters": {"distance": -height, "operation_type": "Cut"}})

    elif shape_type == "cuboid":
        length = measurements.get('length_mm', 0)
        width = measurements.get('width_mm', 0)
        height = measurements.get('height_mm', 0)
        curves = [{"type": "Rectangle", "center_xy": [0, 0], "length": length, "width": width}]
        operations.append({"type": "Sketch", "name": "BaseSketch", "parameters": {"curves": curves}})
        operations.append({"type": "Extrude", "name": "BaseExtrude", "parameters": {"distance": height}})

    elif shape_type == "l_clamp":
        length = measurements.get('overall_length_mm', 0)
        width = measurements.get('overall_width_mm', 0)
        height = measurements.get('height_mm', 0)
        thickness = measurements.get('estimated_thickness_mm', 0)
        operations.append({"type": "Feature", "name": "L-Clamp", "parameters": {
            "length": length, "width": width, "height": height, "thickness": thickness}})

    output = {
        "metadata": {"source_image": source_filename, "component_type": shape_type,
                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "operations": operations
    }
    return json.dumps(output, indent=4, default=lambda x: round(x, 2))

def visualize_measurements(image, shape_type, main_contour, children, measurements):
    vis_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if shape_type == "cylinder":
        (x, y), radius = cv2.minEnclosingCircle(main_contour)
        cv2.circle(vis_image, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        od = measurements.get('outer_diameter_mm', 0)
        cv2.putText(vis_image, f"OD: {od:.2f}mm", (int(x - radius), int(y - radius - 10)), font, 0.6, (0, 0, 255), 2)
        children_sorted = sorted(children, key=cv2.contourArea, reverse=True)
        if "inner_diameter_mm" in measurements and children_sorted:
            id_contour = children_sorted[0]
            (ix, iy), id_radius = cv2.minEnclosingCircle(id_contour)
            cv2.circle(vis_image, (int(ix), int(iy)), int(id_radius), (0, 255, 0), 2)

            id_val = measurements.get('inner_diameter_mm', 0)
            cv2.putText(vis_image, f"ID: {id_val:.2f}mm", (int(ix) + 5, int(iy) - 5), font, 0.6, (0, 255, 0), 2)
        if "bolt_holes" in measurements:
            bolt_hole_contours = children_sorted[1:]
            for i, hole_contour in enumerate(bolt_hole_contours):
                (hx, hy), h_radius = cv2.minEnclosingCircle(hole_contour)
                cv2.circle(vis_image, (int(hx), int(hy)), int(h_radius), (0, 255, 0), 2)
                if i < len(measurements['bolt_holes']):
                    hole_data = measurements['bolt_holes'][i]
                    cv2.putText(vis_image, f"H{i + 1}: {hole_data['diameter_mm']:.2f}mm", (int(hx) + 5, int(hy)), font,
                                0.5,
                                (0, 255, 0), 2)

    elif shape_type == "cuboid":
        rect = cv2.minAreaRect(main_contour)
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(vis_image, [box], 0, (0, 0, 255), 2)
        length = measurements.get('length_mm', 0)
        width = measurements.get('width_mm', 0)
        cv2.putText(vis_image, f"L: {length:.2f}mm", (box[0][0] - 80, box[0][1]), font, 0.6, (0, 0, 255), 2)
        cv2.putText(vis_image, f"W: {width:.2f}mm", (box[1][0], box[1][1] - 10), font, 0.6, (0, 0, 255), 2)

    else:
        cv2.drawContours(vis_image, [main_contour], -1, (0, 0, 255), 2)
        for child in children:
            cv2.drawContours(vis_image, [child], -1, (0, 255, 0), 2)
    return vis_image


def process_component(top_view_path, side_view_path):
    print(f"\nProcessing component: {os.path.basename(top_view_path)}")

    # 1. Image Processing & Contour Detection
    top_image, top_contours, top_hierarchy = get_top_view_contours(top_view_path, debug=False)
    if not top_contours or top_hierarchy is None:
        print("  -> Could not find any contours or hierarchy in the top view.")
        return

    main_contour = max(top_contours, key=cv2.contourArea)
    gray = cv2.cvtColor(top_image, cv2.COLOR_BGR2GRAY)

    # Create mask to find internal holes
    main_mask = np.zeros_like(gray)
    cv2.drawContours(main_mask, [main_contour], -1, 255, -1)

    disk_area = cv2.contourArea(main_contour)
    min_hole_area = disk_area * 0.005
    children = []

    # Filter for valid holes
    for c in top_contours:
        if np.array_equal(c, main_contour):
            continue

        test_point = tuple(c[0][0])
        # Check if contour is inside the main object
        if main_mask[test_point[1], test_point[0]] == 255:
            area = cv2.contourArea(c)
            if area > min_hole_area:
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    if circularity > 0.7:
                        children.append(c)

    # 2. Shape Classification & Measurement
    shape_type = classify_shape(top_contours)
    height = measure_height_from_side_view(side_view_path, PPM_SIDE)
    measurements = {"height_mm": height}

    if shape_type == "cylinder":
        measurements.update(measure_cylinder(main_contour, children, PPM_TOP))
    elif shape_type == "cuboid":
        measurements.update(measure_cuboid(main_contour, PPM_TOP))
    elif shape_type == "l_clamp":
        measurements.update(measure_l_clamp(main_contour, PPM_TOP))
    else:
        measurements["info"] = "Unknown shape - basic measurements only"

    # 3. Preparation for Saving
    base_filename = os.path.splitext(os.path.basename(top_view_path))[0]
    output_vis_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_annotated.jpg")
    output_json_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.json")

    annotated_image = visualize_measurements(top_image, shape_type, main_contour, children, measurements)
    json_output = generate_cad_json(shape_type, measurements, os.path.basename(top_view_path))

    # 4. SAVE EVERYTHING (Automatic - No waiting)
    # Save Image
    cv2.imwrite(output_vis_path, annotated_image)
    print(f"  -> Saved annotated image to {output_vis_path}")

    # Save JSON
    try:
        with open(output_json_path, 'w') as f:
            f.write(json_output)
        print(f"  -> Successfully saved measurements to {output_json_path}")
    except Exception as e:
        print(f"  -> Error saving JSON: {e}")

    # 5. SHOW RESULT (Non-blocking)
    cv2.imshow("Measurement Visualization", annotated_image)
    # 1000ms delay allows the script to continue to the next image automatically
    cv2.waitKey(1000)



if __name__ == "__main__":
    print("--- Starting Measurement Process ---")

    # 1. Setup Argument Parser to catch commands from the Automation Script
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', help="Path to top view image")
    parser.add_argument('--left', help="Path to left (side) view image")  # Automation sends 'left'
    parser.add_argument('--side', help="Path to side view image (alternate)")
    parser.add_argument('--output', help="Path to output directory")
    args, unknown = parser.parse_known_args()

    # 2. CHECK: Are we running via Automation (Arguments provided)?
    if args.top and (args.left or args.side) and args.output:
        print(f"-> Mode: Single Component (Automation)")

        # The automation script sends 'left.png' as the side view
        side_view_path = args.left if args.left else args.side
        top_view_path = args.top

        # Overwrite the global OUTPUT_FOLDER with the one from automation
        OUTPUT_FOLDER = args.output
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        if os.path.exists(top_view_path) and os.path.exists(side_view_path):
            try:
                # Run your existing logic on these specific files
                process_component(top_view_path, side_view_path)
            except Exception as e:
                print(f"Error processing component: {e}")
        else:
            print(f"Error: One of the input files does not exist.\nTop: {top_view_path}\nSide: {side_view_path}")

    # 3. FALLBACK: No arguments? Run the old Batch Mode (Hardcoded Folders)
    else:
        print(f"-> Mode: Batch Directory Scan (Default)")
        print(f"-> Scanning folder: {INPUT_FOLDER}")

        if not os.path.exists(INPUT_FOLDER):
            print(f"Error: Input folder not found: {INPUT_FOLDER}")
        else:
            all_files = os.listdir(INPUT_FOLDER)
            top_view_files = [f for f in all_files if 'top' in f.lower() and f.endswith(('.png', '.jpg', '.jpeg'))]

            if not top_view_files:
                print(f"Error: No top-view images found in {INPUT_FOLDER}.")
            else:
                for top_file in top_view_files:
                    # Logic to find matching side file
                    base_name = os.path.splitext(top_file)[0].lower().replace('_top', '').replace('top', '')
                    side_file = None
                    for f in all_files:
                        f_base = os.path.splitext(f)[0].lower().replace('_side', '').replace('side', '')
                        # Look for 'side' or 'left' in the filename
                        if f_base == base_name and ('side' in f.lower() or 'left' in f.lower()):
                            side_file = f
                            break

                    if side_file:
                        top_path = os.path.join(INPUT_FOLDER, top_file)
                        side_path = os.path.join(INPUT_FOLDER, side_file)
                        process_component(top_path, side_path)
                    else:
                        print(f"Warning: Found top view '{top_file}' but no corresponding side/left view.")

    cv2.destroyAllWindows()
    print("\n--- Process Finished ---")