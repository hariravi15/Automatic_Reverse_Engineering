import cv2
import numpy as np
import json
import os
from datetime import datetime

INPUT_FOLDER = r'D:\pic\mes2'
OUTPUT_FOLDER = r'D:\pic'

PPM_TOP = 3.5250
PPM_SIDE = 16.9000

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)



def get_top_view_contours(image_path, debug=False):  # Set debug=True for more print output
    image = cv2.imread(image_path)
    if image is None:
        print(f"  -> Error: Could not read image {image_path}")
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Find paper mask
    paper_edges = cv2.Canny(blurred, 50, 150)
    paper_contours, _ = cv2.findContours(paper_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not paper_contours:
        print(f"  -> Error: Could not find paper background in {os.path.basename(image_path)}")
        return image, None, None
    paper_contour = max(paper_contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [paper_contour], -1, 255, -1)

    # Find object inside mask
    kernel = np.ones((3, 3), np.uint8)  # Kernel for morphology

    # Adaptive Threshold
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)

    # Otsu Threshold
    _, otsu_thresh_unclean = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_thresh = cv2.erode(otsu_thresh_unclean, kernel, iterations=2)
    combined_thresh = cv2.bitwise_or(adaptive_thresh, otsu_thresh)

    # paper mask
    object_thresh_uncleaned = cv2.bitwise_and(combined_thresh, combined_thresh, mask=mask)

    # Clean up noise around the object
    object_thresh = cv2.morphologyEx(object_thresh_uncleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(object_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        print(f'  -> Total contours found: {len(contours)}')
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                print(f'     Contour {idx}: Hierarchy={h}  Area={cv2.contourArea(contours[idx]):.1f}')
        cv2.imshow("Debug: Otsu Uncleaned", cv2.bitwise_and(otsu_thresh_unclean, otsu_thresh_unclean, mask=mask))
        cv2.imshow("Debug: Otsu ERODED (iter=2)", cv2.bitwise_and(otsu_thresh, otsu_thresh, mask=mask))
        cv2.imshow("Debug: Final Threshold", object_thresh)
        cv2.waitKey(0)

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

                # Get data from measurements dict
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


def get_top_view_contours_v2(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        print("  -> Error: could not read", image_path)
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    edges = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, None, None
    paper_cnt = max(cnts, key=cv2.contourArea)
    paper_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(paper_mask, [paper_cnt], -1, 255, -1)
    bin_im = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (7, 7), 0),
        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    # kill noise outside paper
    bin_im = cv2.bitwise_and(bin_im, bin_im, mask=paper_mask)
    # close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_im = cv2.morphologyEx(bin_im, cv2.MORPH_CLOSE, kernel, iterations=2)

    #distance transform + sure markers
    dist = cv2.distanceTransform(bin_im, cv2.DIST_L2, 5)
    _, sure_bg = cv2.threshold(dist, 0.2 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_bg = np.uint8(sure_bg)
    _, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(colour, markers)
    bin_markers = np.zeros((h, w), dtype=np.uint8)
    bin_markers[markers > 1] = 255  # holes
    bin_markers[markers == 1] = 127  # object body

    outer_mask = np.uint8(bin_markers == 127)
    cnts, _ = cv2.findContours(outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, None, None
    outer_cnt = max(cnts, key=cv2.contourArea)

    hole_mask = np.uint8(bin_markers == 255)
    hole_cnts, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = cv2.contourArea(outer_cnt) * 0.005
    hole_cnts = [c for c in hole_cnts if cv2.contourArea(c) > min_area]

    contours = [outer_cnt] + hole_cnts
    hierarchy = []
    for i, c in enumerate(contours):
        if i == 0:
            hierarchy.append([i + 1 if i + 1 < len(contours) else -1, -1, 1 if len(contours) > 1 else -1, -1])
        else:
            hierarchy.append([i + 1 if i + 1 < len(contours) else -1, i - 1, -1, 0])
    hierarchy = np.array([hierarchy], dtype=np.int32)

    if debug:
        cv2.imshow(" watershed markers", cv2.applyColorMap(np.uint8(markers * 10), cv2.COLORMAP_JET))
        cv2.waitKey(0)

    return img, contours, hierarchy


def get_contours_from_sketch(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"  -> Error: Could not read image {image_path}")
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or hierarchy is None:
        print(f"  -> Error: No contours found in sketch {image_path}")
        return image, None, None
    hierarchy = hierarchy[0]
    filled_mask = np.zeros_like(gray)
    for i, h in enumerate(hierarchy):
        if h[3] == -1:
            cv2.drawContours(filled_mask, contours, i, 255, -1)  # -1 fills
    for i, h in enumerate(hierarchy):
        if h[3] != -1:
            cv2.drawContours(filled_mask, contours, i, 0, -1)  # 0  fills

    if debug:
        cv2.imshow("Debug: Sketch Threshold", thresh)
        cv2.imshow("Debug: Filled Sketch Mask (Corrected)", filled_mask)
        cv2.waitKey(0)
    final_contours, final_hierarchy = cv2.findContours(filled_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return image, final_contours, final_hierarchy

def process_component(top_view_path, side_view_path):
    print(f"\nProcessing component: {os.path.basename(top_view_path)}")
    top_image, top_contours, top_hierarchy = get_top_view_contours_v2(top_view_path, debug=False)
    if not top_contours or top_hierarchy is None:  # Added check for hierarchy
        print("  -> Could not find any contours or hierarchy in the top view.")
        return

    main_contour = max(top_contours, key=cv2.contourArea)
    main_contour_index = -1

    for i, c in enumerate(top_contours):
        if np.array_equal(c, main_contour):
            main_contour_index = i
            break

    if main_contour_index == -1:
        print("  -> Error: Could not find main contour index. Skipping.")
        return
    disk_area = cv2.contourArea(main_contour)
    min_hole_area = disk_area * 0.005

    max_hole_area = disk_area * 0.98

    gray = cv2.cvtColor(top_image, cv2.COLOR_BGR2GRAY)
    children = []
    hierarchy = top_hierarchy[0]
    print(f"Total contours found: {len(top_contours)}. Main contour index: {main_contour_index}")
    print(f" Hierarchy data: {hierarchy}")

    for i, c in enumerate(top_contours):
        if hierarchy[i][3] != main_contour_index:
            continue
        area = cv2.contourArea(c)
        if area < min_hole_area:
            print(f" Hole {i} skipped: too small (Area: {area})")
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > 0.7:
            children.append(c)
        else:
            print(f"Hole {i} skipped: not circular (Circularity: {circularity:.2f})")

    print(f"Found {len(children)} inner holes after hierarchy filtering.")

    shape_type = classify_shape(top_contours)
    print(f"Detected primary shape: {shape_type}")
    print(f"Found {len(children)} valid inner holes.")

    height = measure_height_from_side_view(side_view_path, PPM_SIDE)
    measurements = {"height_mm": height}

    if shape_type == "cylinder":
        measurements.update(measure_cylinder(main_contour, children, PPM_TOP))
    elif shape_type == "cuboid":
        measurements.update(measure_cuboid(main_contour, PPM_TOP))
    elif shape_type == "l_clamp":
        measurements.update(measure_l_clamp(main_contour, PPM_TOP))
    else:
        print(f"No measurement logic implemented for shape: {shape_type}")
        return

    annotated_image = visualize_measurements(top_image, shape_type, main_contour, children, measurements)
    vis_filename = os.path.splitext(os.path.basename(top_view_path))[0] + '_annotated.jpg'
    output_vis_path = os.path.join(OUTPUT_FOLDER, vis_filename)
    cv2.imwrite(output_vis_path, annotated_image)
    print(f"Saved annotated image to {output_vis_path}")

    cv2.imshow("Measurement Visualization", annotated_image)
    cv2.waitKey(0)

    json_output = generate_cad_json(shape_type, measurements, os.path.basename(top_view_path))
    json_filename = os.path.splitext(os.path.basename(top_view_path))[0] + '.json'
    output_path = os.path.join(OUTPUT_FOLDER, json_filename)
    with open(output_path, 'w') as f:
        f.write(json_output)
    print(f"Successfully saved measurements to {output_path}")

if __name__ == "__main__":
    print("Starting Batch Measurement Process")
    all_files = os.listdir(INPUT_FOLDER)
    top_view_files = [f for f in all_files if 'top' in f.lower() and f.endswith(('.png', '.jpg', '.jpeg'))]

    if not top_view_files:
        print(f"Error: No top-view images found in {INPUT_FOLDER}. Ensure files contain 'top' in their name.")
    else:
        for top_file in top_view_files:
            base_name = os.path.splitext(top_file)[0].lower().replace('_top', '').replace('top', '')
            side_file = None
            for f in all_files:
                f_base = os.path.splitext(f)[0].lower().replace('_side', '').replace('side', '')
                if f_base == base_name and 'side' in f.lower():
                    side_file = f
                    break

            if side_file:
                top_path = os.path.join(INPUT_FOLDER, top_file)
                side_path = os.path.join(INPUT_FOLDER, side_file)
                process_component(top_path, side_path)
            else:
                print(f"Warning: Found top view '{top_file}' but no corresponding side view.")

    cv2.destroyAllWindows()  # Close the last window
    print("\n Batch Process Finished ")