
import cv2


def calculate_ppm(image_path, known_dimension_mm, is_dark_object=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Calibration image not found at {image_path}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh_type = cv2.THRESH_BINARY_INV if is_dark_object else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(blurred, 127, 255, thresh_type + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_contour = max(contours, key=cv2.contourArea)
    if not is_dark_object:  # Paper
        rect = cv2.minAreaRect(ref_contour)
        pixel_dimension = min(rect[1])  # Short side of the paper
    else:  # Sticker
        _, _, _, pixel_dimension = cv2.boundingRect(ref_contour)  # Height of the sticker
    ppm = pixel_dimension / known_dimension_mm
    return ppm


if __name__ == "__main__":
    TOP_VIEW_CALIBRATION_IMAGE = r'D:\pic\pic\New folder\2_Color.png'
    SIDE_VIEW_CALIBRATION_IMAGE = r'D:\pic\pic\New folder\2_1_Color.png'
    PAPER_WIDTH_MM = 194.0  # A4 paper short side
    STICKER_HEIGHT_MM = 30.0  # Measure your stickers height and enter it here
    print("Running Calibration")
    ppm_top = calculate_ppm(TOP_VIEW_CALIBRATION_IMAGE, PAPER_WIDTH_MM, is_dark_object=False)
    ppm_side = calculate_ppm(SIDE_VIEW_CALIBRATION_IMAGE, STICKER_HEIGHT_MM, is_dark_object=True)
    if ppm_top and ppm_side:
        print("\nCALIBRATION COMPLETE ")
        print(f"PPM_TOP = {ppm_top:.4f}")
        print(f"PPM_SIDE = {ppm_side:.4f}")