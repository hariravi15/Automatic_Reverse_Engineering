import time
import os
import shutil 

ML_SCRIPT_DIR = r"Path to script"
ML_SCRIPT_NAME = "MV_unified_3.py"
CONDA_ENV = "reverse_engineeering"
WATCH_DIR = r"Path to input folder"
MEASUREMENT_INPUT_DIR = r"path to measurment script"
MEASURE_SCRIPT_PATH = r"\Measurment\Measurment.py"
MERGE_SCRIPT_PATH = r"\Automation\merge_script.py"
ONSHAPE_CONVERT_PATH = r"\Automation\Onshape.py"
SOLIDWORKS_EXPORT_PATH = r"\Automation\Solidworks.py"
NX_CONVERT_PATH = r"\Automation\NX.py"
NX_JOURNAL_EXE_PATH = r"C:\Program Files\Siemens\NX1980\NXBIN\run_journal.exe" # path to simens nx

def prepare_files_for_measurement(source_folder, folder_name):
    dest_folder = os.path.join(MEASUREMENT_INPUT_DIR, folder_name)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    files_to_copy = {
        "top.png": "top.png",
        "left.png": "side.png" 
    }
    files_found = False

    for src_name, dst_name in files_to_copy.items():
        src_path = os.path.join(source_folder, src_name)
        dst_path = os.path.join(dest_folder, dst_name)
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                print(f"   [COPY] {src_name} -> {dst_path}")
                files_found = True
            except Exception as e:
                print(f"   [ERROR] Failed to copy {src_name}: {e}")
            print(f"   [WARNING] {src_name} missing in {source_folder}")
    return files_found, dest_folder


def create_bat_file(folder_path, folder_name):
    bat_file_path = os.path.join(folder_path, "process_pipeline.bat")
    specific_output = os.path.join(r"D:\Automtion\Mes_output", folder_name)
    target_top = os.path.join(MEASUREMENT_INPUT_DIR, folder_name, "top.png")
    target_side = os.path.join(MEASUREMENT_INPUT_DIR, folder_name, "side.png")
    # Define directories
    specific_output = os.path.join(r"D:\Automtion\Mes_output", folder_name)
    merge_output_dir = r"D:\Automtion\Merge_Output"
    nx_script_dir = r"D:\Automtion\Merge_Output_NX"
    bat_content = f"""
@echo off
title Processing Pipeline - {folder_name}
call conda activate {CONDA_ENV}

echo.
echo      STEP 1: RUNNING ML PREDICTION

cd /d "{ML_SCRIPT_DIR}"
python "{ML_SCRIPT_NAME}" --mode test --test_dir "{folder_path}"

echo.

echo      STEP 2: RUNNING MEASUREMENT

python "{MEASURE_SCRIPT_PATH}" --top "{target_top}" --side "{target_side}" --output "{specific_output}"

echo.

echo      STEP 3: MERGING RESULTS

:: We call the merge script. 
:: Since it monitors folders, we run it once to grab the latest JSONs created in Steps 1 & 2.
python "{MERGE_SCRIPT_PATH}"


echo.
echo      STEP 4: GENERATING NX JOURNAL
python "{NX_CONVERT_PATH}"

echo.
echo      STEP 5: EXECUTING NX BUILD

:: 1. Go to the folder where we saved the NX Python script
cd /d "{nx_script_dir}"

:: 2. Find the latest python file in that folder
for /f "tokens=*" %%i in ('dir /b /od /a-d *.py') do set LATEST_NX_PY=%%i

:: 3. Run it using the FULL PATH to run_journal.exe
echo Executing Journal: %LATEST_NX_PY%
"{NX_JOURNAL_EXE_PATH}" "%LATEST_NX_PY%"

echo.
echo      STEP 4: GENERATING ONSHAPE PYTHON
python "{ONSHAPE_CONVERT_PATH}"

echo.
echo      STEP 5: EXECUTING ONSHAPE BUILD
:: Use the actual folder path, not the script path
cd /d "D:\Automtion\Merge_Output"

:: Find the newest .py file in THIS folder and run it
for /f "tokens=*" %%i in ('dir /b /od /a-d *.py') do set LATEST_PY=%%i

echo.
echo      STEP 6: EXPORTING TO SOLIDWORKS
:: Wait 3 seconds to ensure Onshape finishes its internal save
timeout /t 3 >nul
python "{SOLIDWORKS_EXPORT_PATH}"

echo.
echo              COMPLETED

echo Final CAD file should be in 
pause
    """

    with open(bat_file_path, "w") as f:
        f.write(bat_content)
    return bat_file_path
def process_new_folders():
    print(f"--- Monitoring {WATCH_DIR} ---")
    processed_folders = set()
    # Inside process_new_folders() loop:

    # Pre-fill processed list
    if os.path.exists(WATCH_DIR):
        for item in os.listdir(WATCH_DIR):
            full_path = os.path.join(WATCH_DIR, item)
            if os.path.isdir(full_path):
                processed_folders.add(full_path)

    while True:
        try:
            current_folders = []
            if os.path.exists(WATCH_DIR):
                for item in os.listdir(WATCH_DIR):
                    full_path = os.path.join(WATCH_DIR, item)
                    if os.path.isdir(full_path):
                        current_folders.append(full_path)

            for folder_path in current_folders:
                if folder_path not in processed_folders:
                    folder_name = os.path.basename(folder_path)
                    print(f"\n[NEW DETECTED] Processing: {folder_name}")
                    time.sleep(1)

                    # 1. MOVE & RENAME FILES
                    files_ready, _ = prepare_files_for_measurement(folder_path, folder_name)

                    # 2. CREATE & RUN BATCH FILE
                    # Inside process_new_folders() loop:
                    if files_ready:
                        bat_path = create_bat_file(folder_path, folder_name)  # Added folder_name
                        os.system(f'start "" "{bat_path}"')
                        print(f"[DONE] Pipeline launched for {folder_name}")
                    else:
                        print("[SKIP] Could not find top/left images to copy.")

                    processed_folders.add(folder_path)

            time.sleep(2)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    if not os.path.exists(WATCH_DIR): os.makedirs(WATCH_DIR)
    if not os.path.exists(MEASUREMENT_INPUT_DIR): os.makedirs(MEASUREMENT_INPUT_DIR)
    process_new_folders()
