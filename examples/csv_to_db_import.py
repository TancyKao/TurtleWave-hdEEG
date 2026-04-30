"""
csv_to_db_import.py
This script imports existing CSV parameter files into the neural_events.db database.
It's designed for users who already have CSV files from slow wave or spindle detection
and want to import them into the SQLite database for further analysis.

Supports processing multiple subjects in batch mode.

Modules:
    - turtlewave_hdEEG: Custom library for processing EEG events and database operations.

Functions:
    - import_csv_to_database: Imports CSV parameter files to SQLite database

Workflow:
    1. Define root directory and list of subject folders to process.
    2. For each subject:
        a. Define file paths for CSV directories and database location.
        b. Initialize database connection and create tables if needed.
        c. Find all CSV parameter files in the specified directories.
        d. Import each CSV file into the database with duplicate checking.
    3. Report import statistics and completion status for all subjects.
"""

import os
import sys
import glob
from turtlewave_hdEEG import ParalSWA
import logging

# 1. Define the file paths
root_dir = "/Volumes/Sleep/Sleep/2. STAFF/Tancy/OSA CPAP Events/"  # Root directory containing subject folders

# List of subject folders to process 
subject_folders = [
    "sub-03RA",
    # "sub-04RB",  
    # "sub-05RC",
]

# 2. Define which event types to import
import_slow_waves = True   # Set to True to import slow wave parameters
import_spindles = True     # Set to True to import spindle parameters

# 3. CSV file patterns for each event type
sw_pattern = "*sw_parameters*.csv"      # Slow wave parameter files
spindle_pattern = "*spindle_parameters*.csv"  # Spindle parameter files

print("=== CSV to Database Import Tool (Multi-Subject) ===")
print(f"Root Directory: {root_dir}")
print(f"Subjects to process: {', '.join(subject_folders)}")
print(f"Import Slow Waves: {import_slow_waves}")
print(f"Import Spindles: {import_spindles}")
print()

# 4. Create a dummy dataset for ParalSWA initialization
class DummyDataset:
    def __init__(self):
        self.header = {'s_freq': 500}  # default hdEEG sampling rate

dummy_dataset = DummyDataset()

# 5. Initialize ParalSWA processor (reused for all subjects)
print("Initializing database processor...")
event_processor = ParalSWA(
    dataset=dummy_dataset, 
    annotations=None
)

# 6. Process each subject
overall_total_added = 0
overall_total_updated = 0
overall_total_skipped = 0
overall_successful_imports = 0
overall_failed_imports = 0
overall_subjects_processed = 0
overall_subjects_failed = 0

for subject_idx, subject_folder in enumerate(subject_folders, 1):
    print("\n" + "="*60)
    print(f"PROCESSING SUBJECT {subject_idx}/{len(subject_folders)}: {subject_folder}")
    print("="*60)
    
    # Define paths for this subject
    subject_dir = os.path.join(root_dir, subject_folder)
    db_path = os.path.join(subject_dir, "wonambi", "neural_events.db")
    sw_csv_dir = os.path.join(subject_dir, "wonambi", "sw_results")
    spindle_csv_dir = os.path.join(subject_dir, "wonambi", "spindle_results")
    
    # Check if subject directory exists
    if not os.path.isdir(subject_dir):
        print(f"✗ Subject directory not found: {subject_dir}")
        overall_subjects_failed += 1
        continue
    
    print(f"Subject Directory: {subject_dir}")
    print(f"Database Path: {db_path}")
    
    # Verify directories exist and collect CSV files for this subject
    csv_files = []
    directories_to_process = []
    
    if import_slow_waves:
        if os.path.isdir(sw_csv_dir):
            sw_files = glob.glob(os.path.join(sw_csv_dir, sw_pattern))
            csv_files.extend(sw_files)
            directories_to_process.append(("Slow Wave", sw_csv_dir, len(sw_files)))
            print(f"✓ Slow wave directory found: {sw_csv_dir} ({len(sw_files)} files)")
        else:
            print(f"✗ Slow wave directory not found: {sw_csv_dir}")
        
    if import_spindles:
        if os.path.isdir(spindle_csv_dir):
            spindle_files = glob.glob(os.path.join(spindle_csv_dir, spindle_pattern))
            csv_files.extend(spindle_files)
            directories_to_process.append(("Spindle", spindle_csv_dir, len(spindle_files)))
            print(f"✓ Spindle directory found: {spindle_csv_dir} ({len(spindle_files)} files)")
        else:
            print(f"✗ Spindle directory not found: {spindle_csv_dir}")
    
    if not csv_files:
        print(f"\n✗ No CSV files found for {subject_folder}")
        overall_subjects_failed += 1
        continue
    
    # Create database directory if needed
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        print(f"Created database directory: {db_dir}")
    
    # Initialize/create the database for this subject
    print("\nSetting up database...")
    try:
        event_processor.initialize_sqlite_database(db_path)
        print(f"✓ Database initialized: {db_path}")
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        overall_subjects_failed += 1
        continue
    
    # Display found files by type
    print(f"\nFound {len(csv_files)} CSV files for {subject_folder}:")
    for event_type, directory, count in directories_to_process:
        if count > 0:
            print(f"\n{event_type} files ({count}):")
            pattern = sw_pattern if event_type == "Slow Wave" else spindle_pattern
            files = glob.glob(os.path.join(directory, pattern))
            for i, csv_file in enumerate(files, 1):
                print(f"  {i}. {os.path.basename(csv_file)}")
    
    print(f"\n=== Starting Import for {subject_folder} ===")
    
    # Import each CSV file for this subject
    subject_total_added = 0
    subject_total_updated = 0
    subject_total_skipped = 0
    subject_successful_imports = 0
    subject_failed_imports = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        
        # Determine event type from file path
        if "sw_results" in csv_file or "sw_parameters" in filename:
            event_type = "Slow Wave"
        elif "spindle_results" in csv_file or "spindle_parameters" in filename:
            event_type = "Spindle"
        else:
            event_type = "Unknown"
        
        print(f"\n[{i}/{len(csv_files)}] Importing {event_type}: {filename}")
        
        try:
            # Import CSV to database
            result = event_processor.import_parameters_csv_to_database(
                csv_file=csv_file,
                db_path=db_path,
                append=True  # Don't overwrite existing entries
            )
            
            if result and "error" not in result:
                added = result.get("added", 0)
                updated = result.get("updated", 0)
                skipped = result.get("skipped", 0)
                empty_channels = result.get("empty_channels", 0)
                
                print(f"  ✓ Success - Added: {added}, Updated: {updated}, Skipped: {skipped}")
                if empty_channels > 0:
                    print(f"    Note: {empty_channels} channels had no events")
                
                subject_total_added += added
                subject_total_updated += updated
                subject_total_skipped += skipped
                subject_successful_imports += 1
                
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                print(f"  ✗ Failed: {error_msg}")
                subject_failed_imports += 1
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            subject_failed_imports += 1
    
    # Subject summary
    print(f"\n--- Summary for {subject_folder} ---")
    print(f"CSV files processed: {len(csv_files)}")
    print(f"Successful imports: {subject_successful_imports}")
    print(f"Failed imports: {subject_failed_imports}")
    print(f"Events added: {subject_total_added}")
    print(f"Events updated: {subject_total_updated}")
    print(f"Events skipped: {subject_total_skipped}")
    
    # Update overall totals
    overall_total_added += subject_total_added
    overall_total_updated += subject_total_updated
    overall_total_skipped += subject_total_skipped
    overall_successful_imports += subject_successful_imports
    overall_failed_imports += subject_failed_imports
    
    if subject_successful_imports > 0:
        overall_subjects_processed += 1
        print(f"✓ {subject_folder} completed successfully!")
    else:
        overall_subjects_failed += 1
        print(f"✗ {subject_folder} had no successful imports")

# Overall final summary
print("\n" + "="*60)
print("OVERALL IMPORT SUMMARY (ALL SUBJECTS)")
print("="*60)
print(f"Total subjects attempted: {len(subject_folders)}")
print(f"Subjects processed successfully: {overall_subjects_processed}")
print(f"Subjects failed: {overall_subjects_failed}")
print()
print(f"Total CSV files processed: {overall_successful_imports + overall_failed_imports}")
print(f"Successful imports: {overall_successful_imports}")
print(f"Failed imports: {overall_failed_imports}")
print()
print(f"Total events added to databases: {overall_total_added}")
print(f"Total events updated: {overall_total_updated}")
print(f"Total events skipped (duplicates): {overall_total_skipped}")

if overall_subjects_processed > 0:
    print("\n✓ Import completed successfully for at least one subject!")
    print("You can now use the turtlewave_gui for PAC analysis.")
else:
    print("\n✗ No subjects were imported successfully.")
    print("Please check the error messages above and verify your file paths.")

print("="*60)
