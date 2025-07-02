#!/usr/bin/env python3

# Convert EEG events from JSON to XML
from turtlewave_hdEEG.addjson2xml import convert_json_to_xml

# SET YOUR FILE LOCATIONS 

# Main folder cotaining xml and json files
root_folder = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/OSA_BL06AC/wonambi/"

# Folder where your JSON files
json_folder = root_folder + "spindle_results/"

# XML annotation file (should be in the root folder)
existing_xml = root_folder + "OSA_BL06AC_clean_rebuilt.xml"

# List of JSON files you want to convert 
json_files = [
    json_folder + "spindles_Ferrarelli2007_9.0-12.0Hz_NREM2_E17.json",
    json_folder + "spindles_Ferrarelli2007_9.0-12.0Hz_NREM2_E18.json",
]

# Folder to save the new XML file
save_folder = json_folder

# "slowwave" or "spindle"
event_type = "spindle"

# Convert JSON to XML
output_xml = convert_json_to_xml(
    json_files=json_files,
    output_dir=save_folder,
    existing_xml=existing_xml if existing_xml else None,
    event_type=event_type
)

print("=" * 50)
print("✅ Done! Converted XML saved to:")
print(output_xml)
print("=" * 50)
