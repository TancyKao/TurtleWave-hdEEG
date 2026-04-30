#!/usr/bin/env python3
"""
Add artifact annotations from CSV to Wonambi XML file.

This script reads artifact annotations from a CSV file and adds them to an existing
Wonambi XML annotation file. Each row in the CSV represents one artifact event.

CSV Format:
- Column 1: event_start (start time in seconds)
- Column 2: event_end (end time in seconds)

The script will:
1. Load the existing XML file
2. Create an "Artefact" event_type if it doesn't exist
3. Add each CSV row as an event with:
   - event_start: first column
   - event_end: second column
   - event_chan: (all)
   - event_qual: Good
"""

import csv
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime


def add_csv_artifacts_to_xml(csv_file, xml_file, output_xml=None):
    """
    Add artifact events from CSV to Wonambi XML annotation file.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file containing artifact annotations (start, end times)
    xml_file : str
        Path to existing Wonambi XML annotation file
    output_xml : str or None
        Output XML file path. If None, will overwrite the input XML.
    
    Returns:
    --------
    str : Path to the output XML file
    """
    
    # Check if files exist
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"XML file not found: {xml_file}")
    
    # Load existing XML
    print(f"Loading XML file: {xml_file}")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        print(f"✓ Loaded XML successfully")
    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file: {e}")
    
    # Find or create the events element
    events_elem = root.find(".//events")
    if events_elem is None:
        print("Creating new events structure...")
        # Look for rater element
        rater_elem = root.find(".//rater")
        if rater_elem is None:
            # Create rater element
            dataset_elem = root.find(".//dataset")
            if dataset_elem is None:
                dataset_elem = ET.SubElement(root, "dataset")
            
            rater_elem = ET.SubElement(dataset_elem, "rater")
            rater_elem.set("name", "Anon")
            now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            rater_elem.set("created", now)
            rater_elem.set("modified", now)
            ET.SubElement(rater_elem, "bookmarks")
        
        # Create events element
        events_elem = ET.SubElement(rater_elem, "events")
        print("✓ Created events structure")
    
    # Find or create the "Artefact" event_type element
    artefact_type_elem = None
    for elem in events_elem.findall("event_type"):
        if elem.get("type") == "Artefact":
            artefact_type_elem = elem
            break
    
    if artefact_type_elem is None:
        print("Creating new 'Artefact' event_type...")
        artefact_type_elem = ET.SubElement(events_elem, "event_type")
        artefact_type_elem.set("type", "Artefact")
        print("✓ Created 'Artefact' event_type")
    else:
        print("✓ Found existing 'Artefact' event_type")
    
    # Read CSV file and add events
    print(f"\nReading CSV file: {csv_file}")
    event_count = 0
    
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        
        # Skip header if present (check if first row contains non-numeric data)
        first_row = next(csv_reader, None)
        if first_row is None:
            print("Warning: CSV file is empty")
            return xml_file
        
        # Check if first row is header
        try:
            float(first_row[0])
            # First row is numeric, process it
            rows = [first_row]
        except (ValueError, IndexError):
            # First row is header, skip it
            print(f"Skipping header row: {first_row}")
            rows = []
        
        # Add remaining rows
        rows.extend(csv_reader)
        
        # Process each row
        for row_num, row in enumerate(rows, start=1):
            if len(row) < 2:
                print(f"Warning: Row {row_num} has fewer than 2 columns, skipping: {row}")
                continue
            
            try:
                start_time = float(row[0])
                end_time = float(row[1])
            except ValueError as e:
                print(f"Warning: Row {row_num} has invalid numeric values, skipping: {row}")
                continue
            
            # Create event element
            event_elem = ET.SubElement(artefact_type_elem, "event")
            
            # Add start time
            start_elem = ET.SubElement(event_elem, "event_start")
            start_elem.text = str(start_time)
            
            # Add end time
            end_elem = ET.SubElement(event_elem, "event_end")
            end_elem.text = str(end_time)
            
            # Add channel (all)
            chan_elem = ET.SubElement(event_elem, "event_chan")
            chan_elem.text = "(all)"
            
            # Add quality
            qual_elem = ET.SubElement(event_elem, "event_qual")
            qual_elem.text = "Good"
            
            event_count += 1
    
    print(f"✓ Added {event_count} artifact events")
    
    # Determine output path
    if output_xml is None:
        output_xml = xml_file
        print(f"\nOverwriting original XML file: {output_xml}")
    else:
        print(f"\nSaving to new XML file: {output_xml}")
    
    # Create pretty formatted XML string
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Save the XML file
    with open(output_xml, 'w') as f:
        f.write(pretty_xml)
    
    print(f"✓ XML file saved successfully")
    
    return output_xml


if __name__ == "__main__":
    # SET YOUR FILE LOCATIONS
    
    # Path to CSV file with artifact annotations
    csv_file = "/Volumes/Tancy_storage/MCI002_BL/artfact_anno.csv"
    
    # Path to existing XML annotation file
    xml_file = "/Volumes/Tancy_storage/MCI002_BL/wonambi/MCI002_BL_clean_rebuilt.xml"
    
    # Output XML file (None = overwrite original, or specify new path)
    #output_xml = None  # Set to None to overwrite, or specify a new path like:
    output_xml = "/Volumes/Tancy_storage/MCI002_BL/wonambi/MCI002_BL_clean_rebuilt_with_artifacts.xml"
    
    # Add artifacts to XML
    try:
        result_xml = add_csv_artifacts_to_xml(
            csv_file=csv_file,
            xml_file=xml_file,
            output_xml=output_xml
        )
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Artifacts added to XML file:")
        print(result_xml)
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        raise
