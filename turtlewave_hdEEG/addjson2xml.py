import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime

def convert_json_to_xml(json_files, output_dir='.', existing_xml=None, event_type='slowwave'):
    """
    Convert multiple JSON files containing events (slow waves or spindles) into a single XML file.
    Can append to an existing XML file if provided.
    
    Parameters:
    json_files (list): List of paths to JSON files
    output_dir (str): Directory to save the output XML file
    existing_xml (str): Path to existing XML file to append to (optional)
    event_type (str): Type of event to create ('slowwave' or 'spindle')
    """
    # Initialize the root XML element or load existing XML
    if existing_xml and os.path.exists(existing_xml):
        try:
            tree = ET.parse(existing_xml)
            root = tree.getroot()
            print(f"Loaded existing XML file: {existing_xml}")
        except ET.ParseError:
            print(f"Error parsing existing XML file: {existing_xml}")
            root = create_new_wonambi_xml()
    else:
        root = create_new_wonambi_xml()
    
    # Track unique channels to create filename
    channels = set()
    # Find or create the appropriate event_type element in the XML structure
    events_elem = root.find(".//events")
    if events_elem is None:
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
    
    # Find or create the event_type element for this event type
    event_type_elem = None
    for elem in events_elem.findall("event_type"):
        if elem.get("type") == event_type:
            event_type_elem = elem
            break
    
    if event_type_elem is None:
        event_type_elem = ET.SubElement(events_elem, "event_type")
        event_type_elem.set("type", event_type)

    
    # Extract existing channels from XML if it exists
    if existing_xml and os.path.exists(existing_xml):
        for event in root.findall('.//event'):
            chan_elem = event.find('event_chan')
            if chan_elem is not None and chan_elem.text:
                # Extract channel name 
                chan_text = chan_elem.text
                if '(' in chan_text:
                    chan = chan_text.split('(')[0].strip()
                    channels.add(chan)
                else:
                    channels.add(chan_text)
    
    # Process each JSON file
    for json_file in json_files:
        # Load the JSON data
        with open(json_file, 'r') as f:
            events = json.load(f)
        
        if not events:
            continue
        

        # Process each event in the JSON file
        for event in events:
            chan = event.get('chan')
            if chan:
                channels.add(chan)
            
            # Get start and end times
            start_time = event.get('start_time') or event.get('start')
            end_time = event.get('end_time') or event.get('end')
            
            if not start_time or not end_time:
                print(f"Warning: Event missing start/end time: {event}")
                continue
            
            # Create event element under the event_type element
            event_elem = ET.SubElement(event_type_elem, "event")
            
 
            # Add start time
            start_elem = ET.SubElement(event_elem, "event_start")
            start_elem.text = str(start_time)
            
            # Add end time
            end_elem = ET.SubElement(event_elem, "event_end")
            end_elem.text = str(end_time)
            
            # Add channel
            chan_elem = ET.SubElement(event_elem, "event_chan")
            if chan:
                chan_elem.text = chan
            else:
                chan_elem.text = "(all)"
            
            # Add quality
            qual_elem = ET.SubElement(event_elem, "event_qual")
            qual_elem.text = "Good"
    
   
    # Create the filename based on channels and input filenames
    # Extract channel names from JSON filenames if they match pattern
    if not channels:
        # Try to extract channel info from JSON filenames
        for json_file in json_files:
            basename = os.path.basename(json_file)
            if '_' in basename:
                # Extract text after the last underscore before .json
                parts = basename.split('_')
                potential_chan = parts[-1].split('.')[0]
                if potential_chan:
                    channels.add(potential_chan)
    
 
    chan_list = sorted(list(channels))
    
    if existing_xml and os.path.exists(existing_xml):
        # If we're appending to an existing file, start with that basename
        base_filename = os.path.splitext(os.path.basename(existing_xml))[0]
        
        # Add the channel info as a suffix
        if chan_list:
            if len(chan_list) <= 3:
                chan_str = "_" + "_".join(chan_list)
            else:
                chan_str = f"_{chan_list[0]}_plus_{len(chan_list)-1}_chans"
            
            annotation_filename = f"{base_filename}{chan_str}.xml"
        else:
            annotation_filename = os.path.basename(existing_xml)
    else:
       
    
        if len(chan_list) <= 3:
            chan_str = "_".join(chan_list)
        else:
            chan_str = f"{chan_list[0]}_plus_{len(chan_list)-1}_chans"
    
    # Create final filename
        annotation_filename = f"{event_type}_{chan_str}.xml"
    

    
    # Create a pretty formatted XML string
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Save the XML file
    output_path = os.path.join(output_dir, annotation_filename)
    with open(output_path, 'w') as f:
        f.write(pretty_xml)
    
    return output_path
def create_new_wonambi_xml():
    """Create a new Wonambi-compatible XML structure"""
    root = ET.Element("annotations")
    root.set("version", "5")
    
    # Create dataset element
    dataset = ET.SubElement(root, "dataset")
    
    # Add placeholder elements
    filename = ET.SubElement(dataset, "filename")
    filename.text = ""
    
    path = ET.SubElement(dataset, "path")
    path.text = ""
    
    start_time = ET.SubElement(dataset, "start_time")
    start_time.text = ""
    
    first_second = ET.SubElement(dataset, "first_second")
    first_second.text = "0"
    
    last_second = ET.SubElement(dataset, "last_second")
    last_second.text = "0"
    
    return root