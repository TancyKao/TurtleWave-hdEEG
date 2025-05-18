#!/usr/bin/env python
"""
json_to_annotations.py - Add spindles from JSON to Wonambi annotation file

This script loads spindle data from a JSON file and adds selected channels
to a Wonambi annotation XML file.

Usage:
    python json_to_annotations.py --json spindle_results.json --xml annotations.xml --channels E101,E102,E103

"""

import argparse
import json
import os
from wonambi.attr import Annotations, create_annotation

def add_spindles_to_annotations(json_file, xml_file, channels, output_xml=None):
    """
    Add spindles from JSON to Wonambi annotations.
    
    Parameters
    ----------
    json_file : str
        Path to JSON file containing spindle data
    xml_file : str
        Path to Wonambi XML annotation file
    channels : list
        List of channels to add to annotations
    output_xml : str or None
        Output XML file path. If None, will overwrite the input XML.
    """
    # Load JSON data
    print(f"Loading spindle data from {json_file}")
    try:
        with open(json_file, 'r') as f:
            spindles = json.load(f)
        print(f"Loaded {len(spindles)} spindles from JSON")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return False
    
    # Load XML annotations
    print(f"Loading annotations from {xml_file}")
    try:
        annotations = Annotations(xml_file)
        print(f"Loaded annotations successfully")
    except Exception as e:
        print(f"Error loading XML annotations: {e}")
        return False
    
    # Filter spindles for selected channels
    print(f"Filtering spindles for channels: {channels}")
    selected_spindles = [sp for sp in spindles if sp['chan'] in channels]
    print(f"Found {len(selected_spindles)} spindles for selected channels")
    
    # Add spindles to annotations
    spindle_count = 0
    for sp in selected_spindles:
        try:
            # Create annotation
            annot = create_annotation(
                name='spindle',
                time=(sp['start'], sp['end']),
                chan=[sp['chan']],
                qual=sp.get('method', '')
            )
            
            # Add to annotations
            annotations.add_annotations(annot)
            spindle_count += 1
        except Exception as e:
            print(f"Error adding spindle: {e}")
    
    print(f"Added {spindle_count} spindles to annotations")
    
    # Save annotations
    output_path = output_xml if output_xml else xml_file
    try:
        annotations.save(output_path)
        print(f"Saved annotations to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving annotations: {e}")
        return False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Add spindles from JSON to Wonambi annotation file')
    
    parser.add_argument('--json', required=True,
                        help='Path to JSON file containing spindle data')
    
    parser.add_argument('--xml', required=True,
                        help='Path to Wonambi XML annotation file')
    
    parser.add_argument('--channels', required=True,
                        help='Comma-separated list of channels to add (e.g., C3,C4,F3,F4)')
    
    parser.add_argument('--output', default=None,
                        help='Output XML file path. If not specified, will overwrite the input XML.')
    
    args = parser.parse_args()
    
    # Convert channels to list
    channels = [ch.strip() for ch in args.channels.split(',')]
    
    # Add spindles to annotations
    success = add_spindles_to_annotations(args.json, args.xml, channels, args.output)
    
    if success:
        print("Spindles were successfully added to annotations")
    else:
        print("Failed to add spindles to annotations")

if __name__ == "__main__":
    main()