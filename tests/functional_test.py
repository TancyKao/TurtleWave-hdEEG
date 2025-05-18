# functional_test.py

import os
import sys
import tempfile
import shutil
import numpy as np
from turtlewave_hdEEG import LargeDataset, XLAnnotations, ParalEvents, CustomAnnotations

def create_synthetic_data():
    """Create minimal synthetic data for testing"""
    # This is a placeholder - in a real test, you'd create actual EEG-like data
    # For now, we'll just create directory structure
    test_dir = tempfile.mkdtemp()
    wonambi_dir = os.path.join(test_dir, "wonambi")
    os.makedirs(wonambi_dir)
    
    # Create a channels.csv file
    with open(os.path.join(test_dir, "channels.csv"), 'w') as f:
        f.write("channel\nE101\nE102\nE103\n")
    
    return test_dir

def cleanup(test_dir):
    """Clean up test directory"""
    shutil.rmtree(test_dir)

def minimal_functional_test():
    """Run a minimal functional test"""
    print("Running minimal functional test...")
    
    # Create test directory and synthetic data
    test_dir = create_synthetic_data()
    
    try:
        # Test basic class initialization
        print("\nTesting class initialization:")
        
        # Test CustomAnnotations
        custom_annot = CustomAnnotations()
        print("✓ CustomAnnotations initialized")
        
        # Test ParalEvents with minimal args
        events = ParalEvents()
        print("✓ ParalEvents initialized")
        
        # Test utility functions
        channels = turtlewave_hdEEG.utils.read_channels_from_csv(
            os.path.join(test_dir, "channels.csv")
        )
        print(f"✓ read_channels_from_csv returned: {channels}")
        
        print("\nMinimal functional tests passed!")
        
    except Exception as e:
        print(f"✗ Error in functional test: {e}")
    finally:
        # Clean up test directory
        cleanup(test_dir)

if __name__ == "__main__":
    minimal_functional_test()