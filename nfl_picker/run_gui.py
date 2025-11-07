#!/usr/bin/env python3
"""
NFL Picker GUI Launcher
Run the GUI application for team analysis
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Launch the NFL Picker GUI."""
    try:
        from nfl_picker.gui_app import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Error importing GUI: {e}")
        print("Make sure you have tkinter installed:")
        print("pip install tk")
        sys.exit(1)
    except Exception as e:
        print(f"Error running GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
