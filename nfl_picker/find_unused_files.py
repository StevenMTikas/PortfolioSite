#!/usr/bin/env python3
"""
Find unused files in the NFL Picker project.
"""

import os
import sys
import ast
from pathlib import Path

def get_python_files(directory):
    """Get all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def get_imports_from_file(file_path):
    """Extract import statements from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def find_unused_files():
    """Find unused files in the project."""
    project_root = Path(".")
    src_dir = project_root / "src"
    
    # Get all Python files
    all_files = get_python_files(str(project_root))
    src_files = get_python_files(str(src_dir))
    
    print("=== NFL Picker Project File Analysis ===")
    print(f"Total Python files: {len(all_files)}")
    print(f"Source files: {len(src_files)}")
    print()
    
    # Core files that should be kept
    core_files = {
        'run_gui.py',
        'src/nfl_picker/main.py',
        'src/nfl_picker/gui_app.py',
        'src/nfl_picker/crew.py',
        'src/nfl_picker/config.py',
        'src/nfl_picker/database.py',
        'src/nfl_picker/stats_database.py',
        'src/nfl_picker/focused_analysis.py',
        'src/nfl_picker/utils.py',
        'src/nfl_picker/ssl_fix.py',
        'src/nfl_picker/config/agents.yaml',
        'src/nfl_picker/config/tasks.yaml',
        'src/nfl_picker/tools/sportsdata_api_script.py',
        'src/nfl_picker/tools/custom_tool.py',
        'src/nfl_picker/tools/serper_tool.py',
        'src/nfl_picker/data_collectors/api_collector.py',
        'src/nfl_picker/data_collectors/web_scraper.py',
        'src/nfl_picker/enhanced_web_scraper.py',
        'src/nfl_picker/update_stats.py',
        'src/nfl_picker/run_web_scraping.py'
    }
    
    # Files that appear to be sample/test data generators
    sample_data_files = [
        'src/nfl_picker/create_sample_data.py',
        'src/nfl_picker/create_team_sample_data.py', 
        'src/nfl_picker/create_realistic_sample_data.py',
        'src/nfl_picker/create_realistic_roster_data.py',
        'src/nfl_picker/create_current_roster_stats.py',
        'src/nfl_picker/create_starting_qb_stats.py',
        'src/nfl_picker/player_team_mapping.py'
    ]
    
    # Test files
    test_files = [
        'tests/test_nfl_picker_simple.py',
        'tests/test_nfl_picker.py', 
        'tests/test_serper_integration.py'
    ]
    
    # Example files
    example_files = [
        'examples/usage_example.py'
    ]
    
    # Documentation files
    doc_files = [
        'doc/README.md',
        'doc/TEMPLATE.md',
        'doc/create_doc.py',
        'doc/development/BEFORE_AFTER_COMPARISON.md',
        'doc/development/COMPLETE_CHANGES_SUMMARY.md',
        'doc/development/OPTIMIZATION_SUMMARY.md',
        'doc/features/STATS_DATABASE.md',
        'doc/features/TIMESTAMP_FEATURE.md',
        'doc/features/TIMESTAMP_IMPLEMENTATION_SUMMARY.md',
        'doc/user-guides/README.md',
        'doc/user-guides/USAGE_GUIDE.md',
        'STATS_DATABASE_IMPLEMENTATION_SUMMARY.md'
    ]
    
    # Log files
    log_files = [
        'logs/stats_update.log',
        'logs/web_scraping.log'
    ]
    
    # Output files (analysis results)
    output_files = [
        'output/NFL_Analysis_Arizona_Cardinals_vs_Indianapolis_Colts_Week_06.txt',
        'output/NFL_Analysis_Buffalo_Bills_vs_Atlanta_Falcons_Week_06.txt',
        'output/NFL_Analysis_Denver_Broncos_vs_New_Jersey_Jets_Week_06.txt',
        'output/NFL_Analysis_Detriot_Lions_vs_Kansas_City_Chiefs_Week_06.txt',
        'output/NFL_Analysis_Los_Angeles_Chargers_vs_Miami_Dolphins_Week_06.txt',
        'output/NFL_Analysis_Los_Angeles_Ram_vs_Baltimore_Ravens_Week_06.txt',
        'output/NFL_Analysis_New_England_Patriots_vs_New_Orleans_Saints_Week_06.txt',
        'output/NFL_Analysis_Philadelphia_Eagles_vs_New_York_Giants_Week_06.txt',
        'output/NFL_Analysis_San_Francisco_49ers_vs_Tampa_Bay_Buccaneers_Week_06.txt',
        'output/NFL_Analysis_Seattle_Seahawks_vs_Jacksonville_Jaguars_Week_06.txt'
    ]
    
    # Knowledge files
    knowledge_files = [
        'knowledge/user_preference.txt'
    ]
    
    print("=== POTENTIALLY UNUSED FILES ===")
    print()
    
    print("1. SAMPLE DATA GENERATORS (can be removed if not needed):")
    for file in sample_data_files:
        if os.path.exists(file):
            print(f"   - {file}")
    print()
    
    print("2. TEST FILES (can be removed if not needed):")
    for file in test_files:
        if os.path.exists(file):
            print(f"   - {file}")
    print()
    
    print("3. EXAMPLE FILES (can be removed if not needed):")
    for file in example_files:
        if os.path.exists(file):
            print(f"   - {file}")
    print()
    
    print("4. DOCUMENTATION FILES (can be removed if not needed):")
    for file in doc_files:
        if os.path.exists(file):
            print(f"   - {file}")
    print()
    
    print("5. LOG FILES (can be removed to clean up):")
    for file in log_files:
        if os.path.exists(file):
            print(f"   - {file}")
    print()
    
    print("6. OUTPUT FILES (old analysis results, can be removed):")
    for file in output_files:
        if os.path.exists(file):
            print(f"   - {file}")
    print()
    
    print("7. KNOWLEDGE FILES (can be removed if not needed):")
    for file in knowledge_files:
        if os.path.exists(file):
            print(f"   - {file}")
    print()
    
    print("=== CORE FILES (KEEP THESE) ===")
    for file in core_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
    print()
    
    print("=== RECOMMENDATIONS ===")
    print("1. Remove sample data generators if you don't need them")
    print("2. Remove test files if you don't need them") 
    print("3. Remove example files if you don't need them")
    print("4. Remove documentation files if you don't need them")
    print("5. Remove log files to clean up")
    print("6. Remove output files (old analysis results)")
    print("7. Remove knowledge files if you don't need them")
    print()
    print("Keep all core files for the application to work properly.")

if __name__ == "__main__":
    find_unused_files()
