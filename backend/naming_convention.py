import os
from pathlib import Path
import re

def standardize_filenames(base_dir):
    """
    Standardize filenames in the dataset directory to follow the convention:
    category_direction_number.extension
    e.g., Bike_L_001.png
    
    Args:
        base_dir (str): Path to the base directory containing category subdirectories
    """
    base_path = Path(base_dir)
    
    # Valid categories and their directions
    categories = {
        'ambulance': ['L', 'M', 'R'],
        'carhorns': ['L', 'M', 'R'],
        'FireTruck': ['L', 'M', 'R'],
        'policecar': ['L', 'M', 'R'],
        'Bike': ['L', 'R', 'B']
    }
    
    # Process each category directory
    for category, directions in categories.items():
        for direction in directions:
            dir_name = f"{category}_{direction}"
            category_dir = base_path / dir_name
            
            if not category_dir.exists():
                print(f"Directory {dir_name} not found, skipping...")
                continue
                
            print(f"\nProcessing directory: {dir_name}")
            
            # Get all files in the directory
            files = [f for f in category_dir.iterdir() if f.is_file()]
            
            # Counter for new filenames
            counter = 1
            
            # Process each file
            for file in files:
                # Skip if file is already properly named
                if re.match(f"^{category}_{direction}_\\d{{3}}{file.suffix}$", file.name):
                    print(f"File {file.name} already follows naming convention")
                    continue
                
                # Create new filename
                new_name = f"{category}_{direction}_{counter:03d}{file.suffix}"
                new_path = category_dir / new_name
                
                # Handle filename collisions
                while new_path.exists():
                    counter += 1
                    new_name = f"{category}_{direction}_{counter:03d}{file.suffix}"
                    new_path = category_dir / new_name
                
                # Rename file
                try:
                    file.rename(new_path)
                    print(f"Renamed: {file.name} → {new_name}")
                    counter += 1
                except Exception as e:
                    print(f"Error renaming {file.name}: {str(e)}")

def verify_naming_convention(base_dir):
    """
    Verify that all files follow the naming convention and report any issues.
    
    Args:
        base_dir (str): Path to the base directory containing category subdirectories
    """
    base_path = Path(base_dir)
    all_correct = True
    
    # Walk through all subdirectories
    for dir_path in base_path.iterdir():
        if not dir_path.is_dir():
            continue
            
        print(f"\nChecking directory: {dir_path.name}")
        
        # Expected pattern for this directory
        category_direction = dir_path.name
        pattern = f"^{category_direction}_\\d{{3}}\\.(png|jpg|jpeg|wav)$"
        
        # Check each file
        for file in dir_path.iterdir():
            if not file.is_file():
                continue
                
            if not re.match(pattern, file.name, re.IGNORECASE):
                print(f"Invalid filename found: {file.name}")
                all_correct = False
    
    if all_correct:
        print("\nAll files follow the naming convention!")
    else:
        print("\nSome files need to be renamed to follow the convention.")

def main():
    # Get the base directory from user input
    base_dir = input("Enter the path to your dataset directory: ").strip()
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Standardize filenames")
        print("2. Verify naming convention")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\nStandardizing filenames...")
            standardize_filenames(base_dir)
            print("\nFilename standardization complete!")
            
        elif choice == '2':
            print("\nVerifying naming convention...")
            verify_naming_convention(base_dir)
            
        elif choice == '3':
            print("\nExiting program...")
            break
            
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()