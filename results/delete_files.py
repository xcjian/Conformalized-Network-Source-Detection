import os

def delete_specific_files(folder_path):
    """
    Delete files starting with 'set_prec' or 'set_call' in the specified folder and all its subfolders.
    
    Args:
        folder_path (str): Path to the folder to search for files
    """
    try:
        # Ensure the path exists and is a directory
        if not os.path.isdir(folder_path):
            print(f"Error: {folder_path} is not a valid directory")
            return

        # Define patterns for files to delete
        patterns = ['set_prec*', 'set_recall*']
        
        # Walk through directory and subdirectories
        for root, _, files in os.walk(folder_path):
            for pattern in patterns:
                # Filter files matching the pattern
                for file_name in files:
                    if file_name.startswith(('set_prec', 'set_recall')):
                        file_path = os.path.join(root, file_name)
                        try:
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")
                        except OSError as e:
                            print(f"Error deleting {file_path}: {e}")

        print("Deletion process completed")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage
    folder_path = input("Enter the folder path: ")
    delete_specific_files(folder_path)