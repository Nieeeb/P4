import os
import datetime
from collections import defaultdict
import json
import random
import shutil
from tqdm import tqdm
import threading

class DataThingAMagic():
    def __init__(self, period_begin: str, period_end: str, 
                 data_directory: str, json_directory: str,
                 total_files: int,
                 data_split: str):
        """
        Format of period: YYYYMMDD, so 26. of august 2021: 20210826
        Total_files: number of files you want for the specified period
        data_split: Is it train or val or test you're processing

        """
        self.period_begin = period_begin
        self.period_end = period_end
        self.data_directory = data_directory
        self.json_directory = json_directory
        self.total_files = total_files
        self.data_split = data_split
        
    def relevantFiles(self) -> list:
        """
        Finds the files in the period
        """
        matching_files = []
        # Iterate over each file in the directory
        print("Finding files for the period")
        for filename in tqdm(os.listdir(self.data_directory)):
            # Check if the file has the desired extension (.jpg or .txt)
            if filename.lower().endswith(('.jpg')):
                # Extract the date part from the beginning of the filename (first 8 characters)
                file_date = filename[:8]
                # Check if the date is within the specified range
                if self.period_begin <= file_date <= self.period_end:
                    matching_files.append(filename)
        return matching_files

    def getMetaData(self) -> dict:
        """
        Returns a dictionary has the file name and the time it was captured in the format: "2020-05-14T22:40:46"
        """
        mapping = {}
        file_list = self.relevantFiles()

        with open(self.json_directory, 'r') as f:
            json_data = json.load(f)
        
        print("Gathering metadata")
        for image in tqdm(json_data.get("images", [])):
            # print(image)
            
            # Retrieve the original file_name from JSON
            original_file = image.get("file_name", "")
            
            # Replace "/" with "_" to form a comparable string
            transformed = original_file.replace("/", "_")

            # Remove leading 'frames_' if it exists
            if transformed.startswith("frames_"):
                transformed = transformed[len("frames_"):]
            # Remove the file extension (e.g., ".jpg")
            # transformed = transformed.rsplit('.', 1)[0]

            # Check if the transformed name is in the provided file list
            if transformed in file_list:
                mapping[transformed] = image.get("date_captured")
                
        return mapping
    
    def select_equal_distribution_files(self) -> list:
        """
        Selects a total number of files (total_files) from the given file_date_mapping so that 
        the files are evenly distributed by day and, within each day, across hours.
        
        The file_date_mapping is expected to be a dict where:
        - key: a file name (e.g. "20200514_clip_21_2239_image_0050")
        - value: a timestamp string in the format "YYYY-MM-DDTHH:MM:SS"
        
        The function works as follows:
        1. Group the files by day (using the date part of the timestamp).
        2. Distribute the total_files evenly across these days.
            For example, if there are 2 days and total_files is 10, each day should supply 5 files.
        3. For each day, group the files by the hour (extracted from the timestamp).
            For each hour group, the earliest file is chosen (based on the complete timestamp).
        4. Then, for the day, select one file from each hour in ascending order (i.e. first hour, 
            second hour, etc.) until the day's quota is met. If a day has fewer unique hours than required, 
            only those available will be selected.
        
        Parameters:
            file_date_mapping (dict): Mapping of file names to capture timestamps.
            total_files (int): The total number of files to select.
            
        Returns:
            dict: A dictionary mapping the selected file names to their capture timestamp.
        """

        file_date_mapping  = self.getMetaData()
        # Group files by day (YYYY-MM-DD)
        day_groups = defaultdict(list)
        print("Selecting equal number of files pr. day in the period")
        for file_name, timestamp in tqdm(file_date_mapping.items()):
            # Extract date portion from the timestamp (first 10 characters)
            day = timestamp[:10]
            
            # Extract hour from timestamp (characters 11-13) and convert to int
            hour = int(timestamp[11:13])
            # Also parse the full datetime for proper ordering if needed
            dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
            day_groups[day].append((file_name, hour, dt, timestamp))
        
        # Sort the days in chronological order
        days_sorted = sorted(day_groups.keys())
        num_days = len(days_sorted)
        if num_days == 0:
            return {}
        
        # Distribute total_files evenly across days.
        # Each day gets at least 'base' files; the first 'remainder' days get one extra.
        base = self.total_files // num_days
        
        remainder = self.total_files % num_days
        
        selected = {}
        print("Selecting equal number of files pr. hour in the period")
        for i, day in tqdm(enumerate(days_sorted)):
            # Number of files to select for this day
            n_files = base + (1 if i < remainder else 0)
            
            # For this day, group files by hour.
            # hour_dict maps hour -> list of (file_name, dt, timestamp) tuples.
            hour_dict = defaultdict(list)
            for file_name, hour, dt, ts in day_groups[day]:
                hour_dict[hour].append((file_name, dt, ts))
                # print(len(hour_dict[hour]))
            
            # Get available hours in ascending order.
            available_hours = sorted(hour_dict.keys())
            
            count = 0
            # Use a round-robin selection: loop over the hours and select one random file from each
            # until we meet the day's quota (n_files) or run out of files.
            while count < n_files:
                selection_made = False
                for hr in available_hours:
                    if count >= n_files:
                        break
                    if hour_dict[hr]:
                        # Randomly choose one file from this hour
                        chosen = random.choice(hour_dict[hr])
                        hour_dict[hr].remove(chosen)  # Remove it so it won't be chosen again
                        file_name, dt, ts = chosen
                        selected[file_name] = ts
                        count += 1
                        selection_made = True
                # If none of the hours had any files left, break out of the loop.
                if not selection_made:
                    break
        return list(selected.keys())

    def find_corresponding_txt_files(self) -> list:
        """
        Given a list of selected file names (typically .jpg files), this function searches
        for corresponding .txt files in the data_directory. For each .jpg file, it replaces
        the extension with .txt and checks if that file exists in the directory. Both the
        original .jpg file and its corresponding .txt file (if found) are added to the result list.
        
        Parameters:
            data_directory (str): Path to the directory containing the files.
            selected_files (list): List of selected file names (e.g. ['20200514_clip_21_2239_image_0072.jpg', ...]).
        
        Returns:
            list: A list of file names including the original .jpg files and their corresponding .txt files.
        """
        selected_files = self.select_equal_distribution_files()
        combined_files = selected_files
        print("Taking txt files with same name")

        for file in tqdm(selected_files):
            # Only process .jpg files
            if file.lower().endswith('.jpg'):
                # Replace the .jpg extension with .txt
                txt_file = file.rsplit('.', 1)[0] + '.txt'
                # Construct the full path to the txt file
                txt_path = os.path.join(self.data_directory, txt_file)
                # If the corresponding .txt file exists, add it to the list
                if os.path.exists(txt_path):
                    combined_files.append(txt_file)
        return combined_files

    def copy_files_to_temp_folder(self):
        """
        Copies all files specified in file_list from data_directory to a temporary folder.
        
        The temporary folder is created inside data_directory with the name provided by self.data_split.
        
        Parameters:
            data_directory (str): Path to the directory containing the original files.
            file_list (list): List of file names to be copied (e.g., those returned by find_corresponding_txt_files).
            
        Returns:
            str: The path to the temporary folder where the files were copied.
        """
        file_list = self.find_corresponding_txt_files()


        working_directory = os.getcwd()
        temp_folder = os.path.join(f"{working_directory}/Fata/Months", f"{self.data_split}")

        print(temp_folder)
        # Create the temporary folder if it does not exist.
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # Copy each file from the source directory to the temporary folder.
        print("Copying files to temp folder")
        for file_name in tqdm(file_list):
            source_path = os.path.join(self.data_directory, file_name)
            destination_path = os.path.join(temp_folder, file_name)
            try:
                shutil.copy2(source_path, destination_path)
            except Exception as e:
                print(f"Error copying {source_path} to {destination_path}: {e}")
        
        return temp_folder
   


import os

def write_absolute_jpg_paths(dataloader, output_txt_path):
    """
    Writes a text file where each line is the absolute path to a .jpg file.
    The paths are written with forward slashes so that the following code:
        filename.rstrip().split('/')[-1]
    correctly extracts the filename.
    
    Parameters:
        data_thing (DataThingAMagic): An instance of DataThingAMagic.
        output_txt_path (str): The path to the output text file.
    """
    # Get all jpg files from the data directory that meet the period criteria
    jpg_files = dataloader.relevantFiles()
    
    # Get the absolute path to the data directory
    abs_dir = os.path.abspath(dataloader.data_directory)
    
    with open(output_txt_path, 'w') as f:
        for file in jpg_files:
            # Build the absolute path to the jpg file
            abs_path = os.path.join(abs_dir, file)
            # Convert backslashes (Windows) to forward slashes
            abs_path = abs_path.replace('\\', '/')
            f.write(abs_path + "\n")

def write_relative_jpg_paths(dataloader, output_txt_path):
    """
    Writes a text file where each line is the relative path to a .jpg file.
    The paths are written with forward slashes so that the following code:
        filename.rstrip().split('/')[-1]
    correctly extracts the filename.
    
    Parameters:
        data_thing (DataThingAMagic): An instance of DataThingAMagic.
        output_txt_path (str): The path to the output text file.
    """
    # Get all jpg files from the data directory that meet the period criteria
    jpg_files = dataloader.relevantFiles()
    jpg_files.sort()
    
    # Get the absolute path to the data directory
    dir = dataloader.data_directory
    
    with open(output_txt_path, 'w') as f:
        for file in jpg_files:
            # Build the absolute path to the jpg file
            abs_path = os.path.join(dir, file)
            # Convert backslashes (Windows) to forward slashes
            abs_path = abs_path.replace('\\', '/')
            f.write(abs_path + "\n")



if __name__ == "__main__":
    dataloader = DataThingAMagic(period_begin="20100515", 
                                 period_end="20300521", 
                                 data_directory="Data/images/valid",
                                 json_directory="/home/nieb/Projects/Big Data/Images/Seasons_drift/v2/harborfrontv2/Valid.json",
                                 total_files=1000000000000,
                                 data_split="test1")


    write_relative_jpg_paths(dataloader, "Data/valid.txt")
