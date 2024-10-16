import os

def list_all_files_in_subfolders(parent_directory):
    
    file_list = []
    
    try:
        for root, dirs, files in os.walk(parent_directory):
            for file in files:
                if file.endswith('.xml'):
                    file_name_without_extension = os.path.splitext(file)[0]
                    file_list.append(file_name_without_extension)
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return file_list

def process_all():
    
    parent_directory = '../data/trials'
    output_file = '../data/trials/all_trials.txt'
    file_list = list_all_files_in_subfolders(parent_directory)
    
    with open(output_file, 'w') as f:
        for file_name in file_list:
            f.write(file_name + '\n')

if __name__ == "__main__":
	process_all() 