import os
import glob

def delete_pth_files(root_folder):
    # 获取所有 .pth 文件的路径
    pth_files = glob.glob(os.path.join(root_folder, '**', '*.pth'), recursive=True)
    
    for file_path in pth_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")

# 指定你的根文件夹路径
root_folder_path = '/home/jq/Code/CdSC/results'
delete_pth_files(root_folder_path)
    
    