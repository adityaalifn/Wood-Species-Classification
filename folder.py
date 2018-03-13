import os


def getFolderList():
    folder_list = [name for name in os.listdir(".") if os.path.isdir(name)]
    folder_list = [i.split("_")[1] for i in folder_list]
    species_code = [i.split("_")[0] for i in folder_list]
    return folder_list, species_code
