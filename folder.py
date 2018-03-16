import os


def getFolderList():
    folder_list = [name for name in os.listdir(".") if os.path.isdir(name)]
    folder_list = [i.split("_")[-1] for i in folder_list]
    species_code = [i.split("_")[0] for i in folder_list]
    return folder_list, species_code

DATASET_DIRECTORY = "dataset"
def getSpeciesList():
    idx = 0
    species_dict = {} # strukturnya {kategori: spesies}, butuh saran struktur yang baik
    for dir in os.listdir(DATASET_DIRECTORY):
        species_dict[idx] = dir.split("_")[-1]
        idx += 1
    return species_dict

print(getSpeciesList())