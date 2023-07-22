import os


def get_wave_file_list(parent_folder, extensions=[".wav", ".flac"], check_for_features=False):
    """ traverses subfolders of parent_folder in search for files that match the given extension """

    file_list = []

    for root, dirs, files in os.walk(parent_folder, topdown=True):

        for file in files:

            stem, ext = os.path.splitext(file)

            #check for extension
            if not ext in extensions:
                continue

            # check if feature file exists
            if check_for_features and not os.path.isfile(os.path.join(root, stem + "_features.f32")):
                continue

            file_list.append(os.path.join(root, file))

    return file_list