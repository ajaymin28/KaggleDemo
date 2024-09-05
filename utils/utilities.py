import os
from os import listdir
import random
import glob

def getrandomIndexesForImages(gallery_image_count=10, query_image_count=5, max_idx_of_images=150):
    """
    DO NOT CHANGE THE SEEDS
    """
    # Fixed seed for repetitive results (DO NOT CHANGE)
    const_seed_q = 200
    const_seed_g = 100
    print(f"Q seed: {const_seed_q} G seed: {const_seed_g}")
    # Final number of values 
    g_number = gallery_image_count
    q_number = query_image_count
    # Seed and retrieve the values
    random.seed(const_seed_q)
    query_indexes = [random.randint(0, max_idx_of_images) for i in range(0, q_number)]

    # Seed and retrieve the values
    random.seed(const_seed_g)
    gallery_indexes = [random.randint(0, max_idx_of_images) for i in range(0, g_number)]

    # print(query_indexes)
    # print(gallery_indexes)
    return gallery_indexes, query_indexes

def getClassesNames(gallery_images_path,query_images_path):
    return list(set(listdir(gallery_images_path) + listdir(query_images_path)))

def getData(images_path, classes_names):
    classes_idnames = {}
    classes_nameids = {}
    images_paths = []
    image_labels = []
    for classid, cls_name in enumerate(classes_names):
        class_path = os.path.join(images_path, cls_name, "*.jpg")
        class_images_paths = glob.glob(class_path)
        for ip in class_images_paths:
            images_paths.append(ip)
            image_labels.append(cls_name)

        if classid not in classes_idnames.keys():
            classes_idnames[classid] = cls_name
            classes_nameids[cls_name] = classid
    return images_paths, image_labels, classes_idnames, classes_nameids


def getImagePaths(gallery_images_path, query_images_path, gallery_image_count=10, query_image_count=5, max_idx_of_images=150):
    gallery_idxs, query_idxs = getrandomIndexesForImages(gallery_image_count=gallery_image_count, query_image_count=query_image_count,max_idx_of_images=max_idx_of_images)
    classes_names = listdir(gallery_images_path)
    print("classes",classes_names)

    images_paths = {}
    for cls_name in classes_names:
        if cls_name not in images_paths.keys():
            images_paths[cls_name] = {"gallery": [], "query": []}
        class_path = os.path.join(gallery_images_path, cls_name, "*.jpg")
        class_images_paths = glob.glob(class_path)
        for gal_idx in gallery_idxs:
            images_paths[cls_name]["gallery"].append(class_images_paths[gal_idx])

        class_path_q = os.path.join(query_images_path, cls_name, "*.jpg")
        class_images_paths_q = glob.glob(class_path_q)
        for que_idx in query_idxs:
            images_paths[cls_name]["query"].append(class_images_paths_q[que_idx])
            
    return images_paths