import numpy as np
import os
import math
from collections import Counter

def load_from_directory(directory, target_names, max_files_in_one_category=30000):
    file_texts = []
    targets = []
    
    for root, dirs, files in os.walk(directory):
        number_used_files = 0
        for name in files:
            if number_used_files < max_files_in_one_category:
                file_path = os.path.join(root, name)
                try:
                    with open(file_path) as file:
                
                        # считываем файл построчно
                        file_text = ""
                        for line in file.readlines():
                            file_text += (line + " ")
                        if file_text != "" and file_text != " ":
                            file_texts.append(file_text)
                
                            _, cur_target = os.path.split(root)
                            cur_target = target_names.index(cur_target)
                            targets.append(cur_target)
                
                            number_used_files += 1            
                except:
                    print("PROBLEM WITH FILE", file_path)
            
    return file_texts, targets


def divide_to_train_and_test(X, y, num_categories, part_to_test=0.1):

    # In X and y firstly should be elements form catefory 0, than elements from catefory 1 etc
    
    y_train = []
    y_test = []
    indices_to_X_train = []
    indices_to_X_test = []
    
    elem_idx = 0
    counter = Counter(y)
    
    for category_id in range(num_categories):
        # count of elements in category <category_id>
        category_elems_count = counter[category_id] 
        test_elements_count = math.ceil(category_elems_count * part_to_test)
        train_elements_count = category_elems_count - test_elements_count
        
        for categories_elem_idx in range(category_elems_count):
            if train_elements_count > 0:
                indices_to_X_train.append(elem_idx)
                y_train.append(y[elem_idx])
                train_elements_count -= 1
            else:
                indices_to_X_test.append(elem_idx)
                y_test.append(y[elem_idx])
            elem_idx += 1
        
    return np.array(X[indices_to_X_train]), np.array(y_train), np.array(X[indices_to_X_test]), np.array(y_test)

def one_hot(y, classes_count):
    return np.eye(classes_count)[y]