import numpy as np
import os
import math

import skimage
from skimage.io import imread, imsave
from skimage.transform import resize

from collections import Counter

def transform_current_image(cur_X, dimension, size):
    """
    cur_X - numpy array, current image
    dimension - размерность изображений, может быть любым целым положительным числом, 
        но имеет смысл задавать 3, 4 и 2 (серое изображение)
    size - боковой размер изображения (изображение будет приведено к этому размеру, станет квадратным)
    
    returns: cur_X - numpy array, is_error - true if error occured
    """
    
    is_error = False
    
    # меняем 3й размер, если необходимо
    if dimension == 2:
        if len(cur_X.shape) != 2:
            try:
                cur_X = skimage.color.rgb2gray(cur_X)
            except:
                print("CANNOT CONVERT RGB TO GRAY")
                is_error = True
                #continue
    else:
        if len(cur_X.shape) < 3:
            try:
                cur_X = skimage.color.gray2rgb(cur_X)
            except:
                print("CANNOT CONVERT GRAY TO RGB")
                #continue
                is_error = True
            
    # Боковой размер изображеня делаем равным size
    if dimension != 2:
        cur_X = resize(cur_X, (size, size, dimension))
    else:
        cur_X = resize(cur_X, (size, size))
        
    if dimension != 2 and len(cur_X.shape) == 2:
        is_error = True
    
    return cur_X, is_error

def preprocess_and_save_images(source_root, target_root, dimension, size, max_num_files_for_each_category=30000):
    """
    source_root - root of directories with source images
    target_root - root of directories were should be preprocessed images
    dimension - размерность изображений, может быть любым целым положительным числом, 
        но имеет смысл задавать 3, 4 и 2 (серое изображение)
    size - боковой размер изображения (изображение будет приведено к этому размеру, станет квадратным)
    max_num_files_for_each_category - максимальное число файлов, которое будет считываться для каждой категории
    """
     # проверка того, что параметр dimension - положительное число
    assert(dimension >= 2)
    
    for root, dirs, files in os.walk(source_root):
        number_used_files = 0
        for name in files: 
            if number_used_files < max_num_files_for_each_category:
            
                file_path_source = os.path.join(root, name)
                # файлы типа pdf - игнорируем (временное решение, возможно, стоит исправить впоследствии)
                file_extension = os.path.splitext(file_path_source)[1].lower()
                if file_extension == ".pdf" or file_extension == ".tiff" or file_extension == ".tif":
                    continue                    
            
                cur_X = imread(file_path_source)
                _, cur_y = os.path.split(root)
                
                # to solve blue pictures problem
                if (file_extension == ".jpg" or file_extension == ".jpeg") and dimension == 3:
                    cur_X = cur_X[:, :, :3]
                
                cur_directory_path_target = os.path.join(target_root, cur_y)
                # check if such directory exists (if not - create)
                if not os.path.exists(cur_directory_path_target):
                    try:
                        os.makedirs(cur_directory_path_target)
                    except:
                        print("CANNOT CREATE DIRECTORY")
                        continue
                file_path_target = os.path.join(cur_directory_path_target, name)
            
                # change dimension size and all such things if necessary
                cur_X, is_error = transform_current_image(cur_X, dimension, size)
                if(is_error):
                    continue
            
                #print(cur_X.shape)
                
                try:
                    imsave(file_path_target, cur_X)
                    number_used_files += 1
                except e:
                    print(e)    
    
    return 

def load_dir(directory, target_names, already_transformed=True, dimension=None, size=None, max_num_files_for_each_category=30000):
    """
    directory - директория, файлы из которой необходимо загрузить
    target_names - названия категорий (соответствующие номера в из этого списка будут элементами в y)
    already_transformed - True if there is no nesessity in any transformation of images
    dimension - размерность изображений, может быть любым целым положительным числом, 
        но имеет смысл задавать 3, 4 и 2 (серое изображение)
    size - боковой размер изображения (изображение будет приведено к этому размеру, станет квадратным)
    max_num_files_for_each_category - максимальное число файлов, которое будет считываться для каждой категории
    
    returns: X - список numpy array's каждый из которых представляет собой отдельно взятое изображение
             y - target'ы для соответствующих элементов из X
    """
    
    X = []
    y = []
    
    for root, dirs, files in os.walk(directory):
        number_used_files = 0
        for name in files: 
            if number_used_files < max_num_files_for_each_category:
            
                file_path = os.path.join(root, name)
                # файлы типа pdf - игнорируем (временное решение, возможно, стоит исправить впоследствии)
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension == ".pdf" or file_extension == ".tiff" or file_extension == ".tif":
                    continue
            
                cur_X = imread(file_path)
                _, cur_y = os.path.split(root)
            
                # сравниваем считанный cur_y с названиями из target_names
                cur_y = target_names.index(cur_y)
            
                # change dimension size and all such things if necessary
                if not already_transformed:
                    cur_X, is_error = transform_current_image(cur_X, dimension, size)
                    if(is_error):
                        continue
                
                try:
                    X.append(cur_X)
                    y.append(cur_y)
                    number_used_files += 1
                except e:
                    print(e)
                
    return X, y

def divide_to_train_end_test(X, y, num_categories, part_to_test=0.1, dimension_is_not_two=True):

    # In X and y firstly should be elements form catefory 0, than elements from catefory 1 etc
    
    
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    elem_idx = 0
    counter = Counter(y)
    #print(counter)
    
    for category_id in range(num_categories):
        # count of elements in category <category_id>
        category_elems_count = counter[category_id] 
        test_elements_count = math.ceil(category_elems_count * part_to_test)
        train_elements_count = category_elems_count - test_elements_count
        
        #print(train_elements_count, test_elements_count)
        
        for categories_elem_idx in range(category_elems_count):
            if dimension_is_not_two and len(X[elem_idx].shape) == 2:
                elem_idx += 1
                if train_elements_count > 0:
                    train_elements_count -= 1
                continue
            if train_elements_count > 0:
                X_train.append(X[elem_idx])
                y_train.append(y[elem_idx])
                train_elements_count -= 1
            else:
                X_test.append(X[elem_idx])
                y_test.append(y[elem_idx])
            elem_idx += 1
            
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        
    
    