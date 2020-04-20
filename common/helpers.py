def find_size_bounds(limit_num_pictures=None):
    """ Will print and return min/max width/height of pictures in the dataset 
    
    :param limit_num_pictures - limits the number of pictures analyzed if you purposefully 
        want to work with a smaller dataset
    """
    data = ImageFolderWithPaths(data_dir)
    print(data[0][0].size)
    max_h = (data[0][0]).size[1]
    min_h = data[0][0].size[1]
    max_w = data[0][0].size[0]
    min_w = data[0][0].size[0]
    try:
        for (i, pic) in enumerate(data):
            #if we are limiting pics
            if limit_num_pictures:
                if i > limit_num_pictures:
                    break
            print(pic[0].size) # print all size dimensions
            
            # check width records
            if pic[0].size[0] > max_w:
                max_w = pic[0].size[0]
            elif pic[0].size[1] < min_w:
                min_w = pic[0].size[0]

            # check height records
            if pic[0].size[1] > max_h:
                max_h = pic[0].size[1]
            elif pic[0].size[1] < min_h:
                min_h = pic[0].size[1]
    except Exception as e:
        print(e)
        print("error occurred on pic {} number {}".format(pic, i))

    print("Max/min width: {} {}".format(max_w, min_w))
    print("Max/min height: {} {}".format(max_h, min_h))
    return min_w, max_w, min_h, max_h

    
def write_xmp_color_class(self):
    """ Write string containing Missourian labels to .txt files                
    """

    labels_file = open('Mar18_labeled_images.txt', 'w')
    none_file = open('Mar18_unlabeled_images.txt', 'w')
    i = 0

    for root, _, files in os.walk(config.MISSOURIAN_IMAGE_PATH, topdown=True):
        for name in files:
            if not name.endswith('.JPG', '.PNG'):
                continue

            with open(os.path.join(root, name), 'rb') as f:
                img_str = str(f.read())
                xmp_start = img_str.find('photomechanic:ColorClass')
                xmp_end = img_str.find('photomechanic:Tagged')
                if xmp_start != xmp_end and xmp_start != -1:
                    xmp_str = img_str[xmp_start:xmp_end]
                    if xmp_str[26] != '0':
                        labels_file.write(xmp_str[26] + '; ' + str(
                            os.path.join(root, name)) + '; ' + str(i) + '\n')
                    else:
                        none_file.write(xmp_str[26] + '; ' + str(
                            os.path.join(root, name)) + '; ' + str(i) + '\n')
                else:
                    none_file.write('0; ' + str(
                        os.path.join(root, name)) + '; ' + str(i) + '\n')
                i+=1

    labels_file.close()
    none_file.close()