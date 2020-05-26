import sys, os

sys.path.append(os.path.split(sys.path[0])[0])
from common import datasets
from common import config
from common import connections
import logging

def find_size_bounds(data_dir, limit_num_pictures=None):
    """ Will print and return min/max width/height of pictures in the dataset 
    
    :param limit_num_pictures - limits the number of pictures analyzed if you purposefully 
        want to work with a smaller dataset
    """
    data = datasets.ImageFolderWithPaths(data_dir)
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

    
def write_xmp_color_class():
    """ Write string containing Missourian labels to .txt files                
    """
    db, xmp_table = connections.make_db_connection('xmp_color_classes')
    i = 0
    logging.info('path: {}'.format(config.MISSOURIAN_IMAGE_PATH))
    for root, _, files in os.walk(config.MISSOURIAN_IMAGE_PATH, topdown=True):
        logging.info('root: {}\nfiles: {}'.format(root, files))
        for name in files:
            logging.info('name: {}\ntype: {}'.format(name, type(name)))
            if not name.endswith('.JPG') and not name.endswith('.PNG'):
                continue
            try:
                with open(os.path.join(root, name), 'rb') as f:
                    database_tuple = {}
                    img_str = str(f.read())
                    xmp_start = img_str.find('photomechanic:ColorClass')
                    xmp_end = img_str.find('photomechanic:Tagged')
                    if xmp_start != xmp_end and xmp_start != -1:
                        xmp_str = img_str[xmp_start:xmp_end]
                        database_tuple['color_class'] = int(xmp_str[26])
                        database_tuple['photo_path'] = str(os.path.join(root, name))
                        database_tuple['os_walk_index'] = i
                    else:
                        database_tuple['color_class'] = 0
                        database_tuple['photo_path'] = str(os.path.join(root, name))
                        database_tuple['os_walk_index'] = i
                    i+=1
                    result = db.execute(xmp_table.insert().values(database_tuple))
            except Exception as e:
                logging.info('Ran into error for {}\n...\nMoving on.\n'.format(e))

    logging.info('Finished writing xmp color classes to database')
    # labels_file.close()
    # none_file.close()

if __name__ == '__main__':
    logging.basicConfig(filename='fill_db.log', filemode='w', level=logging.DEBUG)
    write_xmp_color_class()
