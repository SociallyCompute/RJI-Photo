import sys, os

sys.path.append(os.path.split(sys.path[0])[0])
from common import datasets
from common import config
from common import connections
import logging
    
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
