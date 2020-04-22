from common import helpers
import logging

logging.basicConfig(filename='logs/fill_db.log', 
                    filemode='w', level=logging.DEBUG)

helpers.write_xmp_color_class()