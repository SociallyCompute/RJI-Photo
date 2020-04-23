import sys
# from ..common import helpers
from misc import write_xmp_color_class
import logging

logging.basicConfig(filename='logs/fill_db.log', 
                    filemode='w', level=logging.DEBUG)

write_xmp_color_class()
