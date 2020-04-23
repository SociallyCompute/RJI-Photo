import sys
# from ..common import helpers
from common import misc
import logging

logging.basicConfig(filename='logs/fill_db.log', 
                    filemode='w', level=logging.DEBUG)

misc.write_xmp_color_class()