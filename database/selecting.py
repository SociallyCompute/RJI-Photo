import logging
from sqlalchemy.sql import select
from sqlalchemy import inspect

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from database import inserting
from database import connections

def get_modelid():
    db, table = connections.make_db_connection('models')
    s = select([table.c.model_id])
    result = db.execute(s).fetchall()
    print((result[-1])[0])
    if not result:
        logging.info('Model ID Retrieval Failed')
        raise Exception('Model ID Retrieval Failed')
    db.dispose()
    return (result[-1])[0]

def get_lossid():
    db, table = connections.make_db_connection('losses')
    s = select([table.c.loss_id])
    result = db.execute(s).fetchall()
    if not result:
        logging.info('Loss ID Retrieval Failed')
        raise Exception('Loss ID Retrieval Failed')
    db.dispose()
    return result[-1]

def get_photoid():
    db, table = connections.make_db_connection('photos')
    s = select([table.c.photo_id])
    result = db.execute(s).fetchall()
    print((result[-1])[0])
    if not result:
        logging.info('Model ID Retrieval Failed')
        raise Exception('Model ID Retrieval Failed')
    db.dispose()
    return (result[-1])[0]