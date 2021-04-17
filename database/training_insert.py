import logging
import connections
from sqlalchemy.sql import select

def db_model_insert(model_name,epochs,start_lr,m_type,num_outputs,loss_fn):
    db, table = connections.make_db_connection('rji.models')
    data = {'model_name':model_name, 'epochs':epochs, 'start_lr':start_lr
            'm_type':m_type, 'num_outputs':num_outputs, 'loss_fn':loss_fn
            }
    result = db.execute(table.insert()values(data))
    if not result:
        logging.info('Model Insert Failed')
        raise Exception('Model Insertion Failed')
    return get_modelid()

def get_modelid():
    db, table = connections.make_db_connection('rji.models')
    s = select(table.c.model_id)
    result = db.execute(s)
    if not result:
        logging.info('Model ID Retrieval Failed')
        raise Eception('Model ID Retrieval Failed')
    return result[-1]
    
    