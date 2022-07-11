import logging

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from old_codes.database import selecting
from old_codes.database import connections


def insert_model(model_name,epochs,start_lr,m_type,num_outputs,loss_fn, filepath):
    db, table = connections.make_db_connection('models')
    result = db.execute(table.insert().values(
        model_name = str(model_name),
        epochs = int(epochs),
        start_lr = float(start_lr),
        m_type = str(m_type),
        num_outputs = int(num_outputs),
        loss_fn = str(loss_fn),
        filepath = str(filepath)
    ))

    if not result:
        logging.info('Model Insert Failed')
        raise Exception('Model Insertion Failed')
    db.dispose()
    return selecting.get_modelid()
    
def insert_loss(modelid, epoch, train_loss, val_loss):
    db, table = connections.make_db_connection('losses')
    data = {
        'model_id':modelid, 'epoch':epoch, 'train_loss':train_loss, 
        'validation_loss':val_loss
    }
    result = db.execute(table.insert().values(data))
    
    if not result:
        logging.info('Loss Insert Failed')
        raise Exception('Loss Insertion Failed')
    db.dispose()
    return selecting.get_lossid()

def insert_photos(photo_fname, ranking):
    db, table = connections.make_db_connection('photos')
    model_id = selecting.get_modelid()
    result = db.execute(table.insert().values(
        photo_fname = str(photo_fname),
        ranking = int(ranking),
        model_id = int(model_id)
    ))

    if not result:
        logging.info('Photo Insert Failed')
        raise Exception('Photo Insertion Failed')
    db.dispose()
    return selecting.get_modelid()
    