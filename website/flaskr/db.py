from flask import current_app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from config_files import db_config

db = SQLAlchemy(current_app)
migrate = Migrate(current_app, db)

def get_db():
    return db

class LossesModel(db.Model):
    __tablename__ = 'losses'

    loss_id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer())
    epoch = db.Column(db.Integer())
    train_loss = db.Column(db.Double())
    validation_loss = db.Column(db.Double())

    def __init__(self, model_id, epoch, train_loss, validation_loss, **kwargs):
        super(LossesModel, self).__init__(**kwargs)
        self.model_id = model_id
        self.epoch = epoch
        self.train_loss = train_loss
        self.validation_loss = validation_loss

    def __repr__(self):
        return f"<Loss {self.loss_id}>"

class ModelsModel(db.Model):
    __tablename__ = 'losses'

    model_id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String())
    epochs = db.Column(db.Integer())
    start_lr = db.Column(db.Double())
    m_type = db.Column(db.String())
    num_outputs = db.Column(db.Integer())
    loss_fn = db.Column(db.String())
    filepath = db.Column(db.String())

    def __init__(self, model_name, epochs, start_lr, m_type, num_outputs, loss_fn, filepath, **kwargs):
        super(ModelsModel, self).__init__(**kwargs)
        self.model_name = model_name
        self.epochs = epochs
        self.start_lr = start_lr
        self.m_type = m_type
        self.num_outputs = num_outputs
        self.loss_fn = loss_fn
        self.filepath = filepath

    def __repr__(self):
        return f"<Model {self.model_name}>"

class PhotosModel(db.Model):
    __tablename__ = 'losses'

    loss_id = db.Column(db.Integer, primary_key=True)
    photo_fname = db.Column(db.String())
    ranking = db.Column(db.Integer())
    create_date = db.Column(db.DateTime())

    def __init__(self, photo_fname, ranking, **kwargs):
        super(PhotosModel, self).__init__(**kwargs)
        self.photo_fname = photo_fname
        self.ranking = ranking

    def __repr__(self):
        return f"<Photo: {self.photo_fname} Ranking: {self.ranking}>"