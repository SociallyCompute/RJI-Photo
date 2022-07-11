from flask import Flask, current_app
from flask_sqlalchemy import SQLAlchemy

from sqlalchemy import Integer, String, Column, Float, DateTime
from sqlalchemy.sql.schema import ForeignKey

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

Base = SQLAlchemy()


class Losses(Base.Model):
    __tablename__ = 'losses'

    loss_id = Column(Integer, primary_key=True)
    model_id = Column(Integer(), ForeignKey('models.model_id'))
    epoch = Column(Integer())
    train_loss = Column(Float())
    validation_loss = Column(Float())

    def __repr__(self):
        return f"<Loss {self.loss_id}>"


class Models(Base.Model):
    __tablename__ = 'models'

    model_id = Column(Integer, primary_key=True)
    model_name = Column(String())
    epochs = Column(Integer())
    start_lr = Column(Float())
    m_type = Column(String())
    num_outputs = Column(Integer())
    loss_fn = Column(String())
    filepath = Column(String())

    def __repr__(self):
        return f"<Model {self.model_name}>"


class Photos(Base.Model):
    __tablename__ = 'photos'

    photo_id = Column(Integer, primary_key=True)
    photo_fname = Column(String())
    ranking = Column(Integer())
    create_date = Column(DateTime())
    model_id = Column(Integer(), ForeignKey('models.model_id'))

    def __repr__(self):
        return f"<Photo: {self.photo_fname} Ranking: {self.ranking}>"


class Users(Base.Model):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(), unique=True)
    password = Column(String())

    def is_active(self):
        """True, as all users are active."""
        return True

    def get_id(self):
        """Return the email address to satisfy Flask-Login's requirements."""
        return self.id

    def is_authenticated(self):
        """Return True if the user is authenticated."""
        return True

    def is_anonymous(self):
        """False, as anonymous users aren't supported."""
        return False

    def __repr__(self):
        return f"<Name {self.username}>"
