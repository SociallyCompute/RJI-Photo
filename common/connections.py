import logging

import sqlalchemy as s
from sqlalchemy import MetaData
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import Column, Integer, String, ForeignKey, BigInteger, TIMESTAMP, Float
from sqlalchemy.orm import relationship

import sys, os
sys.path.append(os.path.split(sys.path[0])[0])
from common import config 

# class ClusterResults(Base):
#     __tablename__ = 'cluster_results'

#     cluster_result_id = Column(BigInteger, primary_key=True)
#     cluster_session_id = Column(ForeignKey('cluster_session.cluster_session_id'))
#     photo_path = Column(String)
#     cluster_number = Column(Integer)
#     date_collection_time = Column(TIMESTAMP)

#     cluster_session = relationship("ClusterSession", back_populates="cluster_results")

# class ClusterSession(Base):
#     __tablename__ = 'cluster_session'

#     cluster_session_id = Column(BigInteger, primary_key=True)
#     distance_between_points = Column(Float)
#     minimum_points = Column(Integer)
#     data_collection_time = Column(TIMESTAMP)

#     cluster_results = relationship("ClusterResults", back_populates="cluster_session")

# class Evaluation(Base):
#     __tablename__ = 'evaluation'

#     photo_id = Column(BigInteger, primary_key=True)
#     photo_path = Column(String)
#     model_score_1 = Column(Float)
#     model_score_2 = Column(Float)
#     model_score_3 = Column(Float)
#     model_score_4 = Column(Float)
#     model_score_5 = Column(Float)
#     model_score_6 = Column(Float)
#     model_score_7 = Column(Float)
#     model_score_8 = Column(Float)
#     model_score_9 = Column(Float)
#     model_score_10 = Column(Float)
#     data_collection_time = Column(TIMESTAMP)
#     training_epoch_id = Column(Integer)
#     model_scores = Column(Float)

# class Training(Base):
#     __tablename__ = 'training'

#     training_epoch_id = Column(BigInteger, primary_key=True)
#     te_dataset = Column(String)
#     te_learning_rate = Column(Float)
#     te_momentum = Column(Float)
#     te_accuracy = Column(Float)
#     te_model = Column(String)
#     te_epoch = Column(Integer)
#     te_batch_size = Column(Integer)
#     te_optimizer = Column(String)
#     te_indices = Column(Integer)
#     data_collection_time = Column(TIMESTAMP)
#     te_loss = Column(Float)

# class XmpColorClasses(Base):
#     __tablename__ = 'xmp_color_classes'

#     xmp_color_class_id = Column(BigInteger, primary_key=True)
#     photo_path = Column(String)
#     color_class = Column(Integer)
#     os_walk_index = Column(Integer)
#     data_collection_date = Column(TIMESTAMP)


# def make_db_connection():
#     logging.info("Connecting to database: {}".format(config.DB_STR))
#     dbschema = 'rji'
#     db = s.create_engine(config.DB_STR, poolclass=s.pool.NullPool, pool_pre_ping=True,
#         connect_args={'options': '-csearch_path={}'.format(dbschema)})

#     Base = automap_base()

#     Base.prepare()
#     logging.info("Database connection successful")
#     return db


def make_db_connection(table_name):
    """ Makes a connection to the database used to store each of the testing values. Allows for 
            standardization of test values to recieve a decent test result

    :param table_name - name of the table to be reflected in SQLAlchemy metadata

    :rtype: (sqlalchemy engine, sqlalchemy table) reference to database engine and 
        specified table
    """
    logging.info("Connecting to database: {}".format(config.DB_STR))

    dbschema = 'rji'
    db = s.create_engine(config.DB_STR, poolclass=s.pool.NullPool, pool_pre_ping=True,
        connect_args={'options': '-csearch_path={}'.format(dbschema)})
    metadata = MetaData()
    metadata.reflect(db, only=[table_name])
    Base = automap_base(metadata=metadata)
    Base.prepare()
    r_table = Base.classes[table_name].__table__
    # class_name = Base.classes[table_name]

    logging.info("Database connection successful")
    return db, r_table
