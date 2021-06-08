import logging

from sqlalchemy import MetaData, create_engine, pool
from sqlalchemy.ext.automap import automap_base

import sys, os
sys.path.append(os.path.split(sys.path[0])[0])
from config_files import db_config

def make_db_connection(table_name):
    """ Makes a connection to the database used to store each of the testing values. Allows for 
            standardization of test values to recieve a decent test result

    :param table_name - name of the table to be reflected in SQLAlchemy metadata

    :rtype: (sqlalchemy engine, sqlalchemy table) reference to database engine and 
        specified table
    """
    logging.info("Connecting to database: {}".format(db_config.DB_STR))

    dbschema = 'rji'
    db = create_engine(db_config.DB_STR, poolclass=pool.NullPool, pool_pre_ping=True,
        connect_args={'options': '-csearch_path={}'.format(dbschema)})
    metadata = MetaData()
    metadata.reflect(db, only=[table_name])
    Base = automap_base(metadata=metadata)
    Base.prepare()
    r_table = Base.classes[table_name].__table__
    # class_name = Base.classes[table_name]

    logging.info("Database connection successful")
    return db, r_table