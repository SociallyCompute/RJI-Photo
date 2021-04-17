CREATE SCHEMA rji;

ALTER SCHEMA rji OWNER TO rji;

SET default_tablespace = '';

SET default_with_oids = false;

CREATE TABLE rji.photos(
    photo_id BIGSERIAL PRIMARY KEY NOT NULL,
    photo_fname character varying,
    ranking int NOT NULL,
    create_date timestamp(0) without time zone DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE rji.models(
    model_id BIGSERIAL PRIMARY KEY NOT NULL,
    model_name character varying NOT NULL,
    epochs int NOT NULL,
    start_lr float NOT NULL,
    m_type character varying NOT NULL,
    num_ouputs int NOT NULL,
    loss_fn character varying NOT NULL
);

CREATE TABLE rji.losses(
    loss_id BIGSERIAL PRIMARY KEY NOT NULL,
    model_id bigint NOT NULL,
    epoch int NOT NULL,
    train_loss float NOT NULL,
    validation_loss float NOT NULL,
    CONSTRAINT model_id
        FOREIGN KEY(model_id)
            REFERENCES models(model_id)
            ON DELETE SET NULL
);

-- INSERT INTO photos(
--     photo_id,
--     photo_fname,
--     ranking
-- ) 
-- VALUES (
--     val1,
--     val2,
--     val3
-- )