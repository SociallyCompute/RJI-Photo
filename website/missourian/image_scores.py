from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from flask_login import login_required
from werkzeug.exceptions import abort

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
from website.missourian.tables import Photos, Base

bp = Blueprint('image_scores', __name__)

IMAGES_PER_PAGE = 12


@bp.route('/')
def index():
    names = []
    page = request.args.get('page', 1, type=int)
    photo_page = Photos.query.paginate(page=page, per_page=IMAGES_PER_PAGE)
    for photo in photo_page.items:
        names.append(photo.photo_fname.split('/')[-1])
    return render_template('image_scores/index.html', photos=photo_page, names=names)


@bp.route('/create', methods=('GET', 'POST'))
@login_required
def create():
    if request.method == 'POST':
        fname = request.form['fname']
        ranking = request.form['ranking']
        error = None

        if not fname:
            error = 'File is required.'

        if error is not None:
            flash(error)
        else:
            photo = Photos(photos_fname=fname, ranking=ranking)
            Base.session.add(photo)
            Base.session.commit()
            return redirect(url_for('image_scores.index'))

    return render_template('image_scores/create.html')


def get_photo(photo_id):
    photos = Photos.query.filter_by(photo_id=photo_id).first()

    if photos is None:
        abort(404, f"Photo id {photo_id} doesn't exist.")

    return photos


@bp.route('/<int:id>/update', methods=('GET', 'POST'))
@login_required
def update(photo_id):
    post = get_photo(photo_id)

    if request.method == 'POST':
        name = request.form['name']
        ranking = request.form['ranking']
        error = None

        if not name:
            error = 'Name is required.'
        elif not ranking:
            error = 'Ranking is required.'

        if error is not None:
            flash(error)
        else:
            photo = Photos.query.filter_by(id=photo_id).first()
            photo.photo_fname = name
            photo.ranking = ranking
            Base.session.commit()
            return redirect(url_for('image_scores.index'))

    return render_template('image_scores/update.html', post=post)


@bp.route('/<int:id>/delete', methods=('POST',))
@login_required
def delete(photo_id):
    photo_id = get_photo(photo_id)
    photo = Photos.query.filter_by(photo_id=photo_id).first()
    Base.session.delete(photo)
    Base.session.commit()
    return redirect(url_for('image_scores.index'))
