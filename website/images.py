import sqlite3
from flask import Flask, jsonify, g, redirect, request, url_for

app = Flask(__name__)

DB_PATH = 'database.db'
HOST = 'localhost'
PORT = 5001

@app.before_request
def before_request():
    g.db = sqlite3.connect(DB_PATH)

@app.teardown_request
def teardown_request(exception):
    if hasattr(g, 'db'):
        g.db.close()

@app.route('/')
def index():
    return redirect(url_for('static',filename='index.html'))

@app.route('/images/')
def image_info():
    get_items = request.args.get('get_items',2)

    cursor = g.db.execute('SELECT * FROM table WHERE ?', (get_items,))
    return jsonify(dict(('item%d' % i, item)
                    for i, item in enumerate(cursor.fetchall() start=1)))

if __name__ == '__main__':
    app.run(debug=True, host=HOST, port=PORT)
else:
    application = app