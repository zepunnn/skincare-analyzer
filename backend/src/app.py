from flask import Flask

app=Flask(__name__)

@app.route('/')
def index():
    return '<h1>ALAMAK</h1>'


@app.route('/about')
def about():
    return '<h1>CIHUYYY</h1>'


app.run (debug=True)