from flask import Flask
from flask import render_template

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route('/')
def my_form():
    print("go home")
    return render_template("home.html") # this should be the name of your html file


@app.route('/record', methods=['POST'])
def go_home():
    print("recording")
    return render_template("record.html") # this should be the name of your html file