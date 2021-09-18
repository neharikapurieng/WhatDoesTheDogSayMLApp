from flask import Flask
from flask import render_template

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route('/')
def my_form():
    print("go home")
    return render_template("processing.html") # this should be the name of your html file