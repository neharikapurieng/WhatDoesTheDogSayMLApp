from flask import Flask
from flask import request
from flask import render_template
from google.cloud import datastore

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route('/')
def my_form():
    print("go home")
    return render_template("home.html") # this should be the name of your html file

@app.route('/store', methods=['POST'])
def my_form_post():
    print("hello2")
    print(request.form)
    af = request.form['user_af']
    print(af)
    print("Storing af")

    # # Instantiates a client
    # datastore_client = datastore.Client()
    #
    # # The kind for the new entity
    # kind = "AudioEntries"
    # # The name/ID for the new entity
    # # The Cloud Datastore key for the new entity
    # task_key = datastore_client.key(kind)
    #
    # # Prepares the new entity
    # audio_entry = datastore.Entity(key=task_key)
    # audio_entry["audio_file"] = af
    #
    # # Saves the entity
    # datastore_client.put(audio_entry)

    return render_template("processing.html")
