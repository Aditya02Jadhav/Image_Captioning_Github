from flask import Flask, render_template, request
import os

import loaded_model

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def marks():
    if request.method == 'POST':
        f = request.files['userfile']
        path = "./static/{}".format(f.filename)
        f.save(path) 

        caption = hello_moto.caption_this_image(path)
        # print(caption)
        result_dic = {
            'image' : path,
            'caption' : caption
        }

        # Ensure the 'uploads' folder exists in your project directory
        # uploads_dir = os.path.join(app.root_path, r'static')
        # os.makedirs(uploads_dir, exist_ok=True)

        # Save the file with its original name
        # path = os.path.join(uploads_dir, f.filename)
        # f.save(path)

    return render_template("index.html", your_result =result_dic)

if __name__ == "__main__":
    app.run(debug=True)
