# Set-ExecutionPolicy Unrestricted -Scope CurrentUser
# .\env\Scripts\activate

from flask import Flask, render_template, request, json

from inference import infer, data
from model import initialize_model
import os, sys
from PIL import Image
import warnings
warnings.filterwarnings(action='ignore')
sys.setrecursionlimit(5000)


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        # img = Image.open(file)  # load with Pillow

        fig_html, fig_list_html = data(file)

        return render_template('result.html', plot=fig_html, plot2=fig_list_html)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))