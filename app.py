import os
from flask import Flask, render_template, request
from pagewhiz import get_answer_from_url
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    url = ""
    question = ""
    if request.method == 'POST':
        url = request.form.get('url')
        question = request.form.get('question')

        if url and question:
            answer = get_answer_from_url(url, question)
        else:
            answer = "Please provide both a URL and a question."

    return render_template('index.html', answer=answer, url=url, question=question)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)