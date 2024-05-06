from flask import Flask, render_template, request
from transformers import AutoModelWithLMHead, AutoTokenizer

app = Flask(__name__, template_folder='templates')

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-common_gen")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-common_gen")

def gen_sentence(words, max_length=32):
    input_text = words
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(
        input_ids=features['input_ids'], 
        attention_mask=features['attention_mask'],
        max_length=max_length
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/generate_sentence", methods=["POST"])
def generate_sentence():
    input_words = request.form["input-words"]
    generated_sentence = gen_sentence(input_words)
    return render_template("index.html", generated_sentence=generated_sentence)

if __name__ == '__main__':
    app.run(debug=True)
