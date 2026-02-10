from flask import Flask, request, render_template
# Import the translation function from your pretrained model


from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Load the saved model
model2 = MBartForConditionalGeneration.from_pretrained('Transmodel3')

# Load the tokenizer
tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50', src_lang="en_XX")

# The text you want to translate
def translator1(w):
  text = w

  # Tokenize the text
  input_ids = tokenizer(text, return_tensors="pt").input_ids

  # Generate the translated text
  translated = model2.generate(input_ids)

  # Decode the translated text
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text[0]                                                                                                                                                                        


app = Flask(__name__)

# Render the input form
@app.route('/', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        user_text = request.form['text']
        print('user_text:',user_text)
        # Call the translation function
        translated_text = translator1(user_text)
        print(translated_text)
        return render_template('result.html', translated_text=translated_text, user_text=user_text)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
















