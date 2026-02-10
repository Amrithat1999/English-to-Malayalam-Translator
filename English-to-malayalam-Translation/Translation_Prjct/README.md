# 🌐 English Translation Web App (Flask + mBART)

This project is a Flask-based web application that translates user-input
text using a pretrained mBART model from Hugging Face Transformers.

------------------------------------------------------------------------

## 🚀 Features

-   Web-based text input using Flask
-   Neural Machine Translation using mBART
-   Pretrained model loaded locally (`Transmodel3`)
-   Simple UI (home & result pages)
-   Runs in a Python virtual environment (`TranS`)

------------------------------------------------------------------------

## 🧠 Model Details

-   Model: MBartForConditionalGeneration\
-   Tokenizer: MBart50Tokenizer\
-   Base Model: facebook/mbart-large-50\
-   Source Language: English (`en_XX`)

------------------------------------------------------------------------

## 📂 Project Structure

    English-to-malayalam-Translation/
    │
    ├── k.py
    ├── Transmodel3/
    ├── templates/
    │   ├── home.html
    │   └── result.html
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

## 🐍 Environment Setup

### Create virtual environment

``` bash
python -m venv TranS
```

### Activate environment

**Windows**

``` bash
TranS\Scripts\activate
```

**Linux / macOS**

``` bash
source TranS/bin/activate
```

------------------------------------------------------------------------

## 📦 Install Dependencies

``` bash
pip install flask transformers torch sentencepiece
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run the Application

``` bash
python mainapp.py
```

Open browser:

    http://127.0.0.1:5000/

------------------------------------------------------------------------

## 🔁 How Translation Works

1.  User enters text
2.  Text is tokenized using mBART tokenizer
3.  Model generates translated output
4.  Result is displayed on the webpage

------------------------------------------------------------------------

## 🧩 Core Translation Function

``` python
def translator1(w):
    input_ids = tokenizer(w, return_tensors="pt").input_ids
    translated = model2.generate(input_ids)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text[0]
```

------------------------------------------------------------------------

## 🔮 Future Enhancements

-   Language selection support
-   Hindi ↔ Malayalam ↔ English
-   REST API
-   GPU support

------------------------------------------------------------------------

## 👩‍💻 Author

Amritha T
