# Next Word Predictor (LSTM)

A next-word prediction model built with **TensorFlow/Keras**. It is trained on a plain-text corpus using a stacked LSTM network and can generate text by repeatedly predicting the next word from a seed phrase.

## How it works

1. **Load text** from `data.txt`.
2. **Tokenize** with Keras `Tokenizer` (vocabulary of 4,993 words, plus 1 for padding = 4,994 classes).
3. **Split into sentences** on `.` and convert each to a sequence of word indices.
4. **Build n-gram training pairs.** For a sentence `[w1, w2, w3, w4]`, the pairs are:
   - `[w1] → w2`
   - `[w1, w2] → w3`
   - `[w1, w2, w3] → w4`
5. **Pad** inputs to the longest sequence (83 tokens, pre-padding) and **one-hot encode** the targets.
6. **Train** the model and **generate** text by feeding each prediction back in as input.

## Model architecture

| Layer | Output shape | Params |
|---|---|---|
| Input | (None, 83) | 0 |
| Embedding (4994 → 64) | (None, 83, 64) | 319,616 |
| LSTM (150, `return_sequences=True`) | (None, 83, 150) | 128,400 |
| LSTM (100) | (None, 100) | 100,400 |
| Dense (4994, softmax) | (None, 4994) | 504,394 |
| **Total** | | **1,053,410** (~4 MB) |

- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Epochs: 50
- Training samples: 25,878 sequences

## Results

After 50 epochs, training accuracy reached roughly **65.5%** (loss ≈ 1.57). Note that this is *training* accuracy only; no validation split is used, so the model is likely overfitting this small corpus.

Example generation from the seed `"I love"`:

```
I love to
I love to be
I love to be a
I love to be a father
I love to be a father butt
```

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/next-word-predictor.git
cd next-word-predictor
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your dataset

Place a plain-text file named `data.txt` in the project root (UTF-8 encoded). See [Dataset](#dataset) below.

### 4. Run the notebook

```bash
jupyter notebook next_word_predictor.ipynb
```

Run all cells in order. Training for 50 epochs took roughly 1.5 minutes per epoch on the author's machine (CPU), so expect it to take a while.

## Dataset

The training text (`data.txt`) is **not included** in this repository. Provide your own plain-text corpus (e.g. book text, scripts, articles). Make sure you have the rights to use and share any data you add.

> **Important:** the notebook hard-codes `num_classes=4994` and `Input(shape=(83,))`, which match the original corpus. If you use a different text, update these values (see below).

## Adapting to a different dataset

Replace the hard-coded numbers with values derived from your data:

```python
vocab_size = len(tokenizer.word_index) + 1          # instead of 4994
max_len = max(len(seq) for seq in text_data_input)  # instead of 83

text_data_output_final = to_categorical(text_data_output, num_classes=vocab_size)

model.add(Input(shape=(max_len,)))
model.add(Embedding(vocab_size, 64))
...
model.add(Dense(vocab_size, activation='softmax'))
```

## Possible improvements

- Add a validation split (`validation_split=0.1`) and `EarlyStopping` to reduce overfitting
- Use `sparse_categorical_crossentropy` to avoid one-hot encoding the targets (saves a lot of memory)
- Sample from the top-k predictions with a temperature instead of always using `argmax` (reduces repetition)
- Save/load the model and tokenizer (`model.save(...)`, `pickle` for the tokenizer) so training isn't repeated
- Use a larger corpus, or subword tokenization

## Project structure

```
next-word-predictor/
├── next_word_predictor.ipynb   # main notebook
├── data.txt                    # your dataset (not included)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Tech stack

Python · TensorFlow / Keras · NumPy · pandas · Jupyter

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
