import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample English and French sentences
english_sentences = ['hello', 'how are you', 'good morning']
french_sentences = ['bonjour', 'comment ça va', 'bonjour']

# Add <start> and <end> tokens for decoder target
french_sentences = ['<start> ' + s + ' <end>' for s in french_sentences]

# English Tokenizer
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(english_sentences)
eng_seq = eng_tokenizer.texts_to_sequences(english_sentences)
eng_vocab_size = len(eng_tokenizer.word_index) + 1

# French Tokenizer
fr_tokenizer = Tokenizer(filters='')
fr_tokenizer.fit_on_texts(french_sentences)
fr_seq = fr_tokenizer.texts_to_sequences(french_sentences)
fr_vocab_size = len(fr_tokenizer.word_index) + 1
fr_index_word = {i: w for w, i in fr_tokenizer.word_index.items()}

# Get max lengths
max_eng_len = max(len(s) for s in eng_seq)
max_fr_len = max(len(s) for s in fr_seq)

# Pad sequences
encoder_input = pad_sequences(eng_seq, maxlen=max_eng_len, padding='post')
decoder_target = pad_sequences([s[1:] for s in fr_seq], maxlen=max_fr_len-1, padding='post')

# Model configuration
embedding_dim = 64
latent_dim = 128

# Build the model
model = Sequential()
model.add(Embedding(input_dim=eng_vocab_size, output_dim=embedding_dim, input_length=max_eng_len))
model.add(LSTM(latent_dim))
model.add(Dense(fr_vocab_size, activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(encoder_input, np.array(decoder_target)[:, 0], epochs=300, verbose=0)

# Translation function
def translate_simple(text):
    seq = eng_tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding='post')
    pred = model.predict(seq, verbose=0)
    word_id = np.argmax(pred[0])
    return fr_index_word.get(word_id, '?')

# Inference loop
print("Simple Translator is ready. Type 'exit' to quit.")
while True:
    input_text = input("Enter English sentence (or 'exit' to quit): ").lower()
    if input_text == 'exit':
        break
    translation = translate_simple(input_text)
    print("French:", translation)
