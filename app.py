import gradio as gr
import tensorflow as tf
import tensorflow_datasets as tfds
from model import transformer
from main import predict
import pickle

with open('pretrained_weights/meta.pickle', 'rb') as handle:
    meta = pickle.load(handle)

tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file('pretrained_weights/tokenizer')

model = transformer(
    d_model = 256,
    vocab_size = meta['vocab_size'],
    num_layers=4,
    num_heads=8,
    dff=1024
)

model.load_weights('pretrained_weights/transformer_weights.h5')

def chatbot(sentence):
    print('\nQ:',sentence)
    resutl = predict(model,tokenizer,sentence,meta)
    print('\nA:',resutl)
    return resutl
    

gr.Interface(chatbot,inputs="text",outputs="text").launch(share=True)