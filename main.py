import tensorflow as tf
import argparse
import pickle

tf.random.set_seed(42)

from model import transformer
from dataset import get_dataset , preprocess_sentence

def save_pickle(data,filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,d_model,warmup_steps = 4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self,step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        temp = tf.cast(self.d_model, dtype=tf.float32)
        return tf.math.rsqrt(temp) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model' : self.d_model,
            'warmup_steps' : self.warmup_steps
        }
        return config

def inference(model,tokenizer,sentence,meta):

    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(
        meta['start_token'] + tokenizer.encode(sentence) + meta['end_token'],
        axis = 0)

    output = tf.expand_dims(meta['start_token'],axis=0)

    for i in range(meta['max_length']):
        prediction = model(inputs=[sentence,output],training=False)

        #select last word
        prediction = prediction[:,-1:,:]
        prediction_id = tf.cast(tf.argmax(prediction,axis=-1),tf.int32)

        if tf.equal(prediction_id,meta['end_token']):
            break
        output = tf.concat([output,prediction_id],axis=-1)

    return tf.squeeze(output,axis=0)

def predict(model,tokenizer,sentence,meta):
    prediction = inference(model,tokenizer,sentence,meta)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    return predicted_sentence

def evaluate(model,tokenizer,meta):
    print('\nEvaluate')
    sentence = "Hi, How are you"
    output = predict(model,tokenizer,sentence,meta)
    print('\ninput: {}\noutput: {}'.format(sentence, output))

    sentence = "Will you marry me"
    output = predict(model,tokenizer,sentence,meta)
    print('\ninput: {}\noutput: {}'.format(sentence, output))

    sentence = "where did you visit in london"
    for _ in range(5):
        output = predict(model,tokenizer,sentence,meta)
        print('\ninput: {}\noutput: {}'.format(sentence, output))
        sentence = output

def main(params):

    print("\n ...loading dataset\n")
    dataset,test_dataset,tokenizer,meta = get_dataset(
        params.max_samples,params.max_length,params.batch_size,
        validation_split=params.validation_split)

    print("\n ...creating model\n")
    model = transformer(params.d_model,meta['vocab_size'],params.num_layers,
        params.num_heads,params.dff,params.rate)

    # saving model without compilation
    model.save('model_untrained.h5')
    optimizer = tf.keras.optimizers.Adam(
        CustomSchedule(params.d_model),beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, params.max_length - 1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def accuracy(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, params.max_length - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    print("\n ...training model\n")
    model.compile(optimizer, loss=loss_function, metrics=[accuracy])
    history = model.fit(dataset, epochs=params.epochs,
        validation_data=test_dataset)

    print("\nSaving model weights, tokenizer and meta data\n")
    model.save('model_trained.h5')
    tokenizer.save_to_file('tokenizer')
    model.save_weights('model_weights.h5')

    # saving history and meta using pickle
    save_pickle(meta,'meta')
    save_pickle(history.history,'history')

    evaluate(model,tokenizer,meta)

if __name__ == '__main__' : 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_samples', default = 50000, type= int,
        help = 'maximum no. of total samples to use')
    parser.add_argument(
        '--max_length',default=40,type=int,help = 'maximum sentence length')
    parser.add_argument('--validation_split',default=0.2,type=float,
        help = 'split total samples in test and train dataset (value in (0,1)')
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--d_model',default=256,type=int)
    parser.add_argument('--num_layers',default=4,type=int)
    parser.add_argument('--num_heads',default=8,type=int)
    parser.add_argument('--dff',default=1024,type=int)
    parser.add_argument('--rate',default=0.1,type=float)
    parser.add_argument('--epochs',default=20,type=int)

    params = parser.parse_args()
    main(params)



