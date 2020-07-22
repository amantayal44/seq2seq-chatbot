import tensorflow as tf

# Multi Head Attention Layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,num_heads,d_model,**kwargs):
        super(MultiHeadAttention,self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.wq  = tf.keras.layers.Dense(self.d_model)
        self.wk  = tf.keras.layers.Dense(self.d_model)
        self.wv  = tf.keras.layers.Dense(self.d_model)

        self.dense  = tf.keras.layers.Dense(self.d_model)

    def split_heads(self,inputs,batch_size):
        inputs = tf.reshape(inputs,
            shape = (batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(inputs,perm=[0,2,1,3])

    def scaled_dot_product_attention(self,query,key,value,mask=None):
        """ Calculate the attention weights """
        matmul_qk = tf.matmul(query,key,transpose_b=True)

        #scale matmul_qk
        depth = tf.cast(tf.shape(key)[-1],tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        # add the mask to zero out padded values
        if mask is not None:
            logits += (mask * -1e9) # adding -Inf value

        # softmax on last axis ( seq_len_k)
        attention_weights = tf.nn.softmax(logits,axis=-1)

        output = tf.matmul(attention_weights,value)

        return output

    def call(self,query,key,value,mask=None):

        batch_size = tf.shape(query)[0]

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        # splitting in (... , num_heads,seq_lem,depth)
        query = self.split_heads(query,batch_size)
        key = self.split_heads(key,batch_size)
        value = self.split_heads(value,batch_size)

        #scaled_dot_product_attention
        scaled_attention = self.scaled_dot_product_attention(
            query,
            key,
            value,
            mask)

        #transpose to (... , seq_lem,num_heads,depth)
        scaled_attention = tf.transpose(scaled_attention,perm=[0,2,1,3])

        #concatenation of attention_heads
        concat_attention = tf.reshape(scaled_attention,
            shape = (batch_size,-1,self.d_model))

        output = self.dense(concat_attention)

        return output

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
                "num_heads" : self.num_heads,
                "d_model" : self.d_model
            })
        return config


# Positional Encoding Layer
class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self,d_model,max_pos_size,**kwargs):
        super(PositionalEncoding,self).__init__(**kwargs)
        self.d_model = d_model
        self.max_pos_size = max_pos_size

        assert d_model % 2 == 0

        self.pos_encoding = self.positional_encoding(self.max_pos_size,
            self.d_model)

    def get_angles(self,position,i,d_model):
        angles = 1/tf.pow(10000,(2 * (i//2))/tf.cast(d_model,tf.float32))
        return position * angles

    def positional_encoding(self,position,d_model):
        angle_rads = self.get_angles(
                position = tf.range(position,dtype = tf.float32)[:,tf.newaxis],
                i = tf.range(d_model,dtype=tf.float32)[tf.newaxis,:],
                d_model = d_model
            )

        #applying sine to even  indices
        sines = tf.math.sin(angle_rads[:,0::2])
        #applying cos to odd indices
        cosines = tf.math.cos(angle_rads[:,1::2])

        # reshaping so that when concat even position come after odd
        sines = tf.reshape(sines,shape=(position,-1,1))
        cosines = tf.reshape(cosines,shape=(position,-1,1))

        pos_encoding = tf.concat([sines,cosines],axis=-1) #(position,d_model/2,2)
        # reshaping to (1,position,d_model)
        pos_encoding = tf.reshape(pos_encoding,(1,position,d_model))

        return pos_encoding

    def call(self,inputs):
        return inputs + self.pos_encoding[:,:tf.shape(inputs)[1],:]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
                "d_model" : self.d_model,
                "max_pos_size" : self.max_pos_size
            })
        return config

        
# padding mask layer
class PaddingMask(tf.keras.layers.Layer):

    def __init__(self,**kwargs):
        super(PaddingMask,self).__init__(**kwargs)

    def call(self,inputs):
        mask = tf.cast(tf.math.equal(inputs,0),tf.float32)
        return mask[:,tf.newaxis,tf.newaxis,:]

# look ahead mask layer 
class LookAheadMask(tf.keras.layers.Layer):
    """ this mask retuns combination of look ahead and padded mask"""
    def __init__(self,**kwargs):
        super(LookAheadMask,self).__init__(**kwargs)

    def call(self,inputs):
        seq_len = tf.shape(inputs)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padded_mask = tf.cast(tf.math.equal(inputs,0),tf.float32)
        padded_mask = padded_mask[:,tf.newaxis,tf.newaxis,:]
        return tf.maximum(look_ahead_mask,padded_mask)


def encoder_layer(d_model,num_heads,dff,rate=0.1,name="encoder_layer"):
    """
        d_model = dim of query,key and value
        num_heads = no. of heads in multi attention layer
        dff = no. of units of hidden layer in feed forward network
        rate = Dropout rate (default = 0.1)
        name = name of model

        NOTE: if using encoder layer model multiple times use different
        name for each model

    """
    inputs = tf.keras.layers.Input(shape=(None,d_model),name="inputs")
    padding_mask = tf.keras.layers.Input(shape=(1,1,None),name="padding_mask")

    mha = MultiHeadAttention(num_heads,d_model)
    attention = mha(
        query = inputs,
        key = inputs,
        value = inputs,
        mask = padding_mask
    )

    attention = tf.keras.layers.Dropout(rate)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    ffn_output = tf.keras.layers.Dense(dff,activation='relu')(attention)
    ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + 
        ffn_output)

    return tf.keras.models.Model(
        inputs=[inputs,padding_mask],outputs=outputs,name=name)

def encoder(d_model,vocab_size,num_layers,num_heads,dff,rate=0.1,name="encoder"):

    inputs = tf.keras.layers.Input(shape=(None,),name="inputs")
    padding_mask = tf.keras.layers.Input(shape=(1,1,None),name="padded_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size,d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(d_model,vocab_size)(embeddings)

    outputs = tf.keras.layers.Dropout(rate)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            d_model,num_heads,dff,rate,
            name = "encoder_layer_{}".format(i)
            )([outputs,padding_mask])

    return tf.keras.models.Model(
        inputs=[inputs,padding_mask],outputs=outputs,name=name)

def decoder_layer(d_model,num_heads,dff,rate=0.1,name="decoder_layer"):
    """
        d_model = dim of query,key and value
        num_heads = no. of heads in multi attention layer
        dff = no. of units of hidden layer in feed forward network
        rate = Dropout rate (default = 0.1)
        name = name of model

        NOTE: if using decoder layer model multiple times use different
        name for each model

    """

    inputs = tf.keras.layers.Input(shape=(None,d_model),name="inputs")
    enc_outputs = tf.keras.layers.Input(
        shape=(None,d_model),name="encoder_outputs")
    look_ahead_mask = tf.keras.layers.Input(
        shape=(1,None,None),name="look_ahead_mask")
    padding_mask = tf.keras.layers.Input(shape=(1,1,None),name="padding_mask")

    mha1 = MultiHeadAttention(num_heads=num_heads,d_model=d_model)
    attention1 = mha1(
        query = inputs,
        key = inputs,
        value = inputs,
        mask = look_ahead_mask)

    attention1 = tf.keras.layers.Dropout(rate)(attention1)
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        attention1 + inputs)

    mha2 = MultiHeadAttention(num_heads=num_heads,d_model=d_model)
    attention2 = mha2(
        query = attention1,
        key = enc_outputs,
        value = enc_outputs,
        mask = padding_mask)

    attention2 = tf.keras.layers.Dropout(rate)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        attention2 + attention1)

    ffn_output = tf.keras.layers.Dense(dff,activation='relu')(attention2)
    ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        ffn_output + attention2)

    return tf.keras.models.Model(
        inputs = [inputs,enc_outputs,look_ahead_mask,padding_mask],
        outputs = outputs,
        name = name)

def decoder(d_model,vocab_size,num_layers,num_heads,dff,rate=0.1,name="decoder"):

    inputs = tf.keras.layers.Input(shape=(None,),name="inputs")
    enc_outputs = tf.keras.layers.Input(
        shape=(None,d_model),name="encoder_outputs")
    look_ahead_mask = tf.keras.layers.Input(
        shape=(1,None,None),name="look_ahead_mask")
    padding_mask = tf.keras.layers.Input(shape=(1,1,None),name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size,d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(d_model,vocab_size)(embeddings)

    outputs = tf.keras.layers.Dropout(rate)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            d_model,num_heads,dff,rate,
            name = "decoder_layer_{}".format(i)
            )([outputs,enc_outputs,look_ahead_mask,padding_mask])

    return tf.keras.models.Model(
        inputs = [inputs,enc_outputs,look_ahead_mask,padding_mask],
        outputs = outputs,
        name = name)

def transformer(d_model,vocab_size,num_layers,num_heads,dff,rate=0.1,
    name="transformer"):
    
    enc_inputs = tf.keras.layers.Input(shape=(None,),name="encoder_inputs")
    dec_inputs = tf.keras.layers.Input(shape=(None,),name="decoder_inputs")

    enc_padding_mask = PaddingMask(name="enc_padding_mask")(enc_inputs)
    dec_look_ahead_mask = LookAheadMask(name="dec_look_ahead_mask")(dec_inputs)
    dec_padding_mask = PaddingMask(name="dec_padding_mask")(enc_inputs)

    enc_outputs = encoder(
        d_model,vocab_size,num_layers,num_heads,dff,rate)(
        [enc_inputs,enc_padding_mask])

    dec_outputs = decoder(
        d_model,vocab_size,num_layers,num_heads,dff,rate)(
        [dec_inputs,enc_outputs,dec_look_ahead_mask,dec_padding_mask])

    outputs = tf.keras.layers.Dense(vocab_size,name="Output")(dec_outputs)

    return tf.keras.models.Model(
        inputs = [enc_inputs,dec_inputs],
        outputs = outputs,
        name = name)
    


