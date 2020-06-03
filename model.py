from keras import Input, Model
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences

# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score

## 3 input
def get_model(nb_words, nb_class, EMBEDDING_DIM, input_length_list, embedding_matrix):
    embedding_layer1 = Embedding(nb_words[0],
                    EMBEDDING_DIM,
                    # weights=[embedding_matrix],
                    input_length=input_length_list[0],
                    trainable=False)
   
    embedding_layer2 = Embedding(nb_words[1],
                    EMBEDDING_DIM,
                    # weights=[embedding_matrix],
                    input_length=input_length_list[1],
                    trainable=False)
   
    embedding_layer3 = Embedding(nb_words[2],
                    EMBEDDING_DIM,
                    # weights=[embedding_matrix],
                    input_length=input_length_list[2],
                    trainable=False)
   
    lstm_layer1 = Bidirectional(LSTM(800, dropout=0.2, recurrent_dropout=0.2)) # ad_id
    lstm_layer2 = Bidirectional(LSTM(400, dropout=0.2, recurrent_dropout=0.2)) # pr_id
    lstm_layer3 = LSTM(200, dropout=0.2, recurrent_dropout=0.2) # adser_id

   
    sequence_1_input = Input(shape=(input_length_list[0],), dtype='int32')
    embedded_sequences_1 = embedding_layer1(sequence_1_input)
    y1 = lstm_layer1(embedded_sequences_1)
   
    sequence_2_input = Input(shape=(input_length_list[1], ), dtype='int32')
    embedded_sequences_2 = embedding_layer2(sequence_2_input)
    y2 = lstm_layer2(embedded_sequences_2)
    
    sequence_3_input = Input(shape=(input_length_list[2], ), dtype='int32')
    embedded_sequences_3 = embedding_layer3(sequence_3_input)
    y3 = lstm_layer3(embedded_sequences_3)
    
    merged = concatenate([y1, y2, y3])
    merged1 = Dropout(0.2)(merged)
    merged2 = BatchNormalization()(merged1)
    
    merged3 = Dense(200, activation="relu")(merged2)
    merged4 = Dropout(0.2)(merged3)
    merged5 = BatchNormalization()(merged4)
    
    merged6 = Dense(100, activation="relu")(merged5)
    merged7 = Dropout(0.2)(merged6)
    merged8 = BatchNormalization()(merged7)
    
    preds = Dense(nb_class, activation='softmax')(merged8)
 
    model = Model(inputs=[sequence_1_input, sequence_2_input, sequence_3_input], \
                  outputs=[preds])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    # model.summary()
    return model
 