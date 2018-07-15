# LSTM-for-sentence-classification
Different kinds of lstm models for sentence classification.

lstm_skipgram.py: A LSTM model with word embedding generated by the skip-gram model.

What you have to modify:

1.data_helper function.

It returns:

  ·raw_text, a list of raw sentences.
  
    eg: ['It is a cat.', 'This is your handbag.', 'Coding is wonderful.']
    
  ·label, a 2-D matrix of labels corresponding to each sentence, shape = (None, 1)
  
    eg: [[1], [0], [3], [2], [0]]
    
2.label_set, the set of all labels, stored in a 2-D matrix.
   
   eg: label_set = [[0], [1], [2]]
  
No more~

lstm.py: A LSTM model with pre-trained word vectors.

What you have to modify:

1.data_helper function. It's the same as that above.

2.label_set. The same as that above.

3.build_dataset

  line 68: pre-trained word vector file. Mine is trained from wiki corpus.

  line 81-87: Modify it to adapt to the file format.
