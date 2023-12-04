from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# hugging face
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)

def get_bert_embeddings_tf(text):
    # use bert tokenizer to tokenize the text in a way bert can then work with it
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)

    # - generate embeddings - bert generates output of shape [batch_size, sequence_length, hidden_size]
    # - batch size is 1 because we use only one description, sequence length is max_length and hidden size is from bert 768
    # (1, 512, 768)
    outputs = model(inputs)

    last_hidden_states = outputs.last_hidden_state

    # this is mean pooling which is done over sequence_length and therefore it averages the embeddings across all tokens in sequence
    embeddings = tf.reduce_mean(last_hidden_states, axis=1)
    return embeddings

description = "Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`."

embeddings = get_bert_embeddings_tf(description)
print(embeddings)
