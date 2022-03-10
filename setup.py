"""
Pre-process the text dataset.

Code adapted from: https://github.com/michiyasunaga/squad/blob/main/setup.py
"""
import numpy as np
import spacy
import ujson as json

from args import get_setup_args
from collections import Counter
from tqdm import tqdm

NULL = "--NULL--"
OOV = "--OOV--"
CH_START = "--CHS--"
CH_END = "--CHE--"

chapters = ["The Boy Who Lived",
            "The Vanishing Glass",
            "The Letters from No One",
            "The Keeper of The Keys",
            "Diagon Alley",
            "The Journey from Platform Nine and Three-Quarters",
            "The Sorting Hat",
            "The Potions Master",
            "The Midnight Duel",
            "Halloween",
            "Quidditch",
            "The Mirror of Erised",
            "Nicolas Flamel",
            "Norbert the Norwegian Ridgeback",
            "The Forbidden Forest",
            "Through the Trapdoor",
            "The Man with Two Faces"]

chapters_dict = {ch.lower() : idx for idx, ch in enumerate(chapters)}

def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def get_embedding(counter, emb_file, vec_size, num_vectors, limit=-1):
    print("Pre-processing word vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    
    with open(emb_file, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, total=num_vectors):
            array = line.split()
            word = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            if word in counter and counter[word] > limit:
                embedding_dict[word] = vector
    print(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding embedding vector")
    
    token2idx_dict = {token : idx for idx, token in enumerate(embedding_dict.keys(), 4)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    token2idx_dict[CH_START] = 2
    token2idx_dict[CH_END] = 3
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    embedding_dict[CH_START] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
    embedding_dict[CH_END] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
    
    idx2emb_dict = {idx: embedding_dict[token]
                   for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

def convert_text_to_indices(args, tokens, word2idx_dict):
    
    def _get_word_idx(word):
        if word in word2idx_dict:
            return word2idx_dict[word]
        return word2idx_dict[OOV]
    
    print("Converting text to indices...")
    text_idxs = []
    for i, token in tqdm(enumerate(tokens)):
        text_idxs.append(_get_word_idx(token))
    
    Object = lambda **kwargs: type("Object", (), kwargs)
    features = Object(text_idxs = text_idxs)
    return features

def process_file(filename, word_counter=None):
    print(f"Preprocessing file {filename}")
    all_tokens = []
    text = ""

    with open(filename, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if line == "" or line.startswith("Page |") or line.isnumeric():
                continue
            line = line.lower()
            if line in chapters_dict:
                line = CH_START + " " + line + " " + CH_END
            text = text + " " + line
            tokens = word_tokenize(line)
            
            if word_counter is not None:
                for token in tokens:
                    word_counter[token] += 1
            
            all_tokens.extend(tokens)
    return all_tokens, text

def pre_process(args, text_file, features_file, word2idx_dict=None):
    word_counter = Counter()
    tokens, text = process_file(text_file, word_counter)
    text_file_processed = text_file.replace(".txt", ".processed.txt")
    save(text_file_processed, text, message="processed data")
    
    if word2idx_dict is None:
        word_emb_mat, word2idx_dict = get_embedding(word_counter, args.glove_file, args.glove_dim, args.glove_num_vecs)
        save(args.word_emb_file, word_emb_mat, message="word embedding")
        save(args.word2idx_file, word2idx_dict, message="word dictionary")
    
    features = convert_text_to_indices(args, tokens, word2idx_dict)
    np.savez(features_file, 
             text_idxs = np.array(features.text_idxs))
    
    return word2idx_dict

if __name__ == '__main__':
    args_ = get_setup_args()
    nlp = spacy.blank("en")

    #Preprocess data
    print("Preprocessing train set...")
    word2idx_dict = pre_process(args_, args_.train_file, args_.train_features_file)

    print("Preprocessing dev set...")
    pre_process(args_, args_.dev_file, args_.dev_features_file, word2idx_dict)

    print("Preprocessing test set...")
    pre_process(args_, args_.test_file, args_.test_features_file, word2idx_dict)


