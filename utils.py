import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def remove_unicode(s):
    """Remove unicode characters from a string."""
    if isinstance(s, str):
        return re.sub(r'[^\x00-\x7F]+', '', s)
    return s


def clean_text(s):
    """Clean the input text by removing URLs, markdown links, mentions, and punctuation."""
    if isinstance(s, str):
        s = re.sub(r"http[s]?://\S+", "", s)
        s = re.sub(r"\[.*?\]\(.*?\)", "", s)
        s = re.sub(r"@\w+", "", s)  
        s = re.sub(r"[^\w\s]", "", s)
        s = s.strip() 
        return s
    return s


def remove_age_gender_pattern(statement):
    """Remove age and gender patterns (like '24M' or '30F') from the statement."""
    if not isinstance(statement, str):
        return statement
    pattern = r'\b\d{1,3}[MF]\b'
    cleaned_statement = re.sub(pattern, '', statement)
    return cleaned_statement.strip()



def stemming(tokens):
    return " ".join(stemmer.stem(str(token)) for token in tokens)



def remove_stopwords(tokens):
    return [token for token in tokens if token.lower() not in stop_words]


def ltsm_model_input_transformer(input):
    input["statement"] = input["statement"].apply(remove_unicode)
    input["statement"] = input["statement"].apply(clean_text)
    input["statement"] = input["statement"].apply(remove_age_gender_pattern)
    input["tokens"] = input["statement"].apply(lambda x: word_tokenize(x) if isinstance(x, str) else [])
    input["tokens"] = input["tokens"].apply(remove_stopwords)
    # print(input)
    input["tokens"] = [' '.join(tokens) for tokens in input["tokens"]]
    return input["tokens"]