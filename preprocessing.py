from bs4 import BeautifulSoup
import spacy
import unidecode
import contractions as cont
from word2number import w2n
nlp = spacy.load('en_core_web_md')

# exclude words from spacy stopwords list
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    return cont.fix(text, slang=False)


def preprocess_text(text, accented_chars=True, contractions=True, convert_num=False, extra_whitespace=True,
                    lemmatization=False, lowercase=True, punctuations=False, remove_html=True, remove_num=False,
                    special_chars=True, stop_words=False):
    """preprocess text with default option set to true for all steps"""
    if remove_html:
        text = strip_html_tags(text)
    if extra_whitespace:
        text = remove_whitespace(text)
    if accented_chars:
        text = remove_accented_chars(text)
    if contractions:
        text = expand_contractions(text)
    if lowercase:
        text = text.lower()

    doc = nlp(text)

    clean_text = []

    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words and token.is_stop and token.pos_ != 'NUM':
            flag = False
        # remove punctuations
        if punctuations and token.pos_ == 'PUNCT' and flag:
            flag = False
        # remove special characters
        if special_chars and token.pos_ == 'SYM' and flag:
            flag = False
        # remove numbers
        if remove_num and (token.pos_ == 'NUM' or token.text.isnumeric()) and flag:
            flag = False
        # convert number words to numeric numbers
        if convert_num and token.pos_ == 'NUM' and flag:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization and token.lemma_ != "-PRON-" and flag:
            edit = token.lemma_
        # append tokens edited and not removed to list
        if edit != "" and flag:
            clean_text.append(edit)

    clean_text = ' '.join(clean_text)
    return clean_text


if __name__ == '__main__':
    s = "you're doomed. WhaT're you talking about?"
    t = preprocess_text(s)
    print(t)
