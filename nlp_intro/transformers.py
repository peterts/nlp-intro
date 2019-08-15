from sklearn.base import TransformerMixin, BaseEstimator
from flashtext import KeywordProcessor
import pandas as pd
import numpy as np
import re
from abc import ABCMeta, abstractmethod


class ABCTextCleaner(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    def fit(self, texts, y=None, **fit_params):
        return self

    def transform(self, texts):
        if isinstance(texts, pd.DataFrame):
            return texts.applymap(self.clean_text)
        if isinstance(texts, pd.Series):
            return texts.apply(self.clean_text)
        _texts = [self.clean_text(text) for text in texts]
        if isinstance(texts, np.ndarray):
            return np.asarray(_texts)
        return _texts

    @abstractmethod
    def clean_text(self, text):
        pass


class TextReplacer(ABCTextCleaner):
    def __init__(self, replacements, case_sensitive=False):
        """
        Transformer used to make replacements in text

        Example:
            >> text_rep_transformer = TextReplacer({"Yo": "Hello", "dude": "buddy"})
            >> text_rep_transformer.transform(["Yo dude. What's up?"])
            ["Hello buddy. What's up?"]

        Args:
            replacements (Union[dict, list]): A dict with replacements, as in the example above, or a list of
                tuples like (keyword, clean_name).
            case_sensitive (bool): Whether or not to do case sensitive replacements
        """
        self.keyword_processor = KeywordProcessor(case_sensitive=case_sensitive)
        if isinstance(replacements, dict):
            replacements = replacements.items()
        for keyword, clean_name in replacements:
            self.keyword_processor.add_keyword(keyword, clean_name)

    def clean_text(self, text):
        return self.keyword_processor.replace_keywords(text)


class RegexReplacer(ABCTextCleaner):
    def __init__(self, replacements, case_sensitive=False):
        """
        Transformer used to make replacements in text

        Example:
            >> text_rep_transformer = TextReplacer({"Yo": "Hello", "dude": "buddy"})
            >> text_rep_transformer.transform(["Yo dude. What's up?"])
            ["Hello buddy. What's up?"]

        Args:
            replacements (Union[dict, list]): A dict with replacements, as in the example above, or a list of
                tuples like (keyword, clean_name).
            case_sensitive (bool): Whether or not to do case sensitive replacements
        """
        if isinstance(replacements, dict):
            replacements = replacements.items()
        groups = {}
        for keyword, clean_name in replacements:
            groups.setdefault(clean_name, []).append(keyword)
        flags = re.IGNORECASE if not case_sensitive else 0
        self.subs = [(re.compile(fr"(?:{'|'.join(keywords)})", flags=flags), clean_name)
                     for clean_name, keywords in groups.items()]

    def clean_text(self, text):
        for pattern, repl in self.subs:
            text = pattern.sub(repl, text)
        return text


class StopwordRemover(TextReplacer):
    def __init__(self, stopwords, case_sensitive=False):
        """
        Remove stopwords from text

        Args:
            stopwords (Iterable): An iterable containing stopwords
            case_sensitive (bool): Whether or not to do case sensitive replacements
        """
        super().__init__(((s, ' ') for s in stopwords), case_sensitive)


class RegexStopwordRemover(RegexReplacer):
    def __init__(self, stopwords, case_sensitive=False):
        """
        Remove stopwords from text

        Args:
            stopwords (Iterable): An iterable containing stopwords
            case_sensitive (bool): Whether or not to do case sensitive replacements
        """
        super().__init__(((s, ' ') for s in stopwords), case_sensitive)


class TextLowerTransformer(ABCTextCleaner):
    def clean_text(self, text):
        return text.lower() if isinstance(text, str) else text


class DuplicateWhitespaceRemover(ABCTextCleaner):
    def clean_text(self, text):
        return ' '.join(text.split()) if isinstance(text, str) else text
