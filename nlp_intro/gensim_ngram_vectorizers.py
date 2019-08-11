from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.exceptions import NotFittedError
from gensim.models.phrases import Phrases, Phraser
from spacy.lang.nb import Norwegian
import functools


class GensimNgramMixin:
    """
    Modifies the functionality of a CountVectorizer or TfidifVectorizer, to use gensim for n-gram generation
    rather than the built in generation
    """
    def __init__(self, phrases_common_terms=None, phrases_min_count=5, phrases_threshold=1, *args, **kwargs):
        self.phrases = functools.partial(
            Phrases, min_count=phrases_min_count, threshold=phrases_threshold, common_terms=phrases_common_terms
        )
        super().__init__(*args, **kwargs)
        self.nlp_pipeline = Norwegian()
        self.nlp_pipeline.add_pipe(self.nlp_pipeline.create_pipe("sentencizer"))
        self.phrasers_ = None

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def fit_transform(self, raw_documents, y=None):
        documents_sentences = self._get_documents_sentences(raw_documents)
        documents_sentences = self._fit_phrasers(documents_sentences)
        documents_tokens = self._flatten_documents_and_filter_tokens(documents_sentences)
        self.analyzer = lambda x: x
        return super().fit_transform(documents_tokens)

    def transform(self, raw_documents):
        if self.phrasers_ is None:
            raise NotFittedError("You need to fit the transformer before calling .transform")
        documents_sentences = self._get_documents_sentences(raw_documents)
        documents_sentences = self._transform_with_phrasers(documents_sentences)
        documents_tokens = self._flatten_documents_and_filter_tokens(documents_sentences)
        return super().transform(documents_tokens)

    def _get_documents_sentences(self, raw_documents):
        # Copied from build_analyzer in CountVectorizer for analyzer="word"
        preprocess = self.build_preprocessor()
        tokenize = self.build_tokenizer()
        stop_words = self.get_stop_words()
        self._check_stop_words_consistency(stop_words, preprocess, tokenize)

        document_sentences = []
        for doc in raw_documents:
            doc = preprocess(self.decode(doc))
            doc = self.nlp_pipeline(doc).sents
            doc = [tokenize(str(sent)) for sent in doc]
            document_sentences.append(doc)

        return document_sentences

    def _fit_phrasers(self, documents_sentences):
        min_n, max_n = self.ngram_range
        assert 1 <= min_n <= max_n, "min n-gram must be greater than 1, and less than max n-gram"
        self.phrasers_ = []
        for _ in range(max_n-1):
            documents_sentences_flat = [sent for doc in documents_sentences for sent in doc]
            phrases = self.phrases(documents_sentences_flat)
            phraser = Phraser(phrases)
            self.phrasers_.append(phraser)
            documents_sentences = phraser[documents_sentences]
        return documents_sentences

    def _transform_with_phrasers(self, documents_sentences):
        for phraser in self.phrasers_:
            documents_sentences = phraser[documents_sentences]
        return documents_sentences

    def _flatten_documents_and_filter_tokens(self, documents_sentences):
        min_n, max_n = self.ngram_range
        stop_words = self.get_stop_words()

        def _sub_tokens(token):
            if token in stop_words:
                return []
            if "_" not in token:
                if min_n == 1:
                    return [token]
                return []
            words = [w for w in token.split("_") if w not in stop_words]
            sub_tokens = []
            if min_n == 1:
                sub_tokens.extend(words)
            if min_n <= len(words) <= max_n:
                sub_tokens.append(token)
            return sub_tokens

        documents_tokens = [[sub_token for sentence in doc for token in sentence for sub_token in _sub_tokens(token)]
                            for doc in documents_sentences]

        return documents_tokens

    def get_stop_words(self):
        stop_words = super().get_stop_words()
        if stop_words is None:
            return []
        return stop_words


class NgramCountVectorizer(GensimNgramMixin, CountVectorizer):
    """
    CountVectorizer using gensim for generating n-grams
    """


class NgramTfidfVectorizer(GensimNgramMixin, TfidfVectorizer):
    """
    TfidifVectorizer using gensim for generating n-grams
    """







