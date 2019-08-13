import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import normalize


class DocVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, vectorizer, embeddings, norm="l2"):
        """
        Used to create a document vector using a one-hot type vectorizer and embeddings. First the documents will be
        transformed using the vectorizer, then this matrix will be multiplied with the matrix for the word embeddings.
        This way, the word embeddings will be weighted according to its magnitude in the sparse/one-hot matrix.

        Args:
            vectorizer: Scikit-style vectorizer for transforming raw documents into sparse matrices
            embeddings: Gensim-style embeddings. You need to be able to check if a word is in the vocab by running
                "'some_word' in embeddings", and to get an embedding by indexing it like "embeddings['some_word']"
            norm: The type of normalization to use. See sklearn.preprocessing.normalization for options
        """
        if "norm" in vectorizer.get_params():
            vectorizer.set_params(norm=None)  # We don't want the vectorizer to do normalization, as we will do it in this transformer
        if norm is True:
            norm = "l2"
        self.vectorizer = vectorizer
        self.norm = norm
        self.embeddings = embeddings  # Note: if using this in a production system, make sure this attribute is not in the model when storing due to it's size
        self.embeddings_ = None  # Subset of self.embeddings used for transformation

    def fit(self, raw_documents, y=None):
        self.vectorizer.fit(raw_documents)

        # Select only the words for which we have an embedding available
        vocabulary = {}
        embeddings_ = []
        i = 0
        for word in self.vectorizer.get_feature_names():
            if word in self.embeddings:
                embeddings_.append(self.embeddings[word])
                vocabulary[word] = i
                i += 1
        self.embeddings_ = np.asarray(embeddings_)

        # Update vocabulary of vectorizer to only return the words selected above
        if self.vectorizer.vocabulary is not None:
            self.vectorizer.vocabulary = vocabulary
        self.vectorizer.vocabulary_ = vocabulary

        return self

    def transform(self, raw_documents):
        X = self.vectorizer.transform(raw_documents)
        X *= self.embeddings_
        if self.norm:
            return normalize(X, self.norm)
        return X
