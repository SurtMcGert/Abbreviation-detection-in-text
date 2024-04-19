import nltk


class PreProcessInput():
    @staticmethod
    def combine_lists_elementwise(list_A, list_B):
        """
        Combines two 2D lists of strings element-wise into a 2D list of tuples.

        Args:
            list_A: A 2D list of strings (e.g., [['A', 'A', 'A'], ['A', 'A', 'A']]).
            list_B: Another 2D list of strings with the same dimensions as list_A.

        Returns:
            A 2D list of tuples, where each tuple combines corresponding elements from list_A and list_B.

        Raises:
            ValueError: If the dimensions of list_A and list_B don't match.
        """

        # Check if dimensions match
        if len(list_A) != len(list_B) or len(list_A[0]) != len(list_B[0]):
            raise ValueError("Dimensions of lists A and B must be equal.")

        # Create the resulting list using list comprehension
        return [[(a, b) for a, b in zip(row_a, row_b)] for row_a, row_b in zip(list_A, list_B)]

    @staticmethod
    def convert_pos_tag(nltk_tag):
        """
        Converts NLTK POS tags to the format expected by the lemmatizer.

        Args:
            nltk_tag: The POS tag in NLTK format (e.g., VBG, NNS).

        Returns:
            The corresponding POS tag for the lemmatizer (n, v, a, r, or s) or None if no match.
        """

        tag_map = {
            'NUM': '',  # Number (not handled by lemmatizer)
            'CCONJ': '',  # Coordinating conjunction (not handled)
            'PRON': '',  # Pronoun (not handled)
            'NOUN': 'n',   # Noun
            'SCONJ': '',  # Subordinating conjunction (not handled)
            'SYM': '',   # Symbol (not handled)
            'INTJ': '',  # Interjection (not handled)
            'ADJ': 'a',    # Adjective
            'ADP': '',   # Preposition (not handled)
            'PUNCT': '',  # Punctuation (not handled)
            'ADV': 'r',    # Adverb
            'AUX': 'v',    # Auxiliary verb
            'DET': '',   # Determiner (not handled)
            'VERB': 'v',   # Verb
            'X': '',      # Other (not handled)
            'PART': '',   # Particle (not handled)
            'PROPN': 'n',   # Proper noun
        }
        return tag_map.get(nltk_tag)

    @staticmethod
    def lemmatize_list(data, pos_tags):
        """
        Lemmatizes a 2D list of tokens using NLTK.

        Args:
            data: A 2D list of strings (tokens) to be lemmatized.

        Returns:
            A 2D list containing the lemmatized tokens.
        """

        # Initialize the WordNet lemmatizer
        lemmatizer = nltk.WordNetLemmatizer()

        pos_tags = [[PreProcessInput.convert_pos_tag(
            tag) for tag in row] for row in pos_tags]

        data = PreProcessInput.combine_lists_elementwise(data, pos_tags)

        # Lemmatize with part-of-speech information
        lemmatized_data = [[token if pos == '' else lemmatizer.lemmatize(
            token, pos) for token, pos in row] for row in data]

        return lemmatized_data

    @staticmethod
    def pre_process_data(tokens, pos_tags):
        # lemmatize the data
        nltk.download('wordnet')
        data = PreProcessInput.lemmatize_list(tokens, pos_tags)
        # lowercase the data
        data = [[string.lower() for string in row] for row in data]
        return data
