# Natural Language Toolkit: NEGRA Corpus Reader
#
# Author: Philipp Nolte <ptn@daslaboratorium.de>
# URL: <http://www.experimentallabor.de/>

"""
Read NEGRA corpus files.
"""

from nltk.tree               import Tree
from nltk.util               import LazyMap
from nltk.util               import LazyConcatenation
from nltk.corpus.reader      import ConllCorpusReader
from nltk.corpus.reader.util import read_regexp_block
from nltk.corpus.reader.api  import CorpusReader

class NegraCorpusReader(ConllCorpusReader):
    """A corpus reader for NEGRA corpus files. A NEGRA corpus file consists out
    of annotated sentences separated by #BOS (beginning of sentence) and #EOS
    (end of sentence) markers on their own line. Each sentence consists of
    words and columns containing the lemma, tag, chunk and/or morphological tag.
    Each word and its tag information has its own line. Eg. the form

    %% word	lemma       tag     parent
    #BOS 1
    The         the         DET     500
    house       house       N       500
    is          be          V       501
    red         red         ADJ     501
    .           --          .       502
    #500        --          NP      502
    #501        --          VP      502
    #502        --          S       0
    #EOS 1

    Because of similar corpus structure, this reader is based on and very
    similar to the ConllCorpusReader. Both corpora have their tokens structured
    as grid. NEGRA corpus file provide more token information though.
    """

    #==========================================================================
    # Default column types
    #==========================================================================
    WORDS   = 'words'   # Column for the word
    LEMMA   = 'lemma'   # Column for the lemma
    POS     = 'pos'     # Column for the tag
    MORPH   = 'morph'   # Column for the morphological tag
    EDGE    = 'edge'    # Column for the grammatical function
    PARENT  = 'parent'  # Column for the words parent in the sentence tree
    SECEDGE = 'secedge' # Column for an optional second grammatical function
    COMMENT = 'comment' # Column for an optional comment from the editor

    # List of supported column types
    COLUMN_TYPES = (WORDS, LEMMA, POS, MORPH, EDGE, PARENT, SECEDGE, COMMENT)

    #==========================================================================
    # Constructor
    #==========================================================================

    def __init__(self,
                 root,
                 fileids,
                 column_types=None,
                 top_node='S',
                 beginning_of_sentence=r'#BOS.+$',
                 end_of_sentence=r'#EOS.+$',
                 encoding=None):
        """ Construct a new corpus reader for reading NEGRA corpus files.
        @param root: The root directory of the corpus files.
        @param fileids: A list of or regex specifying the files to read from.
        @param column_types: An optional C{list} of columns in the corpus.
        @param top_node: The top node of chunked sentence trees.
        @param beginning_of_sentence: A regex specifying the start of a sentence
        @param end_of_sentence: A regex specifying the end of a sentence
        @param encoding: The default corpus file encoding.
        """

        # Make sure there are no invalid column type
        if isinstance(column_types, list):
            for column_type in column_types:
                if column_type not in self.COLUMN_TYPES:
                    raise ValueError("Column %r is not supported." % columntype)
        else:
            column_types = self.COLUMN_TYPES

        # Define stuff
        self._top_node = top_node
        self._column_types = column_types
        self._fileids = fileids
        self._bos = beginning_of_sentence
        self._eos = end_of_sentence
        self._colmap = dict((c,i) for (i,c) in enumerate(column_types))

        # Finish constructing by calling the extended class' constructor
        CorpusReader.__init__(self, root, fileids, encoding)

    #==========================================================================
    # Data access methods
    #==========================================================================

    def lemmatised_words(self, fileids=None):
        """Retrieve a list of lemmatised words. Words are encoded as tuples in
           C{(word, lemma)} form.
        @return: A list of words and their tuples.
        @rtype: C{list} of C{(word, lemma)}
        """

        self._require(self.WORDS, self.LEMMA)
        return LazyConcatenation(LazyMap(self._get_lemmatised_words,
                                         self._grids(fileids)))

    def lemmatised_sents(self, fileids=None):
        """Retrieve a list of sentences and the words' lemma. Words
           are encoded as tuples in C{(word, lemma)} form.
        @return: A list of sentences with words and their lemma.
        @rtype: C{list} of C{list} of C{(word, lemma)}
        """

        self._require(self.WORDS, self.LEMMA)
        return LazyMap(self._get_lemmatised_words, self._grids(fileids))

    def morphological_words(self, fileids=None):
        """Retrieve a list of sentences with the words' morphological type.
           Words are encoded as tuples in C{(word, morph)} form.
        @return: A list of sentences with words and their morphological type.
        @rtype: C{list} of C{(word, morph)}
        """

        self._require(self.WORDS, self.MORPH)
        return LazyConcatenation(LazyMap(self._get_morphological_words,
                                         self._grids(fileids)))

    def morphological_sents(self, fileids=None):
        """Retrieve a list of sentences with the words' morphological type.
           Words are encoded as tuples in C{(word, morph)} form.
        @return: A list of sentences with words and their morphological type.
        @rtype: C{list} of C{list} of C{(word, morph)}
        """

        self._require(self.WORDS, self.MORPH)
        return LazyMap(self._get_morphological_words, self._grids(fileids))


    def chunked_words(self, fileids=None):
        """Retrieve a list of chunked words. Words are encoded as C{(word, tag)}
           and chunks are encoded as trees over C{(word, tag)} leaves.
        @return: A tree representation of the word chunk.
        @rtype: C{list} of (C{(str,str)} and L{Tree})
        """

        self._require(self.WORDS, self.POS, self.PARENT)
        return LazyConcatenation(LazyMap(self._get_chunked_words,
                                         self._grids(fileids)))

    def chunked_sents(self, fileids=None):
        """Retrieve a list of chunked sents as L{Tree} with leaves as tuples
           in C{(word, tag)} format.
        @return: A list of sentence tree representations.
        @rtype: C{list} of L{Tree}
        """

        self._require(self.WORDS, self.POS, self.PARENT)
        return LazyMap(self._get_chunked_words, self._grids(fileids))

    #==========================================================================
    # Transforms
    #==========================================================================


    def _get_morphological_words(self, grid):
        """Retrieve the words and their morphological type.
        @return: Return a list of words and their morphological type.
        @rtype: C{list} of C{(word, morph)}
        """

        return zip(self._get_column(grid, self._colmap[self.WORDS]),
                   self._get_column(grid, self._colmap[self.MORPH]))

    def _get_lemmatised_words(self, grid):
        """Retrieve the words and their corresponding lemma.
        @return: Return a list of lemmatised words.
        @rtype: C{list} of C{(word, lemma)}
        """

        return zip(self._get_column(grid, self._colmap[self.WORDS]),
                   self._get_column(grid, self._colmap[self.LEMMA]))


    def _get_chunked_words(self, grid):
        """Builds a chunk C{Tree} from the grid. The tree leaves are encoded
           as C{(word, tag)} tuples.
        @return: Return a tree representation of chunked words from the grid.
        @rtype: L{Tree}
        """

        # Get the needed columns. The parent column is crucial and contains the
        # token's parent node.
        tokens = zip(
            self._get_column(grid, self._colmap[self.WORDS], filter=False),
            self._get_column(grid, self._colmap[self.POS], filter=False),
            self._get_column(grid, self._colmap[self.PARENT], filter=False)
        )

        # Build a dictionary from the tree nodes. Tree nodes are found at the
        # end of the grid. Their word column consists out of a number starting
        # with the # character identifying the node.
        nodes = dict()
        node_parents = dict()
        top_node = None
        for (word, tag, parent) in [node for node in reversed(tokens)
                                    if node[0][0].startswith('#')]:
            parent = int(parent)
            word = int(word[1:])

            # The root node can be found at the end of the grid.
            if top_node is None and parent is 0:
                top_node = word

            # Prevents two tree roots.
            if top_node is not None and parent is 0:
                parent = top_node

            nodes[word] = Tree(tag, [])
            node_parents[word] = parent

        # Sentence is not correctly formeatted.
        if top_node is None:
            return None

        # Walk through the leaves and add them to their parents.
        last_parent = None
        for (word, tag, parent) in tokens[: - len(nodes)]:
            parent = int(parent)

            # The Negra corpus format allows tokens outside the sentence tree.
            # Prevent this, by changing their parent to the top_node's number.
            if parent is 0:
                parent = top_node

            # A chunk ends as soon as the current token has a new parent. The
            # last chunk has to be added to its own parents until it is located
            # in a subtree of the sentence tree root.
            if not parent == last_parent and last_parent is not None:
                node = last_parent
                while not node == top_node:
                    node_parent = node_parents[node]
                    if nodes[node] not in nodes[node_parent]:
                        nodes[node_parent].append(nodes[node])
                    node = node_parent

            # Add the current token to its parent.
            nodes[parent].append((word, tag))
            last_parent = parent

        return nodes[top_node]

    #==========================================================================
    # Grid reading
    #==========================================================================

    def _read_grid_block(self, stream):
        """Read blocks and return the grid"""

        # Sentence blocks are enclosed in start- and end-of-sentence tags.
        grids = []
        for block in read_regexp_block(stream, self._bos, self._eos):
            block = block.strip()
            if not block:
                continue

            # columns are separated by whitespace.
            grids.append([line.split() for line in block.split("\n")[1:]])

        return grids

    #==========================================================================
    # Helper methods
    #==========================================================================

    @staticmethod
    def _get_column(grid, column_index, filter=True):
        """Overridden; allows filtering sentence tree nodes from the grid"""

        # collect the column
        column_values = [grid[i][column_index] for i in range(len(grid))]

        # filter the column if needed
        if filter:
            column_values = [token for token in column_values
                             if token[0] is not '#']
        return column_values
