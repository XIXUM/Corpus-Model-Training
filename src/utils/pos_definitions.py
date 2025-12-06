# POS Tag Definitions
# Maps Penn Treebank POS tags to their full descriptions

POS_TAG_DEFINITIONS = {
  # Nouns
  'NN': 'Noun, singular or mass',
  'NNS': 'Noun, plural',
  'NNP': 'Proper noun, singular',
  'NNPS': 'Proper noun, plural',
  
  # Verbs
  'VB': 'Verb, base form',
  'VBD': 'Verb, past tense',
  'VBG': 'Verb, gerund or present participle',
  'VBN': 'Verb, past participle',
  'VBP': 'Verb, non-3rd person singular present',
  'VBZ': 'Verb, 3rd person singular present',
  
  # Adjectives
  'JJ': 'Adjective',
  'JJR': 'Adjective, comparative',
  'JJS': 'Adjective, superlative',
  
  # Adverbs
  'RB': 'Adverb',
  'RBR': 'Adverb, comparative',
  'RBS': 'Adverb, superlative',
  
  # Pronouns
  'PRP': 'Personal pronoun',
  'PRP$': 'Possessive pronoun',
  'WP': 'Wh-pronoun',
  'WP$': 'Possessive wh-pronoun',
  
  # Determiners
  'DT': 'Determiner',
  'WDT': 'Wh-determiner',
  'PDT': 'Predeterminer',
  
  # Prepositions
  'IN': 'Preposition or subordinating conjunction',
  'TO': 'To',
  
  # Conjunctions
  'CC': 'Coordinating conjunction',
  
  # Particles
  'RP': 'Particle',
  
  # Possessive
  'POS': 'Possessive ending',
  
  # Numbers
  'CD': 'Cardinal number',
  'OD': 'Ordinal number',
  
  # Punctuation
  '.': 'Sentence-final punctuation',
  ',': 'Comma',
  ':': 'Colon, semicolon',
  'LRB': 'Left round bracket',
  'RRB': 'Right round bracket',
  '``': 'Opening quotation mark',
  "''": 'Closing quotation mark',
  '$': 'Dollar sign',
  '#': 'Pound sign',
  
  # Other
  'EX': 'Existential there',
  'FW': 'Foreign word',
  'LS': 'List item marker',
  'MD': 'Modal',
  'SYM': 'Symbol',
  'UH': 'Interjection',
}

def get_pos_tag_with_definition(tag: str) -> str:
    """
    Get the full definition for a POS tag.
    Example: 'NN' -> 'NN (Noun, singular or mass)'
    """
    if not tag or tag == '-' or not tag.strip():
        return tag
        
    definition = POS_TAG_DEFINITIONS.get(tag)
    if definition:
        return f"{tag} ({definition})"
    
    return tag

