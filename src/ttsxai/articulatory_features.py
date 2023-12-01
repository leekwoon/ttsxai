def get_articulatory_features_for_phoneme(phonesymbols):
    
    # Define the mapping of phonemes to articulatory features
    articulatory_dict = {
        'P': 'Bilabial',
        'B': 'Bilabial',
        'T': 'Alveolar',
        'D': 'Alveolar',
        'K': 'Velar',
        'G': 'Velar',
        'F': 'Labiodental',
        'V': 'Labiodental',
        'S': 'Alveolar',
        'Z': 'Alveolar',
        'SH': 'Post-Alveolar',
        'ZH': 'Post-Alveolar',
        'CH': 'Post-Alveolar',
        'JH': 'Post-Alveolar',
        'M': 'Bilabial',
        'N': 'Alveolar',
        'NG': 'Velar',
        'L': 'Alveolar',
        'R': 'Alveolar',
        'HH': 'Glottal',
        'TH': 'Dental',
        'DH': 'Dental',
        'Y': 'Palatal',
        'W': 'Labio-Velar',
        '.': 'Punctuation',
        ',': 'Punctuation',
        "'": 'Punctuation',
        ';': 'Punctuation',
        '-': 'Punctuation',
        '(': 'Punctuation',
        ')': 'Punctuation',
        ':': 'Punctuation',
        '?': 'Punctuation',
        '!': 'Punctuation',
        ' ': 'Space',
        'AH': 'Vowel',
        'AA': 'Vowel',
        'AE': 'Vowel',
        'AO': 'Vowel',
        'AW': 'Vowel',
        'AY': 'Vowel',
        'EH': 'Vowel',
        'ER': 'Vowel',
        'EY': 'Vowel',
        'IH': 'Vowel',
        'IY': 'Vowel',
        'OW': 'Vowel',
        'OY': 'Vowel',
        'UH': 'Vowel',
        'UW': 'Vowel'
        # Add more phonemes or features if needed here
    }
    
    phonesymbols = [phoneme.upper() for phoneme in phonesymbols]

    # Remove numbers (stress markers) to get pure phonemes
    stripped_phonemes = [''.join([char for char in phoneme if not char.isdigit()]) for phoneme in phonesymbols]

    # Map the stripped phonemes to their articulatory features
    features = [articulatory_dict.get(phoneme, 'Unknown') for phoneme in stripped_phonemes]
    
    return features