"""
The following cell contains the commented code and basic functions to build up
the historical model. They can be used and exploited if needed. You can spend
some time on the code and understand its working logic. Feel free to ask
L2F members for explainations.
"""

import numpy as np
from nltk import ngrams

error_chars = ['n', 'w', 'd', 'x', 'g', 'e', '!', '#', '@']


def is_win(prompt):
    # Check whether a prompt is win for RF (1), lose for RF (0), or neither (-1)
    if prompt == '*':
        return 1
    elif '*' in prompt:
        return 0
    if prompt[0] in error_chars:
        return 0
    elif prompt[-1] in error_chars:
        return 1
    return -1   # Neither won nor lost


def aux_fn(df, poss_prompts, all_shots):
    # Build 3-grams of sequential shots for player (p) and opponent (o)

    # In this list we add all the triples of consequent shots: the first shot
    # is always a RN shot, followed by a RF shot and then by a RN shot.
    shots_opo = []

    for i in range(len(df)):
        shots, outcome = df['Shots'].iloc[i], df['Outcome'].iloc[i]
        length = len(shots.split())
        server = df['Server'].iloc[i]

        if server == 1:
            z = '' if length % 2 == 0 else ' '
            # 'X' for waiting to receive serve
            sentence = 'X ' + shots + z + outcome
            n_grams = list(ngrams(sentence.split(), 3))
            shots_opo.extend(n_grams[0::2])

        if server == 0:
            z = ' ' if length % 2 == 0 else ''
            # 'X' for waiting to receive serve
            sentence = 'X ' + shots + z + outcome
            n_grams = list(ngrams(sentence.split(), 3))
            shots_opo.extend(n_grams[1::2])

    # Remove triples containing uncommon shots
    shots_opo_com = [
        x for x in shots_opo
        if x[0] in poss_prompts and x[1] in poss_prompts and x[2] in poss_prompts
    ]

    # Matrix of statistics of the opponent: this matrix is the core policy of the historical RN
    model_matrix = []

    for i in range(len(poss_prompts)):
        for j in range(len(all_shots)):
            shots_distrib = [
                n_gram[2] for n_gram in shots_opo_com
                if n_gram[0] == poss_prompts[i] and n_gram[1] == all_shots[j]
            ]
            model_matrix.append(shots_distrib)

    Model = np.asarray(model_matrix).reshape((len(poss_prompts), len(all_shots)))

    return Model, shots_opo_com
