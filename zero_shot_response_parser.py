def parse_model_response_universal(prompt_type, response):
    """universal parser for all models"""
    assert prompt_type in ['default', 'default_v2']
    assert (isinstance(response, str))

    t = 'select one from: positive, negative, neutral.' if prompt_type is 'default' else \
        'choose from: positive, negative, neutral.'

    response = response.lower()

    # NOTE: This is the adaptation for the evaluation stage
    # and that some models are tend to output the prompt message first.
    # Hence, we chop the initial part in order to prevent its affection on the final result as follows:
    if t in response:
        response = response.split(t)[1].strip()

    if any(e in response for e in ['neutral', 'нейтральн']):
        return 'neutral'
    elif any(e in response for e in ['positiv', 'позитивн']):
        return 'positive'
    elif any(e in response for e in ['negativ', 'негатив']):
        return 'negative'
    else:
        return 'neutral (na)'
