class ChainOfThoughtDefault:

    @staticmethod
    def prompt_for_aspect_inferring(context, target):
        new_context = f'Given the sentence "{context}", '
        prompt = new_context + f'which specific aspect of {target} is possibly mentioned?'
        return new_context, prompt

    @staticmethod
    def prompt_for_opinion_inferring(context, target, aspect_expr):
        new_context = context + ' The mentioned aspect is about ' + aspect_expr + '.'
        prompt = new_context + f' Based on the common sense, what is the implicit opinion towards the mentioned aspect of {target}, and why?'
        return new_context, prompt

    @staticmethod
    def prompt_for_polarity_inferring(context, target, opinion_expr):
        new_context = context + f' The opinion towards the mentioned aspect of {target} is ' + opinion_expr + '.'
        prompt = new_context + f' Based on such opinion, what is the sentiment polarity towards {target}?'
        return new_context, prompt

    @staticmethod
    def prompt_for_polarity_label(context, polarity_expr, label_list):
        prompt = context + f' The sentiment polarity is {polarity_expr}.' + \
                 ' Based on these contexts, summarize and return the sentiment polarity only, ' \
                 'such as: {}.'.format(", ".join(label_list))
        return prompt
