from termcolor import colored
import re


def print_with_highlighting(text, word, is_pattern=False):
    i = 0
    for match in re.finditer(re.escape(word) if not is_pattern else word, text):
        j, k = match.span()
        print(text[i:j], end="")
        print(_on_yellow(text[j:k]), end="")
        i = k
    print(text[i:])


def _on_yellow(text):
    return colored(text, None, 'on_yellow')