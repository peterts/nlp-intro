from termcolor import colored
from diff_match_patch import diff_match_patch
try:
    from IPython.display import HTML
except ImportError:
    pass


def text_diff(text1, text2):
    dmp = diff_match_patch()
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diffs)
    if _is_notebook():
        return HTML(dmp.diff_prettyHtml(diffs))
    _print_diffs(diffs, dmp)


def _print_diffs(diffs, dmp):
    text = ""
    for (op, data) in diffs:
        if op == dmp.DIFF_EQUAL:
            text += data
        elif op == dmp.DIFF_INSERT:
            text += _color_text(data, True)
        elif op == dmp.DIFF_DELETE:
            text += _color_text(data, False)
    print(text)


def _color_text(text, was_added):
    return colored(text, 'green' if was_added else 'red', attrs=["reverse"])


def _is_notebook():
    try:
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except NameError:
        pass
    return False
