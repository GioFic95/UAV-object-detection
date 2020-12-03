
class Color:  # pylint: disable=W0232
    GRAY = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    CRIMSON = 38


def colorize(num, string, bold=False, highlight=False):
    assert isinstance(num, int)
    attr = []
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
