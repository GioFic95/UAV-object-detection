import string


chars = string.digits + string.ascii_uppercase + string.ascii_lowercase


def preprocess():
    pass


if __name__ == '__main__':
    print(len(chars), chars)
    c = 30
    print(chars[c-1])
