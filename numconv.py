# Word-to-Number Conversion Library

import re


INT_DICT = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
            "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
            "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
            "ninety": 90, "hundred": 100}
POWERS = (("trillion", 12), ("billion", 9), ("million", 6), ("thousand", 3))


def convert_numbers(lines):
    """
    Detect and replace all spelled-out numerical words with digits.
    
    Args:
        lines: A list of str, each representing a line from a document.
    
    Returns:
        A list of str with words replaced by numbers.
    """
    keys = INT_DICT.keys() | set([word for word, _ in POWERS] + ["point"])
    pattern = re.compile(r"(\b(?:{0}|\s)\b){{3,}}".format(r"\b|\b".join(keys)))
    updated = []
    for line in lines:
        match = re.search(pattern, line)
        if match:
            tokens = match.group(0).strip().split()
            if len(tokens) > 1:
                parsed = parse_number(tokens)
                updated.append(line.replace(match.group(0).strip(), parsed))
            else:
                updated.append(line)
        else:
            updated.append(line)
    return updated


def parse_number(tokens):
    """
    Convert numerical words to digits in string format.
    
    Args:
        tokens: A list of str containing numerical words.
    
    Returns:
        A str containing a digit representation of the given tokens.
    """
    number = parse_irregular_int(tokens)
    if number is not None:
        number += parse_fraction(tokens)
        return str(number)
    is_money = ("dollar" in tokens or "dollars" in tokens or
                "cent" in tokens or "cents" in tokens)
    is_percent = "percent" in tokens
    keys = list(INT_DICT.keys()) + [word for word, _ in POWERS] + ["point"]
    tokens = [token for token in tokens if token in keys]
    int_tokens = (tokens if "point" not in tokens
                         else tokens[:tokens.index("point")])
    number = 0
    for word, power in POWERS:
        if word in int_tokens:
            index = tokens.index(word)
            base = 1 if index == 0 else parse_small_int(tokens[:index])
            number += base * 10 ** power
            tokens = tokens[index + 1:]
    number += parse_small_int(tokens)
    number += parse_fraction(tokens)
    if "point" in tokens:
        for word, power in POWERS:
            if word == tokens[-1]:
                number = int(number * 10 ** power)
    if is_money:
        return "${0:,}".format(number)
    elif is_percent:
        return "{0:,}%".format(number)
    elif number in range(2000, 2100):
        return str(number)
    return "{0:,}".format(number)


def parse_irregular_int(tokens):
    """
    Convert numerical words in an irregular format to digits in string format.
    An irregular format is defined as not using power words (e.g. thousands or
    millions) and instead only words for small numbers (e.g. twenty twenty).
    
    Args:
        tokens: A list of str containing numerical words.
    
    Returns:
        An int numerically representing the given tokens.
    """
    if "point" in tokens:
        tokens = tokens[:tokens.index("point")]
    if is_irregular(tokens):
        number = 0
        for i in range(len(tokens) - 1):
            power = 0
            for j in range(i + 1, len(tokens)):
                if INT_DICT[tokens[j]] >= 10:
                    power += 2
                elif INT_DICT[tokens[j]] == 0:
                    power += 1
                elif INT_DICT[tokens[j]] < 10 and INT_DICT[tokens[j - 1]] < 10:
                    power += 1
            number += INT_DICT[tokens[i]] * 10 ** power
        number += INT_DICT[tokens[-1]]
        return number


def is_irregular(tokens):
    """
    Determine whether the given numerical words are of irregular format.
    
    Args:
        tokens: A list of str containing numerical words.
    
    Returns:
        A bool indicating whether the given tokens are in irregular format.
    """
    ones = set(range(0, 10))
    tens = set(range(10, 20)) | set(range(20, 100, 10))
    keys = ["hundred"] + [word for word, _ in POWERS]
    if any(token in keys for token in tokens):
        return False
    for i in range(len(tokens) - 1):
        int_x, int_y = INT_DICT[tokens[i]], INT_DICT[tokens[i + 1]]
        if int_x in ones and int_y in ones:
            return True
        if int_x in ones and int_y in tens:
            return True
        if int_x in tens and int_y in tens:
            return True
    return False


def parse_small_int(tokens):
    """
    Convert numerical words representing a number below 1,000 to digits in
    string format.
    
    Args:
        tokens: A list of str containing numerical words.
    
    Returns:
        An int numerically representing the given tokens.
    """
    number = 0
    if "hundred" in tokens:
        index = tokens.index("hundred")
        hundreds = (1 if index == 0
                      else sum(INT_DICT[token] for token in tokens[:index]))
        number += hundreds * 100
        tokens = tokens[index + 1:]
    if "point" in tokens:
        tokens = tokens[:tokens.index("point")]
    return number + sum(INT_DICT[token] for token in tokens)


def parse_fraction(tokens):
    """
    Convert numerical words that make up the fractional component of a number
    to digits in string format.
    
    Args:
        tokens: A list of str containing numerical words.
    
    Returns:
        A str containing a fractional representation of the given tokens.
    """
    if "point" not in tokens:
        return 0
    else:
        tokens = [token for token in tokens
                        if token not in [word for word, _ in POWERS]]
        index = tokens.index("point")
        tokens = tokens[index + 1:]
        if len(tokens) == 0:
            frac_str = "0"
        else:
            digits_only = all(INT_DICT[token] < 10 for token in tokens)
            if digits_only:
                frac_str = "".join(str(INT_DICT[token]) for token in tokens)
            else:
                frac_str = str(parse_small_int(tokens))
    return float("0." + frac_str)
