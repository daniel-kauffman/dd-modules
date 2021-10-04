# Text Correction Library

import re

import spacy

import numconv


def correct_text(lines, numbers = True, capitalize = True, acronyms = True):
    """
    Perform word-to-number replacement and capitalize named entities.
    
    Args:
        lines: A list of str, each representing a line from a document.
        numbers: A bool indicating whether to perform number replacement.
        capitalize: A bool indicating whether to capitalize named entities.
    
    Returns:
        A list of str with text corrected.
    """
    if numbers:
        lines = numconv.convert_numbers(lines)
    if capitalize:
        lines = capitalize_entities(lines)
    if acronyms:
        lines = bill_types_to_acronyms(lines)
    return lines




def capitalize_entities(lines):
    """
    Capitalize all named entities found in the given list of lines.
    
    Args:
        lines: A list of str, each representing a line from a document.
    
    Returns:
        A list of str with capitalized named entities.
    """
    ner_list = ["PERSON", "NORP", "FACILITY", "ORG", "GPE", "LOC", "PRODUCT",
                "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]
    pos_list = ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]
    nlp = spacy.load("en")
    doc = nlp(" ".join(lines))
    update_dict = {}
    for ent in doc.ents:
        if ent.label_ in ner_list:
            tokens = []
            combine = False
            for token in ent:
                text = token.text
                if token.pos_ in ["PART", "PUNCT", "SYM"]:
                    fmt_str = r"(\s*){0}(\s*)"
                    match = re.search(fmt_str.format(text), ent.text)
                    if match.group(1) == "":
                        if len(tokens) == 0:
                            tokens.append(text)
                        else:
                            tokens[-1] = tokens[-1] + text
                    if match.group(2) == "":
                        combine = True
                else:
                    if token.pos_ in pos_list and not "A" <= text[0] <= "Z":
                        text = text.capitalize()
                    if combine:
                        tokens[-1] = tokens[-1] + text
                    else:
                        tokens.append(text)
                    combine = False
            capitalized = " ".join(tokens)
            if ent.text != capitalized:
                update_dict[ent.text] = capitalized
    updated_lines = []
    for line in lines:
        for old, new in update_dict.items():
            if old in line:
                line = line.replace(old, new)
        updated_lines.append(line)
    return updated_lines




def bill_types_to_acronyms(lines):
    """
    Convert all bill types into their acronym (e.g. "assembly bill" -> "ab")
    
    Args:
        lines: A list of str, each representing a line from a document.
    
    Returns:
        A list of str with bill types in acronym form.
    """
    update_dict = {}
    update_dict["assembly bill"] = "ab"
    update_dict["assembly bill number"] = "ab"
    update_dict["senate bill"] = "sb"
    update_dict["senate bill number"] = "sb"
    update_dict["house resolution"] = "hr"
    update_dict["house resolution number"] = "hr"
    
    updated_lines = []
    for line in lines:
        for old, new in update_dict.items():
            if old in line:
                line = line.replace(old, new)
        updated_lines.append(line)
    return updated_lines
