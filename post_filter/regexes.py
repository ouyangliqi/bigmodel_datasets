#!/usr/bin/env python
import argparse
import json
import os
import random
import re
import sys
from multiprocessing import Pool

import ftfy
import tqdm

"""
This file is meant to be run on the /data/xlmg/gptz/corpus_dedup_10_10_1_0.05/ folder.
"""

# ---- CORPUS SPECIFIC
def raw_bookscorpus(text):
    text = re.sub(r'^Published by.*$', '', text, flags=re.M)
    text = re.sub(r'^Copyright.*$', '', text, flags=re.M)
    text = re.sub(r'^Edited by.*$', '', text, flags=re.M)
    text = re.sub(r'^Smashwords [eE]dition.*$', '', text, flags=re.M)
    text = re.sub(r'^All rights reserved.*$', '', text, flags=re.M)
    text = re.sub(r'^(Part|PART|Chapter|CHAPTER) (I+|i+|\d+)(:.*)?$', '', text, flags=re.M)
    text = re.sub(r'^ISBN.*$', '', text, flags=re.M)
    text = re.sub(r'^[# \-\*]+$', '', text, flags=re.M)
    return text

def remove_markdown_links(text):
    # there's a lot of broken links sadly
    text = re.sub(r'(https?://[^ ]+?-)\n', r'\1', text, flags=re.DOTALL)
    # now strip down markdown links to just the caption
    text = re.sub(r'\[([^\]]*)\]\(([^\)]*)\)', r'\1', text)
    # sometimes that markdown is a no-op and then let's just kill it
    text = re.sub(r'(https?://[^ ]+\.\.\.)', r'', text)
    text = strip_trailing_whitespace(text)
    return text

def hackernews(text):
    text = text.strip()

    # remove usernames
    text = re.sub('\n(~~~|======|------)\n[\w\-]+\n', '\n===POSTBREAK===\n\n', text)

    # get rid of markdown links
    text = remove_markdown_links(text)

    # second line is always the URL
    lines = text.split('\n')
    lines.pop(1)
    # and first line has the username
    try:
        lines[0] = lines[0][:lines[0].rindex(' -')]
    except:
        pass

    # remove any quoteblocks
    lines = [l for l in lines if not l.startswith('> ')]

    text = '\n'.join(lines)

    # lots of people like to leave footnotes
    text = re.sub(r'^\[\d+\]:? .*$', '', text, flags=re.M)

    # remove links to other hackernews
    text = re.sub('<https?://[^>]*>', '', text)

    text = unwrap_like_indents(text, min_length=1000)

    # kill some leading whitespace
    text = re.sub(r'^ *', '', text, flags=re.M)

    # drop down multi-paragraph posts into one
    text = re.sub(r'\n\n+', '\n', text)

    # and now bring back the semantic breaks
    text = text.replace('===POSTBREAK===', '')

    return text

def unindent_blocks(text):
    """
    Unindent well-formatted blocks of text but preserving lines.

    Useful for removing formatting from pg19 but preserving lines
    corresponding to poetry, etc.
    """
    return re.sub(r'^[ ]*', '', text, flags=re.M)

def remove_toc_numbers(text):
    # remove toc
    text = re.sub(r'[ ]{4,}[\d\-]+$', '', text, flags=re.M)
    # remove footnote references
    text = re.sub(r'\[\d+\]', '', text)
    # headers
    text = re.sub(r'^[\* ]+$', '', text, flags=re.M)
    # remove anything with a remaining large amount of whitespace
    lines = text.split('\n')
    lines = [l for l in lines if '     ' not in l]
    text = '\n'.join(lines)
    return text

def unwrap_like_indents(text, min_length=100):
    """
    WARNING: only use this one on corpora you KNOW are line wrapped.
    """
    paragraphs = text.split('\n\n')
    unwrapped = []
    for paragraph in paragraphs:
        lines = paragraph.split('\n')
        if len(lines) == 1:
            # quick exit on this one
            unwrapped.append(paragraph)
            continue

        # awkward way of getting out indentation from first line as a string
        indent_length = len(lines[0]) - len(lines[0].lstrip())
        indentation = lines[0][:indent_length]

        # check that all lines are all wrapped at roughly 100 chars
        all_short = all(len(l) <= min_length for l in lines)

        # check if every line has the same indentation
        all_indented = all(l.startswith(indentation) for l in lines)
        # and we're not seeming to be in a spot of very manicured formatting
        not_same_length = len(set(len(l) for l in lines)) != 1

        # and we're not over indented lol
        for line in lines:
            unindented = line[indent_length:]
            if len(unindented.lstrip()) < len(unindented):
                all_indented = False
                break

        if not (all_indented and all_short and not_same_length):
            # doesn't meet the rule. Don't unwrap this
            unwrapped.append(paragraph)
            continue

        # okay time to unwrap
        newpar = indentation + ' '.join(l[len(indentation):] for l in lines)
        unwrapped.append(newpar)

    return '\n\n'.join(unwrapped)

def strip_single_space(text):
    if text.startswith(' ') and not text.startswith('  '):
        return text[1:]
    else:
        return text

def pg19_strips(text):
    # unwrap paragraphs
    text = unwrap_like_indents(text)

    # remove illustrations
    text = re.sub('^\s*\[Illustration.*$', '', text, flags=re.M | re.I)
    # remove copyright, produced by, etc
    text = re.sub('^\s*Produced by .*$', '', text, flags=re.M | re.I)
    # remove copyright, produced by, etc
    text = re.sub('^\s*Copyright .*$', '', text, flags=re.M | re.I)

    # get rid of any project gutenberg headers etc
    paragraphs = text.split('\n\n')
    paragraphs = [p for p in paragraphs if 'Project Gutenberg' not in p]
    paragraphs = [p for p in paragraphs if "Transcriber's note" not in p]
    text = '\n\n'.join(paragraphs)

    # PG19 uses 3+ newlines to separate sections, 2+ lines to separate paragraphs
    paragraphs = re.split(r'\n{3,}', text)
    paragraphs = [p.replace('\n\n', '\n') for p in paragraphs]
    text = '\n\n'.join(paragraphs)

    # unindent any blocks like poetry
    text = unindent_blocks(text)

    # pull out any table of contents kinda stuff
    text = remove_toc_numbers(text)

    # kill any leftover gutenberg trash
    text = re.sub('End of the Project Gutenberg EBook.*$', '', text, flags=re.DOTALL)

    # and also any separator lines
    text = re.sub(r'^\s*\*+\s*$', '', text, flags=re.M)
    text = re.sub(r'^\s*(\*\s )\*\s*$', '', text, flags=re.M)

    # remove chapters
    text = re.sub(
        # optional labels + some numerals, maybe even roman ones
        r'^\s*((part|section|chapter) )?([\w\-]+)\s*$',
        '',
        text,
        flags=re.M | re.I
    )

    lines = text.split("\n")
    lines = [strip_single_space(l) for l in lines]
    text = "\n".join(lines)


    return text


def opensubtitles_fix_newlines(text):
    # add a newline between turns
    text = text.replace('" "', '"\n"')
    # Some erronious white space
    text = text.replace('\n" ', '\n"')
    # look for repeated lines in a row
    lines = text.split("\n")
    i = len(lines) - 1
    while i > 0:
        lines_i = lines[i]
        lines_imin1 = lines[i - 1]
        if lines_i == lines_imin1:
            lines.pop(i)
        i -= 1
    text = '\n'.join(lines)
    # strip quotes
    text = re.sub('^"(.*)"$', r'\1', text, flags=re.M)
    # remove stage directions
    text = re.sub(r'^\[(.*)\]$', r'', text, flags=re.M)
    # collapse back newlines
    text = re.sub(r'\n+', '\n', text)
    return text

def wikipedia_cleanup(text):
    # get rid of category links at the end
    return re.sub(r'^Category:.*$', '', text, flags=re.MULTILINE)

def dm(text):
    lines = text.split('\n')
    # first 2 lines and last line are always trashed
    return '\n'.join(lines[2:-1])

def stackex(text):
    text = text.replace('Q:\n\n', 'Question:\n')
    text = text.replace('A:\n\n', 'Answer:\n')
    return text

# ---- GENERIC CLEANUPS

def myle_ftfy(text):
    return ftfy.fix_text(text, uncurl_quotes=False, fix_entities=False)

def collapse_triple_newlines(text):
    """
    Replaces 3 or more newlines in a row with a double newline.
    """
    return re.sub(r'[\n]{2,}', '\n\n', text)

def fix_encoding_screwup(text):
    return json.loads('"' + text + '"')

def normalize_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")

def strip_trailing_whitespace(text):
    return re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

# ---- COORDINATION

def fixup(text, fname):

    # if 'BookCorpusFair' not in fname and 'bookcorpus/orig' not in fname:
    #     text = fix_encoding_screwup(text)

    text = normalize_newlines(text)
    # text = myle_ftfy(text)

    if 'Gutenberg' in fname:
        text = pg19_strips(text)

    text = strip_trailing_whitespace(text)
    text = collapse_triple_newlines(text)

    if 'OpenSubtitles' in fname:
        text = opensubtitles_fix_newlines(text)
    if 'Wikipedia' in fname:
        text = wikipedia_cleanup(text)
    if 'HackerNews' in fname:
        text = hackernews(text)
    if 'BookCorpusFair' in fname or 'bookcorpus/orig' in fname:
        text = raw_bookscorpus(text)
    if 'DM_Mathematics' in fname:
        text = dm(text)
    if 'StackExchange' in fname:
        text = stackex(text)

    # double check lol
    text = collapse_triple_newlines(text)

    # finally strip any leading/trailing whitespace in the document
    text = text.strip()

    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    args = parser.parse_args()

    for i, input_ in enumerate(args.inputs, 1):
        output = input_.replace(
            "/path/to/input",
            "/path/to/output"
        )
        lines = []
        print(f"[{i-1}/{len(args.inputs)}] Starting {input_}")
        print(f"Output will go to {output}")
        with tqdm.tqdm(total=os.path.getsize(input_), desc="  Input") as pbar:
            with open(input_) as f:
                for line in f:
                    lines.append((line, input_))
                    pbar.update(len(line))
        pool = Pool(100)
        results = pool.starmap(fixup, tqdm.tqdm(lines, desc="Process"))
        with open(output, 'w') as f:
            for doc in tqdm.tqdm(results, desc=" Output"):
                f.write(doc)
                f.write('\n')
        print(f"[{i}/{len(args.inputs)}] Finished processing {input_}")
        print()
    print("Success")

if __name__ == '__main__':
    main()
