import json
import re


def remove_codes(text):
    # remove html
    # print("1")
    pattern = re.compile(r'((<br>)|(<br>)|(<b>)|(<li>)|(<ol>)|(<blockquote>))')
    text = pattern.sub('', text)

    # remove .
    # print("2")
    pattern = re.compile(r'[\.]{6,}')
    text = pattern.sub('', text)

    # 重复模式匹配，删除连续重复三次或以上的模式，保留一次模式
    # print("3")
    pattern = re.compile(r'(.+?)\1\1+')
    text = pattern.sub(r'\1', text)

    # (REMOVE <SCRIPT> to </script> and variations)
    # print("4")
    # if "script" in text or "html" in text or "style" in text or "div" in text:
    pattern = r'<\s*[script|html|style|div][^>]*>.*?<\s*\/\s*[script|html|style|div]\s*>'  # mach any char zero or more times
    text = "" if re.findall(pattern, text) else text

    # # (REMOVE HTML <STYLE> to </style> and variations)
    # print("6")
    # pattern = r'<[ ]*style.*?\/[ ]*style[ ]*>'  # mach any char zero or more times
    # text = re.sub(pattern, '', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # # (REMOVE HTML COMMENTS <!-- to --> and variations)
    # print("7")
    pattern = r'<[ ]*!--.*?--[ ]*>'  # mach any char zero or more times
    text = re.sub(pattern, '', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    return text


if __name__ == "__main__":
    text = json.load(open("test.json"))['cont']

    for i in text:
        retruned = remove_codes(i)
        if len(i) != len(retruned):
            print(i)
            print("\n")
            print(retruned)
