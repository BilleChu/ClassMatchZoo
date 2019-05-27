import os
import sys
import re

# 判断一个unicode是否是汉字
def is_chinese(uchar):
    if '\u4e00' <= uchar<='\u9fff':
        return True
    else:
        return False

# 判断一个unicode是否是数字
def is_number(uchar):
    if '\u0030' <= uchar <='\u0039':
        return True
    else:
        return False

# 判断一个unicode是否是英文字母
def is_alphabet(uchar):
    if ('\u0041' <= uchar<='\u005a') or ('\u0061' <= uchar<='\u007a'):
        return True
    else:
        return False

# 判断是否是符号
def is_symbol(uchar):
    if ('\u0020' <= uchar <= '\u002F') or ('\u003A' <= uchar <= '\u0040') or ('\u005B' <= uchar <='\u0060') or ('\u007B'<= uchar <='\u007E'):
        return True
    else:
        return False

# 判断是否是长于指定size的纯英文和数字组合
def is_pureNumAlpha(ustr, size):
    module = "[0-9a-z]{" + str(size) + ",}"
    if re.sub(module, "", ustr) == "":
        return True
    else:
        return False

# 判断是否是长于指定size的纯数字组合
def is_pureNum(ustr, size):
    module = "[0-9]{" + str(size) + ",}"
    if re.sub(module, "", ustr) == "":
        return True
    else:
        return False

#判断是否是长于指定size的纯数字组合
def is_pureNum(ustr, size):
    module = "[0-9]{" + str(size) + ",}"
    if re.sub(module, "", ustr) == "":
        return True
    else:
        return False

# 判断是否非汉字，数字和英文字符and symbol
def is_other(uchar):
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar) or is_symbol(uchar)):
        return True
    else:
        return False

# 判断是字符&全角转换成半角字符
def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return "".join(ss)

# delete unseen characters
def deleteSpecialCharacters(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            if is_other(uchar):
                pass
            else:
                ss.append(uchar)
    return "".join(ss)

def filterSymbols(ustring):
    pattern = '[\"\'\\\+`<\.!/_,#\$%^\*()~&;:\|\?<>@\[\]{}\-_=]+'
    return re.sub(pattern, "", ustring)


def filterNums(ustr, size):
    module = "[\d]{" + str(size) + "5,}"
    return re.sub(module, "", ustr)

def transform(ustr):
    ustr = ustr.lower()
    ustr = strQ2B(ustr)
    ustr = filterSymbols(ustr)
    ustr = deleteSpecialCharacters(ustr)
    if is_pureNum(ustr, 5):
        return ""
    return ustr

def readfile(filename):
    chunksize = 10
    with open(filename, "r") as f:
        while True:
            lines = f.readline()
            if not lines:
                break
            yield lines

def process(filename, savefile):
    for line in readfile(filename):
        with open(savefile, "a") as f:
            line = transform(line)
            f.write(line + "\n")

if __name__ == '__main__':
    filename = sys.argv[1]
    savefile = sys.argv[2]
    process(filename, savefile)
