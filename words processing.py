def get_average_word_length(words):
    '''
    Get the average word length of a given words list
    :param words: A list containing words
    :return: The length
    '''
    assert isinstance(words,list)
    t=0
    num=0
    for i in words:
        assert isinstance(i,str)
        for j in i:
            assert j.isalpha()
        t+=len(i)
        num+=1
    return t/num



def get_longest_word(words):
    '''
    Get the longest word of a given words list
    :param words: A list containing words
    :return: The words
    '''
    assert isinstance(words,list)
    max=0
    if words:
        res=words[0]
    for i in words:
        assert isinstance(i, str)
        for j in i:
            assert j.isalpha()
        if max<len(i):
            max=len(i)
            res=i
    return res


def get_longest_words_startswith(words,start):
    '''
    Get the longest word of a given words list start with a given letter
    :param words: A list containing words
    :return: The words
    '''
    assert isinstance(words, list)
    assert start.isalpha()
    max=0
    if words:
        res=words[0]
    for i in words:
        assert isinstance(i, str)
        for j in i:
            assert j.isalpha()
        if i[0]==start:
            if max < len(i):
                max =len(i)
                res=i
    return res


def get_most_common_start(words):
    '''
    Get the most common start letter of a given words list
    :param words: A list containing words
    :return: The most common letter
    '''
    assert isinstance(words, list)
    num={}
    for i in words:
        assert isinstance(i, str)
        for j in i:
            assert j.isalpha()
        if i[0] in num:
            num[i[0]]+=1
        else:
            num[i[0]]=0
    max=0
    res=0
    for x,y in num.items():
        if max<y:
            max=y
            res=x
    return res


def get_most_common_end(words):
    '''
    Get the most common end letter of a given words list
    :param words: A list containing words
    :return: The most common letter
    '''
    assert isinstance(words, list)
    num={}
    for i in words:
        assert isinstance(i, str)
        for j in i:
            assert j.isalpha()
        if i[-1] in num:
            num[i[-1]]+=1
        else:
            num[i[-1]]=0
    max=0
    res=0
    for x,y in num.items():
        if max<y:
            max=y
            res=x
    return res
