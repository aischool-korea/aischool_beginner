from bs4 import BeautifulSoup
import urllib.request
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

url = "https://www.imdb.com/title/tt4154796/reviews?ref_=tt_ov_rt"
htmlData = urllib.request.urlopen(url)

bs = BeautifulSoup(htmlData, 'lxml')

print(BeautifulSoup.prettify(bs))

title_list = bs.findAll('a', 'title')
review_list = bs.findAll('div', 'text show-more__control')
score_list = bs.findAll('span', 'rating-other-user-rating')

for title in title_list:
    print(title.getText())

for i, content in enumerate(review_list):
    print(i, content.getText()+"\n")

for score in score_list:
    print(score.span.getText())

print(len(title_list))
f = open("./data/review.txt", "w", encoding='UTF8')
for i in range(len(title_list)):
    f.write(clean_str(title_list[i].getText())+" "+clean_str(review_list[i].getText())+"\n")
f.close()

f = open("./data/score.txt", "w", encoding='UTF8')
for i in range(len(score_list)):
    f.write(score_list[i].span.getText()+"\n")
f.close()

