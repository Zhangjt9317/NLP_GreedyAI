#!/usr/bin/env python
# coding: utf-8

# ## 搭建一个简单的问答系统 （Building a Simple QA System）

# 本次项目的目标是搭建一个基于检索式的简易的问答系统，这是一个最经典的方法也是最有效的方法。  
# 
# ```不要单独创建一个文件，所有的都在这里面编写，不要试图改已经有的函数名字 （但可以根据需求自己定义新的函数）```
# 
# ```预估完成时间```： 5-10小时

# #### 检索式的问答系统
# 问答系统所需要的数据已经提供，对于每一个问题都可以找得到相应的答案，所以可以理解为每一个样本数据是 ``<问题、答案>``。 那系统的核心是当用户输入一个问题的时候，首先要找到跟这个问题最相近的已经存储在库里的问题，然后直接返回相应的答案即可（但实际上也可以抽取其中的实体或者关键词)。 举一个简单的例子：
# 
# 假设我们的库里面已有存在以下几个<问题,答案>：
# - <"贪心学院主要做什么方面的业务？”， “他们主要做人工智能方面的教育”>
# - <“国内有哪些做人工智能教育的公司？”， “贪心学院”>
# - <"人工智能和机器学习的关系什么？", "其实机器学习是人工智能的一个范畴，很多人工智能的应用要基于机器学习的技术">
# - <"人工智能最核心的语言是什么？"， ”Python“>
# - .....
# 
# 假设一个用户往系统中输入了问题 “贪心学院是做什么的？”， 那这时候系统先去匹配最相近的“已经存在库里的”问题。 那在这里很显然是 “贪心学院是做什么的”和“贪心学院主要做什么方面的业务？”是最相近的。 所以当我们定位到这个问题之后，直接返回它的答案 “他们主要做人工智能方面的教育”就可以了。 所以这里的核心问题可以归结为计算两个问句（query）之间的相似度。

# #### 项目中涉及到的任务描述
# 问答系统看似简单，但其中涉及到的内容比较多。 在这里先做一个简单的解释，总体来讲，我们即将要搭建的模块包括：
# 
# - 文本的读取： 需要从相应的文件里读取```(问题，答案)```
# - 文本预处理： 清洗文本很重要，需要涉及到```停用词过滤```等工作
# - 文本的表示： 如果表示一个句子是非常核心的问题，这里会涉及到```tf-idf```, ```Glove```以及```BERT Embedding```
# - 文本相似度匹配： 在基于检索式系统中一个核心的部分是计算文本之间的```相似度```，从而选择相似度最高的问题然后返回这些问题的答案
# - 倒排表： 为了加速搜索速度，我们需要设计```倒排表```来存储每一个词与出现的文本
# - 词义匹配：直接使用倒排表会忽略到一些意思上相近但不完全一样的单词，我们需要做这部分的处理。我们需要提前构建好```相似的单词```然后搜索阶段使用
# - 拼写纠错：我们不能保证用户输入的准确，所以第一步需要做用户输入检查，如果发现用户拼错了，我们需要及时在后台改正，然后按照修改后的在库里面搜索
# - 文档的排序： 最后返回结果的排序根据文档之间```余弦相似度```有关，同时也跟倒排表中匹配的单词有关
# 

# #### 项目中需要的数据：
# 1. ```dev-v2.0.json```: 这个数据包含了问题和答案的pair， 但是以JSON格式存在，需要编写parser来提取出里面的问题和答案。 
# 2. ```glove.6B```: 这个文件需要从网上下载，下载地址为：https://nlp.stanford.edu/projects/glove/， 请使用d=200的词向量
# 3. ```spell-errors.txt``` 这个文件主要用来编写拼写纠错模块。 文件中第一列为正确的单词，之后列出来的单词都是常见的错误写法。 但这里需要注意的一点是我们没有给出他们之间的概率，也就是p(错误|正确），所以我们可以认为每一种类型的错误都是```同等概率```
# 4. ```vocab.txt``` 这里列了几万个英文常见的单词，可以用这个词库来验证是否有些单词被拼错
# 5. ```testdata.txt``` 这里搜集了一些测试数据，可以用来测试自己的spell corrector。这个文件只是用来测试自己的程序。

# 在本次项目中，你将会用到以下几个工具：
# - ```sklearn```。具体安装请见：http://scikit-learn.org/stable/install.html  sklearn包含了各类机器学习算法和数据处理工具，包括本项目需要使用的词袋模型，均可以在sklearn工具包中找得到。 
# - ```jieba```，用来做分词。具体使用方法请见 https://github.com/fxsjy/jieba
# - ```bert embedding```: https://github.com/imgarylai/bert-embedding
# - ```nltk```：https://www.nltk.org/index.html

# ### 第一部分：对于训练数据的处理：读取文件和预处理

# - ```文本的读取```： 需要从文本中读取数据，此处需要读取的文件是```dev-v2.0.json```，并把读取的文件存入一个列表里（list）
# - ```文本预处理```： 对于问题本身需要做一些停用词过滤等文本方面的处理
# - ```可视化分析```： 对于给定的样本数据，做一些可视化分析来更好地理解数据

# #### 1.1节： 文本的读取
# 把给定的文本数据读入到```qlist```和```alist```当中，这两个分别是列表，其中```qlist```是问题的列表，```alist```是对应的答案列表

# In[1]:


import sys
#sys.path.append('/home/ubuntu/MyFiles/.local/lib/python3.5/site-packages:/usr/local/lib/python3.5/dist-packages:/usr/lib/python3.5/site-packages:/usr/lib/python3/dist-packages')
import json
def read_corpus():
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。 在此过程中，不用对字符换做任何的处理（这部分需要在 Part 2.3里处理）
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）
    """
    # TODO 需要完成的代码部分 ...
    qlist = []
    alist = []
    filename = 'train-v2.0.json'
    datas = json.load(open(filename,'r'))
    data = datas['data']
    for d in data:
        paragraph = d['paragraphs']
        for p in paragraph:
            qas = p['qas']
            for qa in qas:
                #print(qa)
                #处理is_impossible为True时answers空
                if(not qa['is_impossible']):
                    qlist.append(qa['question'])
                    alist.append(qa['answers'][0]['text'])
    #print(qlist[0])
    #print(alist[0])
    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist
qlist,alist = read_corpus()


# #### 1.2 理解数据（可视化分析/统计信息）
# 对数据的理解是任何AI工作的第一步， 需要对数据有个比较直观的认识。在这里，简单地统计一下：
# 
# - 在```qlist```出现的总单词个数
# - 按照词频画一个```histogram``` plot

# In[2]:


# TODO: 统计一下在qlist中总共出现了多少个单词？ 总共出现了多少个不同的单词(unique word)？
#       这里需要做简单的分词，对于英文我们根据空格来分词即可，其他过滤暂不考虑（只需分词）
words_qlist = dict()
for q in qlist:
    #以空格为分词，都转为小写
    words = q.strip().split(' ')
    for w in words:
        if w.lower() in words_qlist:
            words_qlist[w.lower()] += 1
        else:
            words_qlist[w.lower()] = 1
word_total = len(words_qlist)
print (word_total)


# In[4]:


# TODO: 统计一下qlist中出现1次，2次，3次... 出现的单词个数， 然后画一个plot. 这里的x轴是单词出现的次数（1，2，3，..)， y轴是单词个数。
#       从左到右分别是 出现1次的单词数，出现2次的单词数，出现3次的单词数..
import matplotlib.pyplot as plt
import numpy as np
#counts：key出现N次，value：出现N次词有多少
counts = dict()
for w,c in words_qlist.items():
    if c in counts:
        counts[c] += 1
    else:
        counts[c] = 1
#print(counts)
#以histogram画图
fig,ax = plt.subplots()
ax.hist(counts.values(),bins = np.arange(0,250,25),histtype='step',alpha=0.6,label="counts")
ax.legend()
ax.set_xlim(0,250)
ax.set_yticks(np.arange(0,220,20))
plt.show()


# In[4]:


# TODO： 从上面的图中能观察到什么样的现象？ 这样的一个图的形状跟一个非常著名的函数形状很类似，能所出此定理吗？ 
#       hint: [XXX]'s law
# 
# 高斯(正态)分布的右半部分？


# #### 1.3 文本预处理
# 此部分需要做文本方面的处理。 以下是可以用到的一些方法：
# 
# - 1. 停用词过滤 （去网上搜一下 "english stop words list"，会出现很多包含停用词库的网页，或者直接使用NLTK自带的）   
# - 2. 转换成lower_case： 这是一个基本的操作   
# - 3. 去掉一些无用的符号： 比如连续的感叹号！！！， 或者一些奇怪的单词。
# - 4. 去掉出现频率很低的词：比如出现次数少于10,20.... （想一下如何选择阈值）
# - 5. 对于数字的处理： 分词完只有有些单词可能就是数字比如44，415，把所有这些数字都看成是一个单词，这个新的单词我们可以定义为 "#number"
# - 6. lemmazation： 在这里不要使用stemming， 因为stemming的结果有可能不是valid word。
# 

# In[6]:


# TODO： 需要做文本方面的处理。 从上述几个常用的方法中选择合适的方法给qlist做预处理（不一定要按照上面的顺序，不一定要全部使用）
import nltk
from nltk.corpus import stopwords
import codecs
import re

def tokenizer(ori_list):
    #分词时处理标点符号
    SYMBOLS = re.compile('[\s;\"\",.!?\\/\[\]\{\}\(\)-]+')
    new_list = []
    for q in ori_list:
        words = SYMBOLS.split(q.lower().strip())
        new_list.append(' '.join(words))
    return new_list

def removeStopWord(ori_list):
    new_list = []
    #nltk中stopwords包含what等，但是在QA问题中，这算关键词，所以不看作关键词
    restored = ['what','when','which','how','who','where']
    english_stop_words = list(set(stopwords.words('english')))#['what','when','which','how','who','where','a','an','the'] #
    for w in restored:
        english_stop_words.remove(w)
    for q in ori_list:
        sentence = ' '.join([w for w in q.strip().split(' ') if w not in english_stop_words])
        new_list.append(sentence)
    return new_list

def removeLowFrequence(ori_list,vocabulary,thres = 10):
    #根据thres筛选词表，小于thres的词去掉
    new_list = []
    for q in ori_list:
        sentence = ' '.join([w for w in q.strip().split(' ') if vocabulary[w] >= thres])
        new_list.append(sentence)
    return new_list

def replaceDigits(ori_list,replace = '#number'):
    #将数字统一替换为replace,默认#number
    DIGITS = re.compile('\d+')
    new_list = []
    for q in ori_list:
        q = DIGITS.sub(replace,q)
        new_list.append(q)
    return new_list

def createVocab(ori_list):
    count = 0
    vocab_count = dict()
    for q in ori_list:
        words = q.strip().split(' ')
        count += len(words)
        for w in words:
            if w in vocab_count:
                vocab_count[w] += 1
            else:
                vocab_count[w] = 1
    return vocab_count,count
def writeFile(oriList,filename):
    with codecs.open(filename,'w','utf8') as Fout:
        for q in oriList:
            Fout.write(q + u'\n')

def writeVocab(vocabulary,filename):
    sortedList = sorted(vocabulary.items(),key = lambda d:d[1])
    with codecs.open(filename,'w','utf8') as Fout:
        for (w,c) in sortedList:
            Fout.write(w + u':' + str(c) + u'\n')            
new_list = tokenizer(qlist)
#writeFile(qlist,'ori.txt')

new_list = removeStopWord(new_list)
#writeFile(new_list,'removeStop.txt')
new_list = replaceDigits(new_list)
#writeFile(new_list,'removeDigts.txt')
vocabulary,count = createVocab(new_list)
new_list = removeLowFrequence(new_list,vocabulary,5)
#writeFile(new_list,'lowFrequence.txt')
#重新统计词频
vocab_count,count = createVocab(new_list)
writeVocab(vocab_count,"train.vocab")
qlist = new_list
#qlist =     # 更新后的问题列表


# ### 第二部分： 文本的表示
# 当我们做完必要的文本处理之后就需要想办法表示文本了，这里有几种方式
# 
# - 1. 使用```tf-idf vector```
# - 2. 使用embedding技术如```word2vec```, ```bert embedding```等
# 
# 下面我们分别提取这三个特征来做对比。 

# #### 2.1 使用tf-idf表示向量
# 把```qlist```中的每一个问题的字符串转换成```tf-idf```向量, 转换之后的结果存储在```X```矩阵里。 ``X``的大小是： ``N* D``的矩阵。 这里``N``是问题的个数（样本个数），
# ``D``是词典库的大小

# In[7]:


# TODO
import numpy as np

def computeTF(vocab,c):
    #计算每次词的词频
    #vocabCount已经统计好的每词的次数
    #c是统计好的总次数
    TF = np.ones(len(vocab))
    word2id = dict()
    id2word = dict()
    for word,fre in vocab.items():
        TF[len(word2id)] = 1.0 * fre / c
        id2word[len(word2id)] = word
        word2id[word] = len(word2id)
    return TF,word2id,id2word

def computeIDF(word2id,qlist):
    #IDF计算，没有类别，以句子为一个类
    IDF = np.ones(len(word2id))
    for q in qlist:
        words = set(q.strip().split())
        for w in words:
            IDF[word2id[w]] += 1
    IDF /= len(qlist)
    IDF = -1.0 * np.log2(IDF)
    return IDF

def computeSentenceEach(sentence,tfidf,word2id):
    #给定句子，计算句子TF-IDF
    #tfidf是一个1*M的矩阵,M为词表大小
    #不在词表中的词不统计
    sentence_tfidf = np.zeros(len(word2id))
    for w in sentence.strip().split(' '):
        if w not in word2id:
            continue
        sentence_tfidf[word2id[w]] = tfidf[word2id[w]]
    return sentence_tfidf

def computeSentence(qlist,word2id,tfidf):
    #对所有句子分别求tfidf
    X_tfidf = np.zeros((len(qlist),len(word2id)))
    for i,q in enumerate(qlist):
        X_tfidf[i] = computeSentenceEach(q,tfidf,word2id)
        #print(X_tfidf[i])
    return X_tfidf

TF,word2id,id2word = computeTF(vocab_count,count)
print(len(word2id))
IDF = computeIDF(word2id,qlist)
#用TF，IDF计算最终的tf-idf
vectorizer = np.multiply(TF,IDF)# 定义一个tf-idf的vectorizer
#print(vectorizer)
X_tfidf =  computeSentence(qlist,word2id,vectorizer) # 结果存放在X矩阵里
print(X_tfidf[0])
print(X_tfidf.shape)


# 

# #### 2.2 使用wordvec + average pooling
# 词向量方面需要下载： https://nlp.stanford.edu/projects/glove/ （请下载``glove.6B.zip``），并使用``d=200``的词向量（200维）。国外网址如果很慢，可以在百度上搜索国内服务器上的。 每个词向量获取完之后，即可以得到一个句子的向量。 我们通过``average pooling``来实现句子的向量。 

# In[27]:


# TODO 基于Glove向量获取句子向量
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
    
def loadEmbedding(filename):
    #加载glove模型，转化为word2vec，再加载word2vec模型
    word2vec_temp_file = 'word2vec_temp.txt'
    glove2word2vec(filename,word2vec_temp_file)
    model = KeyedVectors.load_word2vec_format(word2vec_temp_file)
    return model

def computeGloveSentenceEach(sentence,embedding):
    #查找句子中每个词的embedding,将所有embedding进行加和求均值
    emb = np.zeros(200)
    words = sentence.strip().split(' ')
    for w in words:
        if w not in embedding:
            #没有lookup的即为unknown
            w = 'unknown'
        #emb += embedding.get_vector(w)
        emb += embedding[w]
    return emb / len(words)

def computeGloveSentence(qlist,embedding):
    #对每一个句子进行求均值的embedding
    X_w2v = np.zeros((len(qlist),200))
    for i,q in enumerate(qlist):
        X_w2v[i] = computeGloveSentenceEach(q,embedding)
        #print(X_w2v)
    return X_w2v
emb  =  loadEmbedding('glove.6B.200d.txt')# 这是 D*H的矩阵，这里的D是词典库的大小， H是词向量的大小。 这里面我们给定的每个单词的词向量，
        # 这需要从文本中读取
    
X_w2v =   computeGloveSentence(qlist,emb)# 初始化完emb之后就可以对每一个句子来构建句子向量了，这个过程使用average pooling来实现


# #### 2.3 使用BERT + average pooling
# 最近流行的BERT也可以用来学出上下文相关的词向量（contex-aware embedding）， 在很多问题上得到了比较好的结果。在这里，我们不做任何的训练，而是直接使用已经训练好的BERT embedding。 具体如何训练BERT将在之后章节里体会到。 为了获取BERT-embedding，可以直接下载已经训练好的模型从而获得每一个单词的向量。可以从这里获取： https://github.com/imgarylai/bert-embedding , 请使用```bert_12_768_12```	当然，你也可以从其他source获取也没问题，只要是合理的词向量。 

# In[23]:


# TODO 基于BERT的句子向量计算
from bert_embedding import BertEmbedding
sentence_embedding = np.ones((len(qlist),768))
#加载Bert模型，model，dataset_name,须指定
bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_cased')
#查询所有句子的Bert  embedding
#all_embedding = []
#for q in qlist:
#    all_embedding.append(bert_embedding([q],'sum'))
all_embedding = bert_embedding(qlist,'sum')
for i in range(len(all_embedding)):
    #print(all_embedding[i][1])
    sentence_embedding[i] = np.sum(all_embedding[i][1],axis = 0) / len(q.strip().split(' '))
    if i == 0:
        print(sentence_embedding[i])

X_bert =  sentence_embedding # 每一个句子的向量结果存放在X_bert矩阵里。行数为句子的总个数，列数为一个句子embedding大小。 


# ### 第三部分： 相似度匹配以及搜索
# 在这部分里，我们需要把用户每一个输入跟知识库里的每一个问题做一个相似度计算，从而得出最相似的问题。但对于这个问题，时间复杂度其实很高，所以我们需要结合倒排表来获取相似度最高的问题，从而获得答案。

# In[ ]:





# #### 3.1 tf-idf + 余弦相似度
# 我们可以直接基于计算出来的``tf-idf``向量，计算用户最新问题与库中存储的问题之间的相似度，从而选择相似度最高的问题的答案。这个方法的复杂度为``O(N)``， ``N``是库中问题的个数。

# In[16]:


import queue as Q
#优先级队列实现大顶堆Heap,每次输出都是相似度最大值
que = Q.PriorityQueue()
def cosineSimilarity(vec1,vec2):
    #定义余弦相似度
    return np.dot(vec1,vec2.T)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2)))

def get_top_results_tfidf_noindex(query):
    # TODO 需要编写
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 对于用户的输入 query 首先做一系列的预处理(上面提到的方法)，然后再转换成tf-idf向量（利用上面的vectorizer)
    2. 计算跟每个库里的问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    top = 5
    query_tfidf = computeSentenceEach(query.lower(),vectorizer,word2id)
    for i,vec in enumerate(X_tfidf):
        result = cosineSimilarity(vec,query_tfidf)
        #print(result)
        que.put((-1 * result,i))
    i = 0
    
    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下标 
                   # hint: 请使用 priority queue来找出top results. 思考为什么可以这么做？ 
    while(i < top and not que.empty()):
        top_idxs.append(que.get()[1])
        i += 1
    print(top_idxs)
    return np.array(alist)[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案 
results = get_top_results_tfidf_noindex('In what city and state did Beyonce  grow up')
print(results)


# In[17]:


# TODO: 编写几个测试用例，并输出结果
print (get_top_results_tfidf_noindex("When did Beyonce start becoming popular"))
#GT:in the late 1990s
print (get_top_results_tfidf_noindex("What counted for more of the population change"))
#GT:births and deaths


# 你会发现上述的程序很慢，没错！ 是因为循环了所有库里的问题。为了优化这个过程，我们需要使用一种数据结构叫做```倒排表```。 使用倒排表我们可以把单词和出现这个单词的文档做关键。 之后假如要搜索包含某一个单词的文档，即可以非常快速的找出这些文档。 在这个QA系统上，我们首先使用倒排表来快速查找包含至少一个单词的文档，然后再进行余弦相似度的计算，即可以大大减少```时间复杂度```。

# #### 3.2 倒排表的创建
# 倒排表的创建其实很简单，最简单的方法就是循环所有的单词一遍，然后记录每一个单词所出现的文档，然后把这些文档的ID保存成list即可。我们可以定义一个类似于```hash_map```, 比如 ``inverted_index = {}``， 然后存放包含每一个关键词的文档出现在了什么位置，也就是，通过关键词的搜索首先来判断包含这些关键词的文档（比如出现至少一个），然后对于candidates问题做相似度比较。

# In[11]:


# TODO 请创建倒排表
word_doc = dict()
#key:word,value:包含该词的句子序号的列表
for i,q in enumerate(qlist):
    words = q.strip().split(' ')
    for w in set(words):
        if w not in word_doc:
            #没在word_doc中的，建立一个空list
            word_doc[w] = set([])    
        word_doc[w] = word_doc[w] | set([i])
inverted_idx = word_doc  # 定一个一个简单的倒排表，是一个map结构。 循环所有qlist一遍就可以


# #### 3.3 语义相似度
# 这里有一个问题还需要解决，就是语义的相似度。可以这么理解： 两个单词比如car, auto这两个单词长得不一样，但从语义上还是类似的。如果只是使用倒排表我们不能考虑到这些单词之间的相似度，这就导致如果我们搜索句子里包含了``car``, 则我们没法获取到包含auto的所有的文档。所以我们希望把这些信息也存下来。那这个问题如何解决呢？ 其实也不难，可以提前构建好相似度的关系，比如对于``car``这个单词，一开始就找好跟它意思上比较类似的单词比如top 10，这些都标记为``related words``。所以最后我们就可以创建一个保存``related words``的一个``map``. 比如调用``related_words['car']``即可以调取出跟``car``意思上相近的TOP 10的单词。 
# 
# 那这个``related_words``又如何构建呢？ 在这里我们仍然使用``Glove``向量，然后计算一下俩俩的相似度（余弦相似度）。之后对于每一个词，存储跟它最相近的top 10单词，最终结果保存在``related_words``里面。 这个计算需要发生在离线，因为计算量很大，复杂度为``O(V*V)``， V是单词的总数。 
# 
# 这个计算过程的代码请放在``related.py``的文件里，然后结果保存在``related_words.txt``里。 我们在使用的时候直接从文件里读取就可以了，不用再重复计算。所以在此notebook里我们就直接读取已经计算好的结果。 作业提交时需要提交``related.py``和``related_words.txt``文件，这样在使用的时候就不再需要做这方面的计算了。

# In[14]:


# TODO 读取语义相关的单词
import codecs
def get_related_words(filename):
    #从预处理的相似词的文件加载相似词信息
    #文件格式w1 w2 w3..w11,其中w1为原词，w2-w11为w1的相似词
    related_words = {}
    with codecs.open(filename,'r','utf8') as Fin:
        lines = Fin.readlines()
    for line in lines:
        words = line.strip().split(' ')
        related_words[words[0]] = words[1:]
    return related_words

related_words = get_related_words('related_words.txt') # 直接放在文件夹的根目录下，不要修改此路径。


# #### 3.4 利用倒排表搜索
# 在这里，我们使用倒排表先获得一批候选问题，然后再通过余弦相似度做精准匹配，这样一来可以节省大量的时间。搜索过程分成两步：
# 
# - 使用倒排表把候选问题全部提取出来。首先，对输入的新问题做分词等必要的预处理工作，然后对于句子里的每一个单词，从``related_words``里提取出跟它意思相近的top 10单词， 然后根据这些top词从倒排表里提取相关的文档，把所有的文档返回。 这部分可以放在下面的函数当中，也可以放在外部。
# - 然后针对于这些文档做余弦相似度的计算，最后排序并选出最好的答案。
# 
# 可以适当定义自定义函数，使得减少重复性代码

# In[28]:


import queue as Q
def cosineSimilarity(vec1,vec2):
    #定义余弦相似度
    return np.dot(vec1,vec2.T)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2)))
def getCandidate(query):
    #根据查询句子中每个词所在的序号列表，求交集
    searched = set()
    for w in query.strip().split(' '):
        if w not in word2id or w not in inverted_idx:
            continue
        #搜索原词所在的序号列表
        if len(searched) == 0:
            searched = set(inverted_idx[w])
        else:
            searched = searched & set(inverted_idx[w])
        #搜索相似词所在的列表
        if w in related_words:
            for similar in related_words[w]:
                searched = searched & set(inverted_idx[similar])
    return searched

def get_top_results_tfidf(query):
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate （需要使用related_words). 
    2. 对于候选文档，计算跟输入问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    top = 5
    query_tfidf = computeSentenceEach(query,vectorizer,word2id)
    results = Q.PriorityQueue()
    searched = getCandidate(query)
    #print(len(searched))
    for candidate in searched:
        #计算candidate与query的余弦相似度
        result = cosineSimilarity(query_tfidf,X_tfidf[candidate])
        #优先级队列中保存相似度和对应的candidate序号
        #-1保证降序
        results.put((-1 * result,candidate))
    i = 0
    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表 
                   # hint: 利用priority queue来找出top results. 思考为什么可以这么做？ 
    while i < top and not results.empty():
        top_idxs.append(results.get()[1])
        i += 1
    return np.array(alist)[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


# In[29]:


def get_top_results_w2v(query):
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate （需要使用related_words). 
    2. 对于候选文档，计算跟输入问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    #embedding用glove
    top = 5
    query_emb = computeGloveSentenceEach(query,emb)
    results = Q.PriorityQueue()
    searched = getCandidate(query)
    for candidate in searched:
        result = cosineSimilarity(query_emb,X_w2v[candidate])
        results.put((-1 * result,candidate))
    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表 
                   # hint: 利用priority queue来找出top results. 思考为什么可以这么做？ 
    i = 0
    while i < top and not results.empty():
        top_idxs.append(results.get()[1])
        i += 1
    return np.array(alist)[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


# In[31]:


def get_top_results_bert(query):
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate （需要使用related_words). 
    2. 对于候选文档，计算跟输入问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    #embedding用Bert embedding
    top = 5
    query_emb = np.sum(bert_embedding([query],'sum')[0][1],axis = 0) / len(query.strip().split())
    results = Q.PriorityQueue()
    searched = getCandidate(query)
    for candidate in searched:
        result = cosineSimilarity(query_emb,X_bert[candidate])
        #print(result)
        results.put((-1 * result,candidate))
    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表 
                   # hint: 利用priority queue来找出top results. 思考为什么可以这么做？ 
    i = 0
    while i < top and not results.empty():
        top_idxs.append(results.get()[1])
        i += 1
    
    return np.array(alist)[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


# In[32]:


# TODO: 编写几个测试用例，并输出结果

test_query1 = "When did Beyonce start becoming popular"
#result:in the late 1990s
test_query2 = "What counted for more of the population change"
#result:births and deaths

print (get_top_results_tfidf(test_query1))
print (get_top_results_w2v(test_query1))
print (get_top_results_bert(test_query1))

print (get_top_results_tfidf(test_query2))
print (get_top_results_w2v(test_query2))
print (get_top_results_bert(test_query2))


# ### 4. 拼写纠错
# 其实用户在输入问题的时候，不能期待他一定会输入正确，有可能输入的单词的拼写错误的。这个时候我们需要后台及时捕获拼写错误，并进行纠正，然后再通过修正之后的结果再跟库里的问题做匹配。这里我们需要实现一个简单的拼写纠错的代码，然后自动去修复错误的单词。
# 
# 这里使用的拼写纠错方法是课程里讲过的方法，就是使用noisy channel model。 我们回想一下它的表示：
# 
# $c^* = \text{argmax}_{c\in candidates} ~~p(c|s) = \text{argmax}_{c\in candidates} ~~p(s|c)p(c)$
# 
# 这里的```candidates```指的是针对于错误的单词的候选集，这部分我们可以假定是通过edit_distance来获取的（比如生成跟当前的词距离为1/2的所有的valid 单词。 valid单词可以定义为存在词典里的单词。 ```c```代表的是正确的单词， ```s```代表的是用户错误拼写的单词。 所以我们的目的是要寻找出在``candidates``里让上述概率最大的正确写法``c``。 
# 
# $p(s|c)$，这个概率我们可以通过历史数据来获得，也就是对于一个正确的单词$c$, 有百分之多少人把它写成了错误的形式1，形式2...  这部分的数据可以从``spell_errors.txt``里面找得到。但在这个文件里，我们并没有标记这个概率，所以可以使用uniform probability来表示。这个也叫做channel probability。
# 
# $p(c)$，这一项代表的是语言模型，也就是假如我们把错误的$s$，改造成了$c$， 把它加入到当前的语句之后有多通顺？在本次项目里我们使用bigram来评估这个概率。 举个例子： 假如有两个候选 $c_1, c_2$， 然后我们希望分别计算出这个语言模型的概率。 由于我们使用的是``bigram``， 我们需要计算出两个概率，分别是当前词前面和后面词的``bigram``概率。 用一个例子来表示：
# 
# 给定： ``We are go to school tomorrow``， 对于这句话我们希望把中间的``go``替换成正确的形式，假如候选集里有个，分别是``going``, ``went``, 这时候我们分别对这俩计算如下的概率：
# $p(going|are)p(to|going)$和 $p(went|are)p(to|went)$， 然后把这个概率当做是$p(c)$的概率。 然后再跟``channel probability``结合给出最终的概率大小。
# 
# 那这里的$p(are|going)$这些bigram概率又如何计算呢？答案是训练一个语言模型！ 但训练一个语言模型需要一些文本数据，这个数据怎么找？ 在这次项目作业里我们会用到``nltk``自带的``reuters``的文本类数据来训练一个语言模型。当然，如果你有资源你也可以尝试其他更大的数据。最终目的就是计算出``bigram``概率。 

# #### 4.1 训练一个语言模型
# 在这里，我们使用``nltk``自带的``reuters``数据来训练一个语言模型。 使用``add-one smoothing``

# In[11]:


from nltk.corpus import reuters
import numpy as np
import codecs
# 读取语料库的数据
categories = reuters.categories()
corpus = reuters.sents(categories=categories)
#print(corpus[0])
# 循环所有的语料库并构建bigram probability. bigram[word1][word2]: 在word1出现的情况下下一个是word2的概率。
new_corpus = []
for sent in corpus:
    #句子前后加入<s>,</s>表示开始和结束
    new_corpus.append(['<s> '] + sent + [' </s>'])
print(new_corpus[0])
word2id = dict()
id2word = dict()
for sent in new_corpus:
    for w in sent:
        w = w.lower()
        if w in word2id:
            continue
        id2word[len(word2id)] = w
        word2id[w] = len(word2id)
vocab_size = len(word2id)
count_uni = np.zeros(vocab_size)
count_bi = np.zeros((vocab_size,vocab_size))
#writeVocab(word2id,"lm_vocab.txt")
for sent in new_corpus:
    for i,w in enumerate(sent):
        w = w.lower()
        count_uni[word2id[w]] += 1
        if i < len(sent) - 1:
            count_bi[word2id[w],word2id[sent[i + 1].lower()]] += 1
print("unigram done")
bigram = np.zeros((vocab_size,vocab_size))
#计算bigram LM，有bigram统计值的加一除以|vocab|+uni统计值，没有统计值,
#1 除以 |vocab|+uni统计值
for i in range(vocab_size):
    for j in range(vocab_size):
        if count_bi[i,j] == 0:
            bigram[i,j] = 1.0 / (vocab_size + count_uni[i])
        else:
            bigram[i,j] = (1.0 + count_bi[i,j]) / (vocab_size + count_uni[i])
def checkLM(word1,word2):
    if word1.lower() in word2id and word2.lower() in word2id:
        return bigram[word2id[word1.lower()],word2id[word2.lower()]]
    else:
        return 0.0
print(checkLM('I','like'))


# #### 4.2 构建Channel Probs
# 基于``spell_errors.txt``文件构建``channel probability``, 其中$channel[c][s]$表示正确的单词$c$被写错成$s$的概率。 

# In[12]:


# TODO 构建channel probability  
channel = {}
#读取文件，格式为w1:w2,w3..
#w1为正确词，w2,w3...为错误词
#没有给出不同w2-wn的概率，暂时按等概率处理
for line in open('spell-errors.txt'):
    # TODO
    (correct,error) = line.strip().split(':')
    errors = error.split(',')
    errorProb = dict()
    for e in errors:
        errorProb[e.strip()] = 1.0 / len(errors)
    channel[correct.strip()] = errorProb

# TODO

#print(channel)   


# #### 4.3 根据错别字生成所有候选集合
# 给定一个错误的单词，首先生成跟这个单词距离为1或者2的所有的候选集合。 这部分的代码我们在课程上也讲过，可以参考一下。 

# In[13]:


def filter(words):
    #将不在词表中的词过滤
    new_words = []
    for w in words:
        if w in word2id:
            new_words.append(w)
    return set(new_words)

def generate_candidates1(word):
    #生成DTW距离为1的词，
    #对于英语来说，插入，替换，删除26个字母
    chars = 'abcdefghijklmnopqrstuvwxyz'
    words = set([])
    #insert 1
    words = set(word[0:i] + chars[j] + word[i:] for i in range(len(word)) for j in range(len(chars)))
    #sub 1
    words = words | set(word[0:i] + chars[j] + word[i+1:] for i in range(len(word)) for j in range(len(chars)))
    #delete 1
    words = words | set(word[0:i] + word[i + 1:] for i in range(len(chars)))
    #交换相邻
    #print(set(word[0:i - 1] + word[i] + word[i - 1] + word[i + 1:] for i in range(1,len(word))))
    words = words | set(word[0:i - 1] + word[i] + word[i - 1] + word[i + 1:] for i in range(1,len(word)))
    #将不在词表中的词去掉
    words = filter(words)
    #去掉word本身
    if word in words:
        words.remove(word)
    return words

def generate_candidates(word):
    # 基于拼写错误的单词，生成跟它的编辑距离为1或者2的单词，并通过词典库的过滤。
    # 只留写法上正确的单词。 
    words = generate_candidates1(word)
    words2 = set([])
    for word in words:
        #将距离为1词，再分别计算距离为1的词，
        #作为距离为2的词候选
        words2 = generate_candidates1(word)
    #过滤掉不在词表中的词
    words2 = filter(words)
    #距离为1，2的词合并列表
    words = words  | words2
    return words
words = generate_candidates('strat')
print(words)


# #### 4.4 给定一个输入，如果有错误需要纠正
# 
# 给定一个输入``query``, 如果这里有些单词是拼错的，就需要把它纠正过来。这部分的实现可以简单一点： 对于``query``分词，然后把分词后的每一个单词在词库里面搜一下，假设搜不到的话可以认为是拼写错误的! 人如果拼写错误了再通过``channel``和``bigram``来计算最适合的候选。

# In[14]:


import numpy as np
import queue as Q
def word_corrector(word,context):
    word = word.lower()
    candidate = generate_candidates(word)
    if len(candidate) == 0:
        return word
    correctors = Q.PriorityQueue()
    for w in candidate:
        if w in channel and word in channel[w] and w in word2id and context[0].lower() in word2id and context[1].lower() in word2id:
            probility = np.log(channel[w][word] + 0.0001) +             np.log(bigram[word2id[context[0].lower()],word2id[w]]) +             np.log(bigram[word2id[context[1].lower()],word2id[w]])
            correctors.put((-1 * probility,w))
    if correctors.empty():
        return word
    return correctors.get()[1]
word = word_corrector('strat',('to','in'))
print(word)
def spell_corrector(line):
    # 1. 首先做分词，然后把``line``表示成``tokens``
    # 2. 循环每一token, 然后判断是否存在词库里。如果不存在就意味着是拼写错误的，需要修正。 
    #    修正的过程就使用上述提到的``noisy channel model``, 然后从而找出最好的修正之后的结果。 
    new_words = []
    words = ['<s>'] + line.strip().lower().split(' ') + ['</s>']
    for i,word in enumerate(words):
        if i == len(words) - 1:
            break
        word = word.lower()
        if word not in word2id:
            #认为错误，需要修正，句子前后加了<s>,</s>
            #不在词表中词,肯定位于[1,len - 2]之间
            new_words.append(word_corrector(word,(words[i - 1].lower(),words[i + 1].lower())))
        else:
            new_words.append(word)
    newline = ' '.join(new_words[1:])
    return newline   # 修正之后的结果，假如用户输入没有问题，那这时候``newline = line``
sentence = spell_corrector('When did Beyonce strat becoming popular')
print(sentence)


# #### 4.5 基于拼写纠错算法，实现用户输入自动矫正
# 首先有了用户的输入``query``， 然后做必要的处理把句子转换成tokens的形状，然后对于每一个token比较是否是valid, 如果不是的话就进行下面的修正过程。 

# In[16]:


test_query1 = "When did Beyonce strat becoming popular"  # 拼写错误的
#result:in the late 1990s
test_query2 = "What counted for more of the poplation change"  # 拼写错误的
#result:births and deaths
test_query1 = spell_corrector(test_query1)
test_query2 = spell_corrector(test_query2)
print(test_query1)
print(test_query2)
#print (get_top_results_tfidf(test_query1))
#print (get_top_results_w2v(test_query1))
#print (get_top_results_bert(test_query1))

#print (get_top_results_tfidf(test_query2))
#print (get_top_results_w2v(test_query2))
#print (get_top_results_bert(test_query2))


# ### 附录 
# 在本次项目中我们实现了一个简易的问答系统。基于这个项目，我们其实可以有很多方面的延伸。
# - 在这里，我们使用文本向量之间的余弦相似度作为了一个标准。但实际上，我们也可以基于基于包含关键词的情况来给一定的权重。比如一个单词跟related word有多相似，越相似就意味着相似度更高，权重也会更大。 
# - 另外 ，除了根据词向量去寻找``related words``也可以提前定义好同义词库，但这个需要大量的人力成本。 
# - 在这里，我们直接返回了问题的答案。 但在理想情况下，我们还是希望通过问题的种类来返回最合适的答案。 比如一个用户问：“明天北京的天气是多少？”， 那这个问题的答案其实是一个具体的温度（其实也叫做实体），所以需要在答案的基础上做进一步的抽取。这项技术其实是跟信息抽取相关的。 
# - 对于词向量，我们只是使用了``average pooling``， 除了average pooling，我们也还有其他的经典的方法直接去学出一个句子的向量。
# - 短文的相似度分析一直是业界和学术界一个具有挑战性的问题。在这里我们使用尽可能多的同义词来提升系统的性能。但除了这种简单的方法，可以尝试其他的方法比如WMD，或者适当结合parsing相关的知识点。 

# 好了，祝你好运！ 
