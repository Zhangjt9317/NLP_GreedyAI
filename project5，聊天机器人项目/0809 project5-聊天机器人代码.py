# coding=utf-8
import sys,os
import pandas as pd
import fool
import re
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import argparse
import jieba

# -----------------------------------------------------
# 加载停用词词典
stopwords = {}
with open(r'stopword.txt', 'r', encoding='utf-8') as fr:
    for word in fr:
        stopwords[word.strip()] = 0
# -----------------------------------------------------
# 加载同义词词典
simi = {}
with open(r'simi.txt', 'r', encoding='utf-8') as sr:
    for line in sr:
    	items = line.strip().split()
    	if len(items)>=2:
        	stopwords[items[0]] = items[1]

# 定义类
class CLF_MODEL:
    # 类目标：该类将所有模型训练、预测、数据预处理、意图识别的函数包括其中

    # 初始化模块
    def __init__(self):
        self.model = LogisticRegression()  # 成员变量，用于存储模型
        self.vectorizer = TfidfVectorizer()  # 成员变量，用于存储tfidf统计值

    def load_model(self, modelpath):
        with open(modelpath, 'rb') as f:
            self.vectorizer, self.model = pickle.load(f)

    def save_model(self, modelpath):
        with open(modelpath, 'wb') as f:
            pickle.dump((self.vectorizer, self.model), f)

    # 训练模块
    def train(self):
        # 函数目标：读取训练数据，训练意图分类模型，并将训练好的分类模型赋值给成员变量self.model
        # input：无
        # output：无

        # 从excel文件读取训练样本
        d_train = pd.read_excel("data_train.xlsx")
        # 对训练数据进行预处理
        d_train.sentence_train = d_train.sentence_fenci.apply(self.fun_clean)
        print("训练样本 = %d" % len(d_train))

        """
        TODO：利用sklearn中的函数进行训练，将句子转化为特征features
        """

        features = self.vectorizer.fit_transform(d_train.sentence_fenci.to_list())
        self.model.fit(features, d_train.label)


    # 预测模块（使用模型预测）
    def predict_model(self, sentence):
        # 函数目标：使用意图分类模型预测意图
        #  input：sentence（用户输入）
        # output：clf_result（意图类别），score（意图分数）

        # --------------
        # 对样本中没有的特殊情况做特别判断
        if sentence in ["好的", "需要", "是的", "要的", "好", "要", "是"]:
            return 1, 0.8
        # --------------

        """
        TODO：利用已训练好的意图分类模型进行意图识别
        """

        sent = self.fun_clean(' '.join(fool.cut(sentence)[0]))
        inputs = self.vectorizer.transform([sent])
        scores = self.model.predict_proba(inputs)[0]
        clf_result = np.argmax(scores, axis=0)
        score = scores[clf_result]

        return clf_result, score

    # 预测模块（使用规则）
    def predict_rule(self, sentence):
        # 函数目标：如果模型训练出现异常，可以使用规则进行预测，同时也可以让学员融合"模型"及"规则"的预测方式
        # input：sentence（用户输入）
        # output：clf_result（意图类别），score（意图分数）

        sentence = sentence.replace(' ', '')
        if re.findall(r'不需要|不要|停止|终止|退出|不买|不定|不订', sentence):
            return 2, 0.8
        elif re.findall(r'订|定|预定|买|购', sentence) or sentence in ["好的","需要","是的","要的","好","要","是"]:
            return 1, 0.8
        else:
            return 0, 0.8

    # 预处理函数
    def fun_clean(self, sentence):
        # 函数目标：预处理函数，将必要的实体转换成统一符号（利于分类准确），去除停用词等
        # input：sentence（用户输入语句）
        # output：sentence（预处理结果）

        """
        TODO：预处理函数，将必要的实体转换成统一符号（利于分类准确），去除停用词等

        """
        tokens = map(lambda x:simi.get(x,x), sentence.split())
        tokens = filter(lambda x:x not in stopwords, tokens)

        sentence = ' '.join(tokens)

        return sentence

    # 分类主函数
    def fun_clf(self, sentence):
        # 函数目标：意图识别主函数
        # input：sentence（ 用户输入语句）
        # output：clf_result（意图类别），score（意图分数）

        # 对用户输入进行预处理
        sentence = self.fun_clean(sentence)
        # 得到意图分类结果（0为“查询”类别，1为“订票”类别，2为“终止服务”类别）
        clf_result, score = self.predict_model(sentence)  # 使用训练的模型进行意图预测
        #clf_result, score = self.predict_rule(sentence)  # 使用规则进行意图预测（可与用模型进行意图识别的方法二选一）
        return clf_result, score


def fun_replace_num(sentence):
    # 函数目标：替换时间中的数字（目的是便于实体识别包fool对实体的识别）
    # input：sentence
    # output：sentence

    # 定义要替换的数字
    time_num = {"一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","十":"10","十一":"11","十二":"12"}
    for k, v in time_num.items():
        sentence = sentence.replace(k, v)
    return sentence


def slot_extract(sentence, key=None):
    # 函数目标：填槽函数（该函数从sentence中寻找需要的内容，完成填槽工作）
    # input：sentence（用户输入）, key（指定槽位，只对该句话提取指定槽位的信息）
    # output：slot（返回填槽的结果，以json格式返回，key为槽位名，value为值）

    slot = {}
    # 进行实体识别
    words, ners = fool.analysis(sentence)

    """
    TODO：从sentence中寻找需要的内容，完成填槽工作
    """

    for item in ners:
    	name, value = item[2], item[3]
    	if name=='location':
    		if 'from_city' in slot:
    			slot['to_city']=value
    		else:
    			slot['from_city']=value
    	else:
    		slot[name]=value

    return slot if not key else slot.get(key,{})


def fun_wait(clf_obj):
    # 函数目标：等待，获取用户输入问句
    # input：CLF_MODEL类实例化对象
    # output：clf_result（用户输入意图类别）, score（意图识别分数）, sentence（用户输入）

    # 等待用户输入
    print("\n\n\n")
    print("-------------------------------------------------------------")
    print("----*------*-----*-----*----*-----*-----*-----*-----*------")
    print("Starting ...")
    sentence = input("客服：请问需要什么服务？(时间请用12小时制表示）\n")
    # 对用户输入进行意图识别
    clf_result, score = clf_obj.fun_clf(sentence)
    return clf_result, score, sentence


def fun_search(clf_result, sentence):
    # 函数目标：为用户查询余票
    # input：clf_result（意图分类结果）, sentence（用户输入问句）
    # output：是否有票

    # 定义槽存储空间
    name = {"time":"出发时间", "date":"出发日期", "from_city":"出发城市", "to_city":"到达城市"}
    goal = {"time":"", "date":"", "from_city":"", "to_city":""}
    # 使用用户第一句话进行填槽
    sentence = fun_replace_num(sentence)
    slot_init = slot_extract(sentence)
    for key in slot_init.keys():
        goal[key] = slot_init[key]
    # 对未填充对槽位，向用户提问，进行针对性填槽
    while "" in goal.values():
        for key in goal.keys():
            if goal[key]=="":
                sentence = input("客服：请问%s是？\n"%(name[key]))   #  NLG
                sentence = fun_replace_num(sentence)
                slot_cur = slot_extract(sentence, key)
                for key in slot_cur.keys():
                    if goal[key]=="":
                        goal[key] = slot_cur[key]

    # 查询是否有票，并答复用户（本次查询是否有票使用随机数完成，实际情况可查询数据库返回）
    if random.random()>0.5:
        print("客服：%s%s从%s到%s的票充足"%(goal["date"], goal["time"], goal["from_city"], goal["to_city"]))
        # 返回1表示有票
        return 1
    else:
        print("客服：%s%s从%s到%s无票" % (goal["date"], goal["time"], goal["from_city"], goal["to_city"]))
        print("End !!!")
        print("----*------*-----*-----*----*-----*-----*-----*-----*------")
        print("-------------------------------------------------------------")
        # 返回0表示无票
        return 0


def fun_book():
    # 函数目标：执行下单订票动作
    # input：无
    # output：无

    print("客服：已为您完成订票。\n\n\n")
    print("End !!!")
    print("----*------*-----*-----*----*-----*-----*-----*-----*------")
    print("-------------------------------------------------------------")



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', default='./model.pkl')

    args = parser.parse_args()
    # 实例化对象
    clf_obj = CLF_MODEL()

    # 意图识别模型训练 Or 加载
    if not os.path.exists(args.modelpath):
        clf_obj.train()
        clf_obj.save_model(args.modelpath)
    else:
        clf_obj.load_model(args.modelpath)

    # 用户定义阈值（当分类器分类的分数大于阈值才采纳本次意图分类结果，目的是排除分数过低的意图分类结果）
    threshold = 0.55
    # 循环提供服务
    while 1:
        clf_result, score, sentence = fun_wait(clf_obj)
        # -------------------------------------------------------------------------------
        # 状态转移条件（等待-->等待）：用户输入未达到“查询”、“订票”类别的阈值 OR 意图被分类为“终止服务”
        # -------------------------------------------------------------------------------
        if score<threshold or clf_result==2:
            continue

        # -------------------------------------------------------------------------------
        # 状态转移条件（等待-->查询）：用户输入分类为“查询” OR “订票”
        # -------------------------------------------------------------------------------
        else:
            # 收集订票细节信息
            search_result = fun_search(clf_result, sentence)
            # 查询无票
            # -------------------------------------------------------------------------------
            # 状态转移条件（查询-->等待）：FUN_SEARCH执行完后用户输入意图为“终止服务” OR FUN_SEARCH返回无票
            # -------------------------------------------------------------------------------
            if search_result==0:
                continue
            # 查询有票
            else:
                # 等待用户输入
                sentence = input("客服：需要为您订票吗？\n")
                # 对用户输入进行意图识别
                clf_result, score = clf_obj.fun_clf(sentence)
                # -------------------------------------------------------------------------------
                # 状态转移条件（查询-->订票）：FUN_SEARCH返回有票 AND 用户输入意图为“订票”
                # -------------------------------------------------------------------------------
                if clf_result == 1:
                    fun_book()
                    continue

