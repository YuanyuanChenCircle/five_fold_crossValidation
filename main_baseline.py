import json
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os


def splitDataSet(onefile, split_size,outdir):
    if not os.path.exists(outdir): #if not outdir,makrdir
        os.makedirs(outdir)
    # fr = open(fileName,'r')#open fileName to read
    # num_line = 0
    # onefile = fr.readlines()
    # print(onefile)

    # onefile = onefile[1:]

    num_line = len(onefile)  #输出多少条数据
    print(num_line)

    arr = np.arange(num_line) #get a seq and set len=numLine,产生一个随机数组
    np.random.shuffle(arr) #generate a random seq from arr

    list_all = arr.tolist() #把数组转换为list形式
    print(list_all)
    print("*****************")

    each_size = (num_line+1) // split_size #size of each split sets
    print(each_size)
    split_all = []; each_split = []
    count_num = 0; count_split = 0  #count_num 统计每次遍历的当前个数
                                    #count_split 统计切分次数
    for i in range(len(list_all)): #遍历整个数字序列
        # print("##########")
        # print(int(list_all[i]))
        # print("#######################")
        # print(onefile[int(list_all[i])].strip())
        each_split.append(onefile[int(list_all[i])])
        # each_split.append('\n')


        count_num += 1

        if count_num == each_size:
            count_split += 1 
            # array_ = np.array(each_split)
            print("##########")
            print(each_split)
            print(len(each_split))
            # print


            file = open(outdir + "6split_" + str(count_split) + '.txt','w',encoding='utf-8')


            for i in each_split:

                # file = open(outdir + "2split_" + str(count_split) + '.txt','w',encoding='utf-8')

                # print(i)

                json.dump(i,file,ensure_ascii=False)
                file.write('\n')



                # file.close()
            # json.dump(each_split,file,ensure_ascii=False)
            # np.savetxt(outdir + "split_" + str(count_split) + '.txt',\
            #             each_split,fmt="%s", delimiter=',')  #输出每一份数据
            file.close()
            split_all.append(each_split) #将每一份数据加入到一个list中
            each_split = []
            count_num = 0
    return split_all

def underSample(datafile): #只针对一个数据集的下采样
    # dataMat,labelMat = loadDataSet(datafile) #加载数据
    # pos_num = 0; pos_indexs = []; neg_indexs = []
    # for i in range(len(labelMat)):#统计正负样本的下标 ，#使样本均衡
    #         if labelMat[i] == 1:
    #         pos_num +=1
    #         pos_indexs.append(i)
    #         continue
    #     neg_indexs.append(i)
    # np.random.shuffle(neg_indexs)
    # neg_indexs = neg_indexs[0:pos_num]




    # f = open("input.txt", "r", encoding="utf8")
    # for line in f:
    #     x = json.loads(line)
    #     # print("******")
    #     # print(line)
    #     # print(x)
    #     sum_data.append(x)
    fr = open(datafile, 'r',encoding="utf8")

    onefile = []
    for line in fr:
        # print("*************")
        # print(line)
        # line = line.replace("'",'"')
        # print(line)
        # print("************")
        # print(line)
        # line = line.replace("'",'"')



        # line = line.replace("u", "")
        x = json.loads(line)
        # print(x)
        onefile.append(x)



    # onefile = json.loads(oneline_train)

    # onefile = fr.readlines()

    # outfile = []
    # for i in range(pos_num):
    #     pos_line = onefile[pos_indexs[i]]    
    #     outfile.append(pos_line)
    #     neg_line = onefile[neg_indexs[i]]      
    #     outfile.append(neg_line)
    return onefile #输出单个数据集采样结果

def generateDataset(datadir,outdir): #从切分的数据集中，对其中九份抽样汇成一个,\
    #剩余一个做为测试集,将最后的结果按照训练集和测试集输出到outdir中
    if not os.path.exists(outdir): #if not outdir,makrdir
        os.makedirs(outdir)
    listfile = os.listdir(datadir)#获取目录下的所有文件
    print(listfile)
    listfile1 = []

    for i in listfile:
        if i[0] != '6':
            continue
        else:
            listfile1.append(i)
    print(listfile1)

    train_all = []; test_all = [];cross_now = 0
    for eachfile1 in listfile1:
        train_sets = []; test_sets = []; 
        cross_now += 1 #记录当前的交叉次数
        for eachfile2 in listfile1:
            if eachfile2 != eachfile1:#对其余九份欠抽样构成训练集，使样本均衡
                one_sample = underSample(datadir + '/' + eachfile2)
                for i in range(len(one_sample)):
                    train_sets.append(one_sample[i])
        #将训练集和测试集文件单独保存起来
        with open(outdir +"/test_"+str(cross_now)+".datasets",'w') as fw_test:
            with open(datadir + '/' + eachfile1, 'r') as fr_testsets:


                for each_testline in fr_testsets:
                    print("********************")
                    # print(each_testline)

                    each_testline = json.loads(each_testline)


                    test_sets.append(each_testline)
                    print(test_sets) 
            # for oneline_test in test_sets:
            #     fw_test.write(oneline_test) #输出测试集
            test_all.append(test_sets)#保存训练集
        with open(outdir+"/train_"+str(cross_now)+".datasets",'w') as fw_train:
            for oneline_train in train_sets: 

                # oneline_train = json.loads(oneline_train)


                oneline_train = oneline_train
                # fw_train.write(oneline_train)#输出训练集
            train_all.append(train_sets)#保存训练集
    return train_all,test_all


def mean_fun(onelist):
    count = 0
    for i in onelist:
        count += i
    return float(count/len(onelist))

#最终计算结果
def get_score(input_path, ground_truth_path, output_path):
    cnt = 0
    f1 = open(ground_truth_path,"r")
    f2 = open(output_path,"r")
    correct = 0
    for line in f1:
        cnt += 1
        try:
            if line[:-1] == f2.readline()[:-1]:
                correct += 1
        except:
            pass

    return 1.0*correct/cnt

#分词
def trans(x):
    x = list(jieba.cut(x))
    return " ".join(x)

def classf(train_data,test_data):


    se = []
    for line in train_data:
        # x = json.loads(line)
        se.append(line["A"])
        se.append(line["B"])
        se.append(line["C"])


    for a in range(0, len(se)):
        se[a] = trans(se[a])

    tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(se)
    sparse_result = tfidf_model.transform(se)
    os.chdir('/Users/cy/Downloads/6.2_1 2 2/')

    ouf = open('./2.txt', "w", encoding="utf8")



    for line in test_data:
        # x = json.loads(line)
        y = [
            trans(line["A"]),
            trans(line["B"]),
            trans(line["C"])
        ]

        y = tfidf_model.transform(y)
        y = y.todense()

        v1 = np.sum(np.dot(y[0], np.transpose(y[1])))
        v2 = np.sum(np.dot(y[0], np.transpose(y[2])))
        if v1 > v2:
            print("B", file=ouf)
        else:
            print("C", file=ouf)
    print(os.getcwd())
    # os.chdir('/Users/cy/Downloads/6.2_1 2 2/')
    ouf.close()


    de = get_score('./input.txt','./output_truth.txt','./2.txt')
    # ouf.close()
    return de





#获得分析后的训练语料
se = set()
f = open("input.txt", "r", encoding="utf8")
for line in f:
    x = json.loads(line)
    se.add(x["A"])
    se.add(x["B"])
    se.add(x["C"])

# f = open("/input/input.txt", "r", encoding="utf8")
# for line in f:
#     x = json.loads(line)
#     se.add(x["A"])
#     se.add(x["B"])
#     se.add(x["C"])

# data = list(se)
# for a in range(0, len(data)):
#     data[a] = trans(data[a])


sum_data = []
f = open("input.txt", "r", encoding="utf8")
for line in f:
    x = json.loads(line)
    # print(x)
    sum_data.append(x)


# print(sum_data)
# for 
# print(data)
print(len(sum_data))
# print(len(data))
datadir = './split_data/'
outdir = "./sample_data4/" #抽样的数据集存放目录

splitDataSet(sum_data,5,datadir)

train_all,test_all = generateDataset(datadir,outdir)

print(len(train_all))









print(os.getcwd())
# os.makedirs('./t')
curdir = '/Users/cy/Downloads/6.2_1 2 2/t1'
os.chdir(curdir)

#构造出纯数据型样本集
cur_path = curdir


ACCs = [];SNs = []; SPs =[]
for i in range(len(train_all)):
    os.chdir(cur_path)
    train_data = train_all[i];train_X = [];train_y = []
    test_data = test_all[i];test_X = [];test_y = []
    for eachline_train in train_data:
        # one_train = eachline_train.split('\t') 
        # one_train_format = []
        # for index in range(3,len(one_train)-1):
        #     one_train_format.append(float(one_train[index]))
        train_X.append(eachline_train)

        # train_y.append(int(one_train[-1].strip()))
    for eachline_test in test_data:
        # one_test = eachline_test.split('\t')
        # one_test_format = []
        # for index in range(3,len(one_test)-1):
        #     one_test_format.append(float(one_test[index]))
        test_X.append(eachline_test)
        # test_y.append(int(one_test[-1].strip()))
    #======================================================================
    #classifier start here
    # if not os.path.exists(clfname):#使用的分类器
    #     os.mkdir(clfname)
    # out_path = clfname + "/" + clfname + "_00" + str(i)#计算结果文件夹
    # if not os.path.exists(out_path):
    #     os.mkdir(out_path)
    # os.chdir(out_path)
    # print(train_X)


    ACC = classf(train_X,test_X)









    # ACC= classifier(clf, train_X, train_y, test_X, test_y)
    ACCs.append(ACC)
    # ;SNs.append(SN);SPs.append(SP)
    #======================================================================
ACC_mean = mean_fun(ACCs)
print(ACC_mean)







# train_X,test_X = train_test_split(sum_data,test_size = 0.2)


# #先划分比例，然后进行分词
# se = []
# for line in train_X:
#     # x = json.loads(line)
#     se.append(line["A"])
#     se.append(line["B"])
#     se.append(line["C"])


# for a in range(0, len(se)):
#     se[a] = trans(se[a])



#再写划分数据集

# train_X,test_X = train_test_split(data,test_size = 0.2)

# tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(se)
# sparse_result = tfidf_model.transform(se)



# # f = open("/input/input.txt", "r", encoding="utf8")
# # ouf = open("/output/output.txt", "w", encoding="utf8")
# ouf = open("./output.txt", "w", encoding="utf8")

# for line in test_X:
#     # x = json.loads(line)
#     y = [
#         trans(line["A"]),
#         trans(line["B"]),
#         trans(line["C"])
#     ]

#     y = tfidf_model.transform(y)
#     y = y.todense()

#     v1 = np.sum(np.dot(y[0], np.transpose(y[1])))
#     v2 = np.sum(np.dot(y[0], np.transpose(y[2])))
#     if v1 > v2:
#         print("B", file=ouf)
#     else:
#         print("C", file=ouf)

# ouf.close()