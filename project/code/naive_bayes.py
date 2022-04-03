import numpy as np

class NaiveBayesClassifier(object):
    def __init__(self):
        self.label_prob = {}
        '''
        self.condition_prob represent the probability of each feature in a certain condition
        Such as the features are [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        Label: [1, 0, 1, 0, 1]
        Thus self.label_prob will be shown as below:     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}

    def fit(self, feature, label, laplace=1):
        '''
        train the model and save probabilities in self.label_prob and self.condition_prob represently
        :param feature: ndarray contains all features in training set
        :param label:ndarray contains all labels in training set
        :return: void
        '''

        #********* Begin *********#
        trainNum=len(label)
        featureNum=len(feature[0])

        labelDic={}
        for l in label:
            labelDic[l]=labelDic.get(l, 0)+1
        
        for lTemp,nTemp in labelDic.items():
            self.label_prob[lTemp]=(nTemp+1)/(trainNum+len(labelDic))
        
        labelFeatureDic={}
        for i in range(len(label)):
            if labelFeatureDic.get(label[i]):
                labelFeatureDic[label[i]].append(feature[i])
            else:
                labelFeatureDic[label[i]]=[feature[i]]

        #feature num for each colum
        featureTypeNum=[]
        featureTypeDic={}
        for i in range(featureNum):
            typeDic={}
            for fea in feature:
                typeDic[fea[i]]=1
            featureTypeNum.append(len(typeDic))
            featureTypeDic[i]=typeDic
        
        for lTemp,feasTemp in labelFeatureDic.items():
            lTempNum=len(feasTemp)
            helpDic1={}
            for i in range(featureNum):
                helpDic2={}
                for feas in feasTemp:
                    helpDic2[feas[i]]=helpDic2.get(feas[i], 0)+1
                # add feature types that haven't appeared
                for featType in featureTypeDic[i].keys():
                    if not helpDic2.get(featType):
                        helpDic2[featType]=0
                for feaTemp, nTemp in helpDic2.items():
                    helpDic2[feaTemp]=(nTemp+laplace)/(laplace * lTempNum+featureTypeNum[i])
                helpDic1[i]=helpDic2
            self.condition_prob[lTemp]=helpDic1

        #********* End *********#


    def predict(self, feature):
        '''
        predict then return the result
        :param feature: ndarray contains all features in test set
        :return:
        '''
        # ********* Begin *********#
        returnArr=[]
        for fea in feature:
            labelPossibility={}
            for label, labelPorb in self.label_prob.items():
                labelPossibility[label]=labelPorb
                for feaNth, helpDic2 in self.condition_prob[label].items():
                    if helpDic2.get(fea[feaNth]):
                        labelPossibility[label]=labelPossibility[label]*helpDic2[fea[feaNth]]
                    else:
                        labelPossibility[label]=0

            returnArr.append(max(labelPossibility, key=labelPossibility.get))
        
        return returnArr

feat = [[2, 1, 1],
        [1, 2, 2],
        [2, 2, 2],
        [2, 1, 2],
        [1, 2, 3]]

label = [1, 0, 1, 0, 1]

test = [[2, 1, 1],
        [1, 2, 2],
        [2, 2, 2],
        [2, 100, 2],
        [1, 2, 3]]

if __name__ == "__main__":

    p = NaiveBayesClassifier()
    p.fit(feat, label)
    res = p.predict(test)

    print(res)
