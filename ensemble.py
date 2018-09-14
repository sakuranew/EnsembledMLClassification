from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict,StratifiedKFold
import pandas as pd
class Ensemble(object):
    def __init__(self, estimators):
        self.estimators_names=[]
        self.estimators=[]
        self.result={}
        self.prob={}
        self.accuracy={}
        self.votedResult=[]
        self.votedAccuracy=0
        self.datasize=0
        
        for item in estimators:
            self.estimators.append(item[1])
            self.estimators_names.append(item[0])
            pass
        pass
    def fit(self, x,y):
       
        for i in self.estimators:
            i.fit(x,y)
            pass
    def predict(self, x,y=None):

        for name,fun in zip(self.estimators_names,self.estimators):
            if(name is not "GaussianProcess"):
                self.result[name]=fun.predict(x)

        # for name,fun in zip(self.estimators_names,self.estimators):
        #     self.result[name]=fun.predict(x)
        #     if y.any():
        #         self.accuracy[name]=accuracy_score(y,self.result[name])
        #         print("{} accuracy is {}".format(name, self.accuracy[name]))
        #         if self.accuracy[name]>0.7:
        #             target_names = ['dark', 'good', 'light']
        #             print(classification_report(y,self.result[name], target_names=target_names))
           
    def predict_prob(self, x,y,index):
        for name,fun in zip(self.estimators_names,self.estimators):
            if(name is not "GaussianProcess"):
                self.result[name][index,0]=fun.predict(x)
            else:
                self.prob["GaussianProcess"][index]=fun.predict_proba(x)
            # plt.figure(figsize=(200,200))
            # axis=np.arange(0,self.datasize)
            # axis=axis/10
            # for j in range(1,4):                
            #     plt.subplot(3,1,j)
            #     for i in range(self.datasize):
            #         plt.scatter(axis[i],self.prob[name][i,j-1],c='r' if y[i]==0 else( 'y' if y[i]==2 else 'b'),s=10 if y[i]==self.result[name][i] else 40)
            # plt.show()

    def select(self,y):
        # for index,name in enumerate(self.estimators_names):
        for index in range(1):

            prob=self.prob["GaussianProcess"]
            # error=np.zeros((self.datasize,3))
            # good=np.zeros((self.datasize,3))

            plt.figure(index,figsize=(200,200))
            axis=np.arange(0,self.datasize)
            axis=axis
            for j in range(1,4):                
                plt.subplot(3,1,j)
                for i in range(self.datasize):
                    flag=y[i]/2==np.argmax(prob[i])
                    plt.scatter(axis[i],prob[i,j-1],c='r' if y[i]==0 else( 'y' if y[i]==2 else 'b'),s=10 if flag else 40)
            #         # if not flag:
                    #     error[i,j-1]=prob[i,j-1]
                    # if y[i]==2:
                    #     good[i,j-1]=prob[i,j-1]

                        # plt.annotate(str(prob[i,j-1]), xy = (axis[i],prob[i,j-1]), xytext = (axis[i]+0.1, prob[i,j-1]+0.1))
            # plt.savefig(name+".png")
            plt.show()
            # np.save(name+"-error.npy", error)
            # np.save(name+"-good.npy", good)

            for i,v in enumerate(prob):
                
                maxi=np.argmax(v)
                # self.result[name][i]=maxi*2

                if(maxi==1):
                
                    if v[maxi]>=0.60:
                        self.result["GaussianProcess"][i]=maxi*2
                    else:
                        
                        # temp=[]
                        # for name in self.estimators_names:
                        #     if(name is not "GaussianProcess"):
                        #         temp.append(self.result[name][i])
                        # count=np.bincount(np.array(temp).astype(np.int64)[:,0])
                        # if np.argmax(count)==2:
                        #     self.result["GaussianProcess"][i]=2
                        # else:
                        #     self.result["GaussianProcess"][i]=6
                        self.result["GaussianProcess"][i]=6

                        # if(min(v[0],v[2])<0.02):
                        #     self.result[name][i]=6
                        # else:
                        #     self.result[name][i]=2
                #     if 0.9>v[maxi]>=0.75 and (min(v[0],v[2])>0.02):
                #         self.result[name][i]=maxi*2
                #     elif v[maxi]>=0.9:
                #         self.result[name][i]=maxi*2
                #     else:
                #         self.result[name][i]=6
 
                else:
                    self.result["GaussianProcess"][i]=maxi*2

    def crossValidation(self, x,y,n=7):
        self.datasize=x.shape[0]
        for name in self.estimators_names:
            self.prob[name]=np.empty((self.datasize,3))
            self.result[name]=np.empty((self.datasize,1))

        cv = StratifiedKFold(n_splits=n)
        
        for train, test in cv.split(x, y):
            self.fit(x[train], y[train])
            # self.predict(X,y=y)
            # alla=alla+self.predict(x[test], y[test])
            self.predict_prob(x[test], y[test],test)
        
        self.select(y)
        for name in self.estimators_names:

            target_names = ['dark', 'good', 'light']
            print(name)
            print(classification_report(y,self.result[name], target_names=target_names))
        self.vote(y)
        self.report(y)
        
        # self.datasize=x.shape[0]
        # for name,fun in zip(self.estimators_names,self.estimators):
        #     self.result[name]=cross_val_predict(fun,x,y,cv=7)
        
        #     self.accuracy[name]=accuracy_score(y,self.result[name])
        #     print("{} accuracy is {}".format(name, self.accuracy[name]))
        #     if self.accuracy[name]>0.7:
        #         target_names = ['shallow', 'good', 'deep']
        #         print(classification_report(y,self.result[name], target_names=target_names))
        # pass
    def vote(self,y=None, weight=None):
        temp=np.zeros((self.datasize,len(self.estimators_names)))
        i=0
        for _,value in self.result.items():
            value=np.reshape(value,(value.shape[0],1))
            # print(value[:][0])    [:,0]才可以得到一列
            temp[:,i]=value[:,0]
            i=i+1
        self.estimators_names.append('voted')
        self.result['voted']=np.empty((self.datasize,1))

        for i in range(temp.shape[0]):
            count=np.bincount(temp[i].astype('int'),weights=weight)
            self.result['voted'][i]=np.argmax(count*2)
        # print(self.votedResult)
    
        if y.any():
            # self.votedAccuracy=accuracy_score(y,self.votedResult)
            # print("voted accuracy is {}".format( self.votedAccuracy))
            target_names = ['dark', 'good', 'light']
            print(classification_report(y,self.result['voted'], target_names=target_names))
          
    def report(self,y_true):
        writer = pd.ExcelWriter('report6.xlsx')
        for i,funname in enumerate(self.estimators_names):
            y_pre=np.array(self.result[funname]).astype(np.int64)
            temp=np.empty(self.datasize)
            temp.fill(0)
            index=temp==y_true
            res=y_pre[index]
            res=np.bincount(res[:,0])
            dark2dark=res[0]
            dark2good=  res[2] if (res.shape[0]>2) else 0
            dark2shallow=res[4] if (res.shape[0]>4) else 0
            dark2no=res[6] if (res.shape[0]>6) else 0

            temp.fill(2)
            index=temp==y_true
            res=y_pre[index]
            res=np.bincount(res[:,0])
            
            good2dark=res[0]
            good2good=res[2] if (res.shape[0]>2) else 0
            good2shallow=res[4] if (res.shape[0]>4) else 0
            good2no=res[6] if (res.shape[0]>6) else 0

            temp.fill(4)
            index=temp==y_true
            res=y_pre[index]
            res=np.bincount(res[:,0])
            
            shallow2dark=res[0]
            shallow2good=res[2] if (res.shape[0]>2) else 0
            shallow2shallow=res[4] if (res.shape[0]>4) else 0
            shallow2no=res[6] if (res.shape[0]>6) else 0

            total_good_pre=good2shallow+good2good+good2dark+good2no
            total__dark_pre=dark2dark+dark2good+dark2shallow+dark2no
            total__shallow_pre=shallow2dark+shallow2good+shallow2shallow+shallow2no

            dark_recall=dark2dark/total__dark_pre*100
            good_recall=good2good/total_good_pre*100
            shallow_recall=shallow2shallow/total__shallow_pre*100

            dark_accuracy=dark2dark/(dark2dark+shallow2dark+good2dark)*100
            good_accuracy=good2good/(dark2good+shallow2good+good2good)*100
            shallow_accuracy=shallow2shallow/(dark2shallow+shallow2shallow+good2shallow)*100

            dark_f=dark2good/total__dark_pre*100
            good_f=(good2dark+good2shallow+good2no)/total_good_pre*100
            shallow_f=shallow2good/total__shallow_pre*100

            dark_no=dark2no/total__dark_pre*100
            good_no=good2no/total_good_pre*100
            shallow_no=shallow2no/total__shallow_pre*100

            r=np.array([[dark2dark,dark2good,dark2shallow,dark2no,total__dark_pre,dark_recall,dark_accuracy,dark_f,dark_no],
                        [good2dark,good2good,good2shallow,good2no,total_good_pre,good_recall,good_accuracy,good_f,good_no],
                        [shallow2dark,shallow2good,shallow2shallow,shallow2no,total__shallow_pre,shallow_recall,shallow_accuracy,shallow_f,shallow_no]])
            r=np.around(r,decimals=2)
            df = pd.DataFrame(r,index=['dark_r','good_r','light_r'], columns=['dark_p','good_p','light_p','discard',"total",'recall',"accuracy",'fault',"dis-rate"])
            df["recall"] =df["recall"].apply(lambda x :str(x)+'%')
            df["accuracy"] =df["accuracy"].apply(lambda x :str(x)+'%')
            df["dis-rate"] =df["dis-rate"].apply(lambda x :str(x)+'%')

            df["fault"] =df["fault"].apply(lambda x :str(x)+'%')

            # df["fault"] =[ ' %s\%' % i for i in df["fault"]]
# data[u'buy_place'] = data[u'buy_place'].apply(lambda x :x.split(' ')[-1])
            df.to_excel(writer,funname)
        writer.save()
    # def round_operation(self, v,d=2):
    #     return round(v,d)
        