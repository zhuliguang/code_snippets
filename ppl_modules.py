"""
@Author: Leon Zhu
@Created: 18/Jul/2017
"""
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from reason_to_group_mapping import reason_to_group_mapping

class ColumnsSelector(BaseEstimator, TransformerMixin):
    """
    Select subset of data at provided keys.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key, text=False):
        self.key = key
        self.text = text
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(self.key, basestring):
            #if self.text:
            return X[self.key]
            #return X.loc[:,self.key].values.reshape(-1,1)
        if isinstance(self.key, list):
            return X[self.key]
        return None

class Recordify(BaseEstimator, TransformerMixin):
    """
    Transfer a categorical feature into dict format
    Mainly for feature union
    """
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X.to_dict('records')

class LabelMapping(BaseEstimator, TransformerMixin):
    """
    This class takes reason group mapping from reason_group_mapping.py
    Retures a mapping resuilt as a pandas series
    Dumps a fited labelEncoder(sklearn)
    """

    def __init__(self, coding='encoder'):
        self.coding = coding
        self.y_all = list(set(reason_to_group_mapping.values()))
        self.y_all.append('Other')
        self.dump_pkl(self.y_all)   

    def fit(self, y):
        return self

    def transform(self,y):
        tmp = []
        for label in y:
            try:
                reason = reason_to_group_mapping[label]
            except KeyError:
                reason = 'Other'
            tmp.append(reason)
        return pd.Series(tmp)

    def dump_pkl(self, labels, pkl_path='model/labelEncoder.pkl'):
        import cPickle as pickle
        labelset = labels
        if self.coding == 'encoder':
            lb = LabelEncoder()
        elif self.coding == 'binarizer':
            lb = LabelBinarizer()
        else:
            print "could not encode lables"
            exit()
        lb.fit(labelset)
        with open(pkl_path,'wb') as handler:
            pickle.dump(lb, handler)

class Sampling:
    """
    An under and over sampler. Not actually used in the project

    """
    def __init__(self, y, cutoff=5):
        """
        :param: y is a pandas series type
        :param: cutoff is the n- top frequent class label
        """
        from collections import Counter
        value_counts = dict(y.value_counts())
        self.sampling_target = Counter(value_counts).most_common(cutoff)
        self.lower_limit = self.sampling_target[-1][1]

    def sampling(self,X,y):
        """
        :param X pandas dataframe
        :param y pandas series
        """
        assert len(X) == len(y)
        from random import shuffle
        target = y
        mask = [True for i in range(len(y))]
        for label, amount in self.sampling_target:
            idx = list(y[y == label].keys())
            shuffle(idx)
            for i in idx[self.lower_limit:]:
                mask[i] = False

        chosen = [ idx for idx,item in enumerate(mask) if item ]
        return X.iloc[chosen,:], y.iloc[chosen]

class ClassFilter:
    """
    Filters certain class labels based on their frequencies.value_counts
    """
    def __init__(self,y,cutoff=4):
        from collections import Counter
        value_counts = dict(y.value_counts())
        self.ignore_target = Counter(value_counts).most_common(cutoff)
    def filter(self, X,y):
        assert len(X) == len(y)
        mask = [True for i in range(len(y))]
        for label, _ in self.ignore_target:
            indices = list(np.where(y == label)[0])
            for ind in indices:
                mask[ind] = False
        chosen = [ idx for idx, item in enumerate(mask) if item ]
        return X.iloc[chosen,:], y.iloc[chosen] 


class Preproc_intent_prediction(BaseEstimator, TransformerMixin):
"""
Preprocessing class for intent prediction.
"""
    def __init__(self, last_number_of_events=5):
        self.lne = last_number_of_events

    def fit(self, X, y=None):
        return self

    def add_spaces_to_string(self,vq_str):
        """
        @param vq_str_str is a string input (orginially from QI360 virtual queue column)
        @return a string of sentence that uses space as word delimiter.

        This function adds spaces to a sentance if the sentance uses capital letters as word delimiter (I know, annoying)
        Mostly for QI360 Virtual queue
        
        e.g.
        "helloWorld" -> "hello World"
        "HelloWorld" -> "Hello World"
        "HelloABCWorld" -> "Hello ABC World"
        "HelloWorldABC" -> "Hello World ABC"
        "ABCHelloWorld" -> "ABC Hello World"
        "ABC HELLO WORLD" -> "ABC HELLO WORLD"
        "ABCHELLOWORLD" -> "ABCHELLOWORLD"
        "A" -> "A"

        """
        if vq_str == '':
            return ''
        res = vq_str[0]
        for idx in xrange(1, len(vq_str)):
            if vq_str[idx].isupper():
                prev = vq_str[idx-1]
                if prev.isupper():
                    if idx < (len(vq_str)-1):
                        nxt = vq_str[idx+1]
                        if ((not nxt.isupper()) and (nxt != ' ')):
                            res += ' '
                elif prev.islower():
                    res += ' '
            res += vq_str[idx]
        return res.lower().strip()

    def event_tokeniser(self,event):
        """
        @param event, a event string from query
               should have a format like serviceTag_str_str_str...
        @return a string seperated by ' ' as a bag of words
                    or empty string if it does not have any actual content
        """
        tokens = event.split("_")
        try:
            tokens[1]
            bows = "".join([token+" " for token in tokens[1:] if token != ""])
            return bows.strip()
        except IndexError:
            return ""


    def transform(self, X, *_):
        from datetime import datetime

        """
        @paran X a string of events (and siebel customer id)
        @return pandas data frame format with proper column name
            labelled: a series of events ending with IVR, used as training and validation
            unlabelled: a series of events ending without IVR event, used for prediction only

        Process each line of text coming out of greenplum query
        e.g.
        "customerid eventtype1 eventtype2 eventtype3 ivr_XXXXX"

        it process each 'word' seperated by space according to its "service_tag"
        e.g
        callview starts with callv_
        QI360 starts with qi360_
        IVR starts with ivr_ This is the class label
        etc.
        

        """
        import re
        labeled = []
        unlabeled = []
        for row in X:
            event_info = []

            c_info = {
            'cid':'',
            'gender':'',
            'age':'',
            'type':'',
            'royalty':'',
            'haswww':'',
            'srv_out_last3':0,
            'offer_made':0
            }

            final_bill = {
                        'method':'',
                        'nrc_description':'',
                        'nrc_charges':0,
                        'usage_description':'',
                        'usage_charges':0,
                        'service_charges':0,
                        'nrc_from_last':0,
                        'usage_from_last':0,
                        'service_from_last':0,
                        'avg_nrc_charges':0,
                        'avg_usage_charges':0,
                        'avg_service_charges':0,
                        'past_num_bills':0,
                        'bill_prep':'',
                        'days_since':0
                    }

            bill_history = []
            events = []

            for item in row:
                item = item.strip().strip('_')

                if re.match('[0-9]+',item):
                    c_info['cid'] = item
                if item.startswith('title'):
                    if item[5:] == 'MR':
                        c_info['gender'] = 'male'
                    elif item[5:] == 'MRS' or item[5:] == 'MS' or item[5:] == 'MISS':
                        c_info['gender'] = 'female'
                    else:
                        c_info['gender'] = 'other'
                if item.startswith('age'):
                    c_info['age'] = item[3:]
                if item.startswith('type'):
                    c_info['type'] = item[4:]
                if item.startswith('royalty'):
                    c_info['royalty'] = item[7:]
                if item.startswith('haswww'):
                    try:
                       c_info['haswww'] = item.split("_")[1]
                    except IndexError:
                       c_info['haswww'] = 'n'

                if item.startswith('qi360'):
                    cols = item.split("_")
                    try:
                        events.append(self.add_spaces_to_string(cols[2]))
                    except IndexError:
                        continue
                if item.startswith('athen_'):
                    tokens = item.split("_")
                    try:
                        tokens[1]
                        c_info['offer_made'] = 1
                    except IndexError:
                        continue

                if item.startswith('blkhwk'):
                    tokens = item.split("_")
                    try:
                        if tokens[2] == 'current':
                            c_info['srv_out_last3'] = 1
                    except IndexError:
                        continue

                if item.startswith('callv'):
                    events.append(self.event_tokeniser(item))

                if item.startswith('sblord_'):
                    events.append(self.event_tokeniser(item))

                if item.startswith('srvy_'):
                    events.append(self.event_tokeniser(item))

                if item.startswith('web_'):
                    events.append(self.event_tokeniser(item))

                if item.startswith('kenanBS_'):
                    # reinitialise bill_info so that only the last bill is considered to be related to the ivr event
                    # e.g. kenanBS1 event1 event 2 kenanBS2 event3 kenanBS3 event4 ivr_EnquireBill
                    # only kenanBS3 is being considered under the assumption that this bill is the reason that the customer calls
                    bill_info = {
                        'method':'',
                        'nrc_description':'',
                        'nrc_charges':0,
                        'usage_description':'',
                        'usage_charges':0,
                        'service_charges':0,
                        'nrc_from_last':0,
                        'usage_from_last':0,
                        'service_from_last':0,
                        'avg_nrc_charges':0,
                        'avg_usage_charges':0,
                        'avg_service_charges':0,
                        'past_num_bills':0,
                        'bill_prep':'',
                        'days_since':0
                    }
                    #  2000000042620,             tokens[0]   siebel id
                    #  A4 Paper,                  tokens[1]   delivery method
                    #  ,                          tokens[2]   nrc description
                    #  0.0,                       tokens[3]   nrc charges
                    #  International Roaming SMS|International Roaming Data|SMS|National Direct Dialled Calls|SMS|MMS|National Direct Dialled Calls,      
                    #                             tokens[4]   usage description
                    #  32.021,                    tokens[5]   usage charges
                    #  47.021,                    tokens[6]   service charges
                    #  3,                         tokens[7]   number of past bills maximum 3
                    #  -0.0033333333333333335,    tokens[8]   past average nrc charges
                    #  29.000000000000004,        tokens[9]   past average usage charges
                    #  44.0,                      tokens[10]  past average service charges
                    #  2017-05-18 05:38:47        tokens[11]  timestamp
                    dm = {'Email-PDF':0,
                            'A4-Paper':1,
                            'Online-Bill':2,
                            'Paper-(A4)-+-Email-PDF':3}
                    tokens = item.split("_")
                    try:
                        bill_info['method'] = dm.get(tokens[1],4)
                    except TypeError:
                        pass
                    try:
                        bill_info['nrc_description'] = tokens[2].replace("|"," ")
                    except TypeError:
                        pass
                    try:
                        bill_info['nrc_charges'] = float(tokens[3])
                    except ValueError:
                        bill_info['nrc_charges'] = 0
                    try:
                        bill_info['usage_description'] = tokens[4].replace("|"," ")
                    except TypeError:
                        pass
                    try:    
                        bill_info['usage_charges'] = float(tokens[5])
                    except ValueError:
                        bill_info['usage_charges'] = 0
                    try:    
                        bill_info['service_charges'] = float(tokens[6])
                    except ValueError:
                        bill_info['service_charges'] = 0
                    try:
                        bill_info['past_num_bills']  = int(tokens[7])
                    except ValueError:
                        bill_info['past_num_bills'] = 0
                    try:
                        bill_info['avg_nrc_charges']  = float(tokens[8])
                    except ValueError:
                        bill_info['avg_nrc_charges'] = 0
                    try:
                        bill_info['avg_usage_charges']  = float(tokens[9])
                    except ValueError:
                        bill_info['avg_usage_charges'] = 0
                    try:
                        bill_info['avg_service_charges']  = float(tokens[10])
                    except ValueError:
                        bill_info['avg_service_charges'] = 0
                    try:
                        bill_info['bill_prep']  = datetime.strptime(tokens[11], '%Y-%m-%d-%H:%M:%S').strftime('%Y-%m-%d')
                    except TypeError:
                        bill_info['bill_prep'] = ''
                    try:
                        last_bill = bill_history[-1]
                        bill_info['nrc_from_last'] = bill_info['nrc_charges'] - last_bill['nrc_charges']
                        bill_info['usage_from_last'] = bill_info['usage_charges'] - last_bill['usage_charges']
                        bill_info['service_from_last'] = bill_info['service_charges'] - last_bill['service_charges']
                    except IndexError:
                        bill_info['nrc_from_last'] = 0
                        bill_info['usage_from_last'] = 0
                        bill_info['service_from_last'] = 0

                    bill_info['days_since'] = 0
                    bill_history.append(bill_info)

                # if item.startswith('cares_'):
                #     tokens = item.split("_")
                #     try:
                #         tokens[1]
                #         events += "".join([self.add_spaces_to_string(token)+" " for token in tokens[1:] if token != ""])
                #         n_events += 1
                #     except IndexError:
                #         continue

                # token ivr indicates the end of an event 

                if item.startswith('ivr'):
                    tokens = item.split("_")
                    ivr = tokens[1]
                    timestamp = datetime.strptime(tokens[2], '%Y-%m-%d')
                    if ivr != '':
                        for k,v in c_info.iteritems():
                            event_info.append(v)
                        event_str = ' '.join(events[-self.lne:])
                        event_info.append(event_str.strip())

                        #billing
                        try:
                            final_bill = bill_history[-1]
                        except IndexError:
                            pass
                    
                        for k,v in final_bill.iteritems():
                            event_info.append(v)
                        #timestamp
                        event_info.append(timestamp)
                        event_info.append(ivr)
                        labeled.append(event_info)

                    event_info = []
                    c_info['srv_out_last3'] = 0
                    c_info['offer_made'] = 0
                    events = []

            # if predict mode add last batch of events as the features for prediction
            for k,v in c_info.iteritems():
                event_info.append(v)
            event_str = ' '.join(events[-self.lne:])
            event_info.append(event_str.strip())

            #billing
            try:
                final_bill = bill_history[-1]
            except IndexError:
                pass
        
            for k,v in final_bill.iteritems():
                event_info.append(v)

            #timestamp
            event_info.append(datetime.today())
            unlabeled.append(event_info)
            
        labeled = pd.DataFrame(labeled)
        try:
            labeled.columns = c_info.keys() + ['events'] + final_bill.keys() + ['timestamp','label']
        except ValueError:
            # There are not any labelled data, continue and return None
            labeled = None
            
        unlabeled = pd.DataFrame(unlabeled)
        unlabeled.columns = c_info.keys() + ['events'] + final_bill.keys() + ['timestamp']
       
        # labeled['days_since'] = (labeled['timestamp'] - labeled['bill_prep']).astype('timedelta64[D]')
        # unlabeled['days_since'] = (unlabeled['timestamp'] - unlabeled['bill_prep']).astype('timedelta64[D]')
        return labeled, unlabeled



if __name__ == "__main__":
    ##
    # The following is self testing
    ##
    import gzip
    from sklearn.pipeline import Pipeline,make_pipeline,FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import train_test_split

    import cPickle as pickle
    pre = Preproc()
    f = open("sn_reason_group/training.txt",'rb')

    rawdata = [ line.strip().split(" ") for line in f.readlines()]
    print len(rawdata)

    res,_ = pre.transform(rawdata)


    y = res['label']
    X = res[['gender','age','type','royalty','haswww','n_events','events']]

    y = LabelMapping().transform(res['label'])
    try:
        with open('labelBinarizer.pkl','rb') as handle:
            lb = pickle.load(handle)
    except IOError:
        Warning("Could not find label binarizer pickle file 'LabelBinarizer.pkl' which should been generated by reason_to_group_mapping.py module, please check")
    # y = lb.transform(y)
    # n_classes = len(lb.classes_)
    ## start of cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc

    rf = RandomForestClassifier()
    lr = LogisticRegression()
    clf = OneVsRestClassifier(rf,-1)

    mario = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('cus_info', Pipeline([
                    ('col_selector', ColumnsSelector(['gender','age','type','royalty','haswww'])),
                    ('dummify',Recordify()),
                    ('vectorizer',DictVectorizer())
                    ])),
                ('numeric',Pipeline([
                    ('col_selector',ColumnsSelector(['n_events'])),
                    ('scaler',StandardScaler())
                    ])),
                ('events', Pipeline([
                    ('col_selector',ColumnsSelector('events')),
                    ('vectorizer',TfidfVectorizer())
                    ]))
            ])),
        ('classification',clf)
        ])

    mario.fit(X_train,y_train)

    y_score = mario.predict_proba(X_test)
    print y_score.shape
    # print(lb.classes_)
    # print(y_test)
    # print(y_score)
    from sklearn.metrics import roc_curve
    fpr, tpr, roc_auc = {},{},{}
    n_classes = len(np.unique(y_test))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i],tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(roc_auc)
    for i,v in enumerate(lb.classes_):
        print i,v
