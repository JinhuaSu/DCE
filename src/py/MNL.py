#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm 
import matplotlib.pyplot as plt
#%%
new_df = pd.read_excel('../../data/clean_data.xls',index=None)
new_df['intercept'] = 1.0
exog_list = ['sex', 'age', 'income', 'region', 'edu', 'work', 'retire',
       'chronic', 'check', 'hospital', 'attention', 'risk_dec', 'frequency',
       'pain']
mlogit_mod = sm.MNLogit(new_df['choice'], new_df[exog_list])
mlogit_res = mlogit_mod.fit(method='bfgs',maxiter=100)
mlogit_res.summary()
#%%
np.exp(mlogit_res.params) - 1
# %%
plt.rc('figure', figsize=(7, 8))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(mlogit_res.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('../../res/naive.png')

# %%
hist_list = ['sex', 'age', 'income', 'region', 'edu', 'work',
       'retire', 'chronic', 'check', 'hospital', 'attention', 'risk_dec',
       'frequency', 'pain', 'choice']
new_df[hist_list].hist(figsize=(10,12))
plt.savefig('../../res/hist.png')

# %%
new_df['intercept'] = 1.0
exog_list = [ 'risk_dec', 'frequency','pain']
mlogit_mod = sm.MNLogit(new_df['choice'], new_df[exog_list])
mlogit_res = mlogit_mod.fit(method='bfgs',maxiter=100)
mlogit_res.summary()

# %%
mlogit_res._results.tvalues


# %%
def step(feature_list,enog_name,df,threshold):
    select_feature = []
    while(len(feature_list)>0):
        max_value, max_feature = 0,None
        for feature in feature_list:
            print(feature)
            mlogit_mod = sm.MNLogit(df[enog_name], df[select_feature + [feature]])
            try:
                mlogit_res = mlogit_mod.fit()
            except:
                print('singular matrix')
                continue
            if threshold == None:
                try:
                    value = np.sum(np.abs(mlogit_res._results.tvalues[-1]))
                except:
                    print('bug:%s' %feature)
                    continue
                if value >max_value:
                    max_value = value
                    max_feature = feature
            elif np.sum(mlogit_res._results.tvalues[-1]) > 2 * threshold:
                print('\n%s\n' % feature)
                feature_list.remove(feature)
                select_feature.append(feature)
        if(max_feature!=None):
            print('\n%s\n' % max_feature)
            feature_list.remove(max_feature)
            select_feature.append(max_feature)
        else:
            break
    print(select_feature)
    return select_feature
#%%
exog_list = ['sex', 'age', 'income', 'region', 'edu', 'work', 'retire',
       'chronic', 'check', 'hospital', 'attention', 'risk_dec', 'frequency',
       'pain']
step(exog_list+['intercept'],'choice',new_df,None)

# %%
new_df['intercept'] = 1.0
select_feature = ['frequency', 'intercept', 'income', 'sex', \
    'edu', 'work', 'retire', 'check', 'hospital', 'age', \
        'chronic', 'region', 'risk_dec', 'attention']
for i in range(len(select_feature)):
    mlogit_mod = sm.MNLogit(new_df['choice'], new_df[select_feature[:i+1]])
    mlogit_res = mlogit_mod.fit()
    plt.rc('figure', figsize=(7, 3 + int((float(i)/len(select_feature))*6)))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 0.05, str(mlogit_res.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../../res/res1_%s.png'%(i+1))
    plt.show()
#%%
' >> '.join(select_feature)
# %%
interact_df = new_df.copy(deep=True)
factors = ['risk_dec','frequency','pain']
characters = ['income', 'sex', \
    'edu', 'work', 'retire', 'check', 'hospital', 'age', \
        'chronic', 'region','attention']
interact_list = []
for character in characters:
    for factor in factors:
        name = character+'*' +factor
        interact_list.append(name)
        interact_df[name] = interact_df[character] * interact_df[factor]
interact_list

# %%
exog_list = ['sex', 'age', 'income', 'region', 'edu', 'work', 'retire',
       'chronic', 'check', 'hospital', 'attention', 'risk_dec', 'frequency',
       'pain']
inter_select = step(exog_list+interact_list+['intercept'],'choice',interact_df,None)
inter_select
#%%
' >> '.join(inter_select)

# %%
"""
保存数据
"""
for i in range(len(inter_select)):
    mlogit_mod = sm.MNLogit(interact_df['choice'], interact_df[inter_select[:i+1]])
    mlogit_res = mlogit_mod.fit()
    plt.rc('figure', figsize=(8, 3 + int((float(i)/len(inter_select))*10)))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 0.05, str(mlogit_res.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../../res/res2_%s.png'%(i+1))
    plt.show()

# %%


new_df['intercept'] = 1.0
select_feature = ['frequency', 'intercept', 'income', 'sex', \
    'edu', 'work', 'retire', 'check', 'hospital', 'age', \
        'chronic', 'region', 'risk_dec', 'attention']
for i in range(len(select_feature)):
    mlogit_mod = sm.MNLogit(new_df['choice'], new_df[select_feature[:i+1]])
    mlogit_res = mlogit_mod.fit()
    s = mlogit_res.summary()
    with open('../../res/res1_%s.csv' % (i+1),'w') as f:
        f.write(s.as_csv())
    with open('../../res/res1_%s.txt' % (i+1),'w') as f:
        f.write(s.as_text())
#%%
for i in range(len(inter_select)):
    mlogit_mod = sm.MNLogit(interact_df['choice'], interact_df[select_feature[:i+1]])
    mlogit_res = mlogit_mod.fit()
    s = mlogit_res.summary()
    with open('../../res/res2_%s.csv' % (i+1),'w') as f:
        f.write(s.as_csv())
    with open('../../res/res2_%s.txt' % (i+1),'w') as f:
        f.write(s.as_text())
