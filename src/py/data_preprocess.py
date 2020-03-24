#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm 
import matplotlib.pyplot as plt

# %%
df = pd.read_excel('../../data/41707559_0_结直肠癌筛查偏好_1038_525(1).xls')
df

#%%
anes_data = sm.datasets.anes96.load(as_pandas=False)
anes_exog = anes_data.exog
anes_exog = sm.add_constant(anes_exog, prepend=False)
print(anes_data.exog[:5,:])
print(anes_data.endog[:30])
#%%
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()
print(mlogit_res.params)
#%%
mlogit_res.summary()
# %%
"""
结果保存
"""
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(7, 12))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(mlogit_res.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('../../res/test.png')

# %%
def get_choice_factors(str_choice):
    choice_dict = {'粪便潜血试验':0,'乙状结肠镜':1,'全结肠镜':2}
    Pain_dict = {'粪便潜血试验':0,'乙状结肠镜':60,'全结肠镜':100}
    choice,factors = str_choice.split('：')
    factors = factors.split('，')
    # print(factors)
    RISK_DES = int(factors[0][-3:-1])
    # print(factors[1][2:].split('年'))
    tmp = factors[1][2:].split('年')[0]
    FREQUENCY = int(tmp) if tmp != '1/3' else 1/3
    return choice_dict[choice],RISK_DES,FREQUENCY,Pain_dict[choice]
data_list = []
for i in range(525):
    sex = int(df.loc[i,'1、您的性别：']) - 1
    age = int(df.loc[i,'2、Q3：您的年龄']) - 1
    income = int(df.loc[i,'3、您的年收入是多少']) - 1
    region = int(df.loc[i,'4、Q2：2您所属的地区为']) - 1
    edu = int(df.loc[i,'5、Q5：5您的学历情况']) - 1
    tmp = int(df.loc[i,'6、就业情况'])
    work,retire = 0,0
    if tmp == 1:
        work = 1
    elif tmp ==2:
        retire = 1
    chronic = 1-(int(df.loc[i,'7、12有慢性病史']) - 1)
    tmp = int(df.loc[i,'8、12做过结直肠筛查'])
    if tmp == 1:
        check = 1
    else:
        check = 0
    hospital = 1-(int(df.loc[i,'9、12有过住院经历']) - 1)
    attention = 5 - int(df.loc[i,'10、平时是否关注健康状况'])
    base_dict = {
        'no':i,
        'sex':sex,
        'age':age,
        'income':income,
        'region':region,
        'edu':edu,
        'work':work,
        'retire':retire,
        'chronic':chronic,
        'check':check,
        'hospital':hospital,
        'attention':attention,
    }
    for j in range(11,29):
        # if j == 11:
        #     choice,risk_dec,frequency,pain = get_choice_factors(df.loc[i,'%s、请在以下的几种假设情况下选择您愿意接受的情况' % j])
        # else:
        #     choice,risk_dec,frequency,pain = get_choice_factors(df.loc[i,'%s、请您在以下几种假设情景中选择您愿意接受的一种方案' % j])
        choice,risk_dec,frequency,pain = get_choice_factors(df.iloc[i,j+5])

        item = base_dict.copy()
        item.update({
            'Q':j,
            'risk_dec':risk_dec,
            'frequency':frequency,
            'pain':pain,
            'choice':choice,
        })
        data_list.append(item)
new_df = pd.DataFrame(data_list)
new_df
# %%
exog_list = ['sex', 'age', 'income', 'region', 'edu', 'work', 'retire',
       'chronic', 'check', 'hospital', 'attention', 'risk_dec', 'frequency',
       'pain']
mlogit_mod = sm.MNLogit(new_df['choice'], new_df[exog_list])
mlogit_res = mlogit_mod.fit()
mlogit_res.summary()

# %%
new_df.to_excel('../../data/clean_data.xls',index=None)

# %%
