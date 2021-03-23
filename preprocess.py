import csv
import pandas as pd
from utils import *
# cols=con_mul+cat_mul+['oegest_comb']
# tb2014=pd.read_csv('../natl2014.csv',usecols=cols)
# print('2014 done, columns are', tb2014.columns)
# tb2015=pd.read_csv('../natl2015.csv',usecols=cols)
# print('2015 done')
# tb2016=pd.read_csv('../natl2016.csv',usecols=cols)
# print('2016 done')
# tb2017=pd.read_csv('../natl2017.csv',usecols=cols)
# print('2017 done')
# tb2018=pd.read_csv('../natl2018us.csv',usecols=cols)
# print('2018 done')
# tb2014_2017=pd.concat([tb2014,tb2015,tb2016,tb2017])
# ori_1417=(len(tb2014_2017))
# ori_18=(len(tb2018))
# print('original 2014-2017 has size %s, 2018 has size %s'%(ori_1417,ori_18))
#
# tb2014_2017=tb2014_2017.reset_index()
#
# col_list=['rf_pdiab','rf_gdiab','rf_phype','rf_ghype','rf_ehype','rf_ppterm','rf_inftr','rf_fedrg','rf_artec','ip_gon','ip_syph','ip_chlam','ip_hepatb','ip_hepatc','sex','mar_p','wic']
# tb2014_2017=tb2014_2017[tb2014_2017['oegest_comb']!=99]
# tb2018=tb2018[tb2018['oegest_comb']!=99]
# print('remained oegest_comb %s for 2014-2017, %s for 2018'%(len(tb2014_2017['oegest_comb']==99),len(tb2018['oegest_comb']==99)))
# tb2014_2017,tb2018=define_missing(tb2014_2017),define_missing(tb2018)
# # tb2014_2017,tb2018=recode(tb2014_2017,col_list),recode(tb2018,col_list)
# tb2014_2017=tb2014_2017[tb2014_2017['precare']<6]
# tb2018=tb2018[tb2018['precare']<6]
# precare_1417=(len(tb2014_2017))
# precare_18=(len(tb2018))
# d1_1417= precare_1417-ori_1417
# d1_18=precare_18-ori_18
# tb2014_2017.to_csv('../tb1417.csv')
# tb2018.to_csv('../tb2018.csv')
# print('after screening out precare>=6,  2014-2017 has size %s, 2018 has size %s.\n screened out %s subjects, %s subjects seperately'%(precare_1417,precare_18,d1_1417,d1_18))
#
#
# #
# tb2018=pd.read_csv('../tb2018.csv')
# tb2014_2017=pd.read_csv('../tb1417.csv')
# tb2014_2017=preterm_recode(tb2014_2017)
# tb2018=preterm_recode(tb2018)
# print(len(tb2018))
# nulli_2014_2017,multi_2014_2017=split_nul_mul(tb2014_2017)
# nulli_2018,multi_2018=split_nul_mul(tb2018)
# print(len(tb2018))
# nulli_2014_2017,multi_2014_2017=select_useful(nulli_2014_2017,multi_2014_2017)
# nulli_2018,multi_2018=select_useful(nulli_2018,multi_2018)
# print(len(nulli_2018))

# nulli_2014_2017.to_csv('../nulli_2014_2018.csv')
# nulli_2014_2017[nulli_2014_2017['cat3']==0].to_csv('../pretermNul.csv')
# nulli_2014_2017[nulli_2014_2017['cat3']==1].to_csv('../normalNul.csv')
multi_2014_2017=pd.read_csv('../multi_2014_2018.csv')
# multi_2014_2017.to_csv('../multi_2014_2018.csv')
multi_2014_2017[multi_2014_2017['cat3']==0].to_csv('../pretermMul.csv')
multi_2014_2017[multi_2014_2017['cat3']==1].to_csv('../normalMul.csv')
# nulli_2018.to_csv('../nulli_2018.csv')
# multi_2018.to_csv('../multi_2018.csv')



flag = 1
with open('/home/shuying/predictBirth/normalMul.csv', newline='') as f1:
    with open('/home/shuying/predictBirth/pretermMul.csv', newline='') as f2:
        f2.readline()
        with open('../mul_1417_even.csv', 'w+', newline='') as f:
            # while flag < 845369:
            a=1
            while a:
                if flag % 10 != 0:
                    a=f1.readline()
                    f.write(a)
                else:
                    f.write(f2.readline())
                flag += 1
print('mul done')
# flag = 1
# with open('/home/shuying/predictBirth/normalNul.csv', newline='') as f1:
#     with open('/home/shuying/predictBirth/pretermNul.csv', newline='') as f2:
#         f2.readline()
#         with open('../nul_1417_even.csv', 'w+', newline='') as f:
#             a=1
#             while a:
#                 # while True:
#                 if flag % 11 != 0:
#                     a=f1.readline()
#                     f.write(a)
#                 else:
#                     f.write(f2.readline())
#                 flag += 1
# print('nul done')
# #
# f=pd.read_csv('/home/shuying/predictBirth/nul_1417_even.csv')
# for col in cat_nul:
#     # if 'U' in f[col].unique():
#     print(col)
#     print(f[col].value_counts()/len(f))
#
# f=pd.read_csv('/home/shuying/predictBirth/mul_1417_even.csv')
# for col in cat_mul:
#     # if 'U' in f[col].unique():
#     print(col)
#     print(f[col].value_counts()/len(f))