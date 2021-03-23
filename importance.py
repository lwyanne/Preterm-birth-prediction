from utils import *
logger=setup_logs('importance','importance')
batchsize=1200
procs = [FillMissing, Categorify, Normalize]
#
cat_names,cont_names,variables,dep_var,embdic=get_variables_nul()
logger.info('loading data...')
train_val_df,test_df=read_train_val_nul(fillna='ANN',split_x_y=False),\
    read_test_nul(fillna='ANN',split_x_y=False)

path='/home/shuying/predictBirth/code'
val_idx=range(len(train_val_df)-int(0.5e6),len(train_val_df))
logger.info('creating data bunch')
data = TabularDataBunch.from_df(path, train_val_df, dep_var, valid_idx=val_idx,bs=batchsize,procs=procs,test_df=test_df,cont_names=cont_names,cat_names=cat_names,num_workers=24)
layer=[60,30]
drop=[0.2, 0.1]
wt=[1, 0.1]
bestLearner='NulDelete_60_30_0.2_0.1'

learn = tabular_learner(data,
                    layers=layer,
                    ps=drop,
                    emb_szs=embdic,
                    metrics=accuracy,
                    loss_func=nn.CrossEntropyLoss(weight=torch.tensor(wt).cuda()))
eval_importance_nul(learn,bestLearner)

#
# # ===================Mul
# cat_names,cont_names,variables,dep_var,embdic=get_variables_mul()
# logger.info('\n\n\n\nloading data for multiparous...')
# train_val_df,test_df=read_train_val_mul(fillna='ANN',split_x_y=False),\
#     read_test_mul(fillna='ANN',split_x_y=False)
# val_idx=range(len(train_val_df)-int(0.5e6),len(train_val_df))
# logger.info('creating data bunch')
# # tmp=train_val_df['cat3'][1]
# # train_val_df.iloc[1,:-1]=np.nan
# # train_val_df['cat3'][1]=tmp
#
# data = TabularDataBunch.from_df(path, train_val_df, dep_var, classes=[0.0,1.0],valid_idx=val_idx,bs=batchsize,procs=procs,test_df=test_df,cont_names=cont_names,cat_names=cat_names,num_workers=24)
# layer=[60,30]
# drop=[0.2, 0.1]
# wt=[1, 0.1]
# bestLearner='MulDelete_60_30_0.4_0.3'
#
# learn = tabular_learner(data,
#                     layers=layer,
#                     ps=drop,
#                     emb_szs=embdic,
#                     metrics=accuracy,
#                     loss_func=nn.CrossEntropyLoss(weight=torch.tensor(wt).cuda()))
# eval_importance_mul(learn,bestLearner)