from utils import *
resume=0

logger=setup_logs('mul_ANN')
batchsize=1200
procs = [FillMissing, Categorify, Normalize]
#--------mul
cat_names,cont_names,variables,dep_var,embdic=get_variables_mul()
logger.info('loading data...')
train_val_df,test_df=read_train_val_mul(fillna='ANN',split_x_y=False),\
    read_test_mul(fillna='ANN',split_x_y=False)
# tmp=train_val_df['cat3'][1]
# train_val_df.iloc[1,:-1]=np.nan
# train_val_df['cat3'][1]=tmp
path='/home/shuying/predictBirth/code'
val_idx=range(len(train_val_df)-int(0.5e6),len(train_val_df))
logger.info('creating data bunch')
data = TabularDataBunch.from_df(path, train_val_df, dep_var, valid_idx=val_idx,bs=batchsize,procs=procs,test_df=test_df,cont_names=cont_names,cat_names=cat_names,num_workers=24)
layer=[60,30]
drop=[0.4, 0.3]
wt=[1, 0.1]
bestLearner='MulDelete_60_30_0.4_0.3'
learn = tabular_learner(data,
                    layers=layer,
                    ps=drop,
                    emb_szs=embdic,
                    metrics=accuracy,
                    loss_func=nn.CrossEntropyLoss(weight=torch.tensor(wt).cuda()),
                        callback_fns=AUROC)
logger.info(str(learn))

if resume:
    learn.load(bestLearner)
    logger.info('load learner from %s'%bestLearner)
else:
    logger.info('Starting Fitting')
    learn.fit_one_cycle(10, 0.05,
                        callbacks=[SaveModelCallback(learn, every='improvement', monitor='AUROC', name=bestLearner)])



y=learn.get_preds(DatasetType.Train)
y2=y[1]
Y2=1-y2
y1=y[0]
Y1=to_np(y1[:,0])
auc, auc_var, ci = auc_ci_Delong(y_true=Y2,y_scores=Y1)
logger.info('Training: %s, AUROC of %.3f (%%95 CI: %.3f-%.3f)'%('ANN', auc, ci[0],ci[1]))




y=learn.get_preds(DatasetType.Valid)
y2=y[1]
Y2=1-y2

y1=y[0]
Y1=to_np(y1[:,0])
auc, auc_var, ci = auc_ci_Delong(y_true=Y2,y_scores=Y1)
logger.info('Validation: %s, AUROC of %.3f (%%95 CI: %.3f-%.3f)'%('ANN', auc, ci[0],ci[1]))


y=learn.get_preds(ds_type=DatasetType.Test)
Y2=pd.read_csv(pj(path,'yMulTest'),names=['0'])
Y2=1-np.array(Y2['0'].tolist())
y1=y[0]
Y1=to_np(y1[:,0])
auc, auc_var, ci = auc_ci_Delong(y_true=Y2,y_scores=Y1)
logger.info('Test: %s, AUROC of %.3f (%%95 CI: %.3f-%.3f)'%('ANN', auc, ci[0],ci[1]))
np.savetxt("ANNMulTest.csv" , Y1, delimiter=",")


