from utils import *
logger=setup_logs('clfs','clfs')

#-----------------Nul
logger.info('----------\nReading Nulliparous data\n----------\n\n')
train_x,train_y=read_train_nul(fillna='manual',split_x_y=True)
val_x,val_y=read_val_nul(fillna='manual',split_x_y=True)
test_x,test_y=read_test_nul(fillna='manual',split_x_y=True)
np.savetxt('yFirstTest',test_y,delimiter=",")
np.savetxt('yFirstVal',val_y,delimiter=",")

logger.info('----------\nStart classification\n----------\n')
#define clfs
rf = RandomForestClassifier(n_estimators=40)
lr = LogisticRegression()
xg=XGBClassifier(n_estimators=40,n_jobs=10)
rf.name,lr.name,xg.name='rf','lr','xg'
# fit the models
# for clf in [rf,lr,xg]:
# for clf in [rf,lr,xg]:
#     clf.fit(train_x,train_y)
#     logger.info('----------\n%s\n----------\n'%clf.name)
#     train_pred=clf.predict_proba(train_x)[:,1]
#     auc, auc_var, ci = auc_ci_Delong(y_true=train_y, y_scores=train_pred)
#     logger.info('training: %s, AUROC of %.3f (%%95 CI: %.3f-%.3f)'%(clf.name, auc, ci[0],ci[1]))
#     del train_pred
#     val_pred=clf.predict_proba(val_x)[:,1]
#     auc, auc_var, ci = auc_ci_Delong(y_true=val_y, y_scores=val_pred)
#     logger.info('Validation: %s, AUROC of %.3f (%%95 CI: %.3f-%.3f)'%(clf.name, auc, ci[0],ci[1]))
#     del val_pred
#     test_pred=clf.predict_proba(test_x)[:,1]
#     auc, auc_var, ci = auc_ci_Delong(y_true=test_y, y_scores=test_pred)
#     logger.info('Test: %s, AUROC of %.3f (%%95 CI: %.3f-%.3f)'%(clf.name, auc, ci[0],ci[1]))
#     np.savetxt("%sFirstTest.csv"%clf.name, test_pred, delimiter=",")
#     del test_pred
# del train_x,train_y,val_y,val_x,test_y,test_x
#-----------------multi
logger.info('----------\nReading Multiparous data\n----------\n\n')
train_x,train_y=read_train_mul(fillna='manual',split_x_y=True)
val_x,val_y=read_val_mul(fillna='manual',split_x_y=True)
test_x,test_y=read_test_mul(fillna='manual',split_x_y=True)
np.savetxt('yMulTest',test_y,delimiter=",")
np.savetxt('yMulVal',val_y,delimiter=",")

logger.info('----------\nStart classification\n----------\n')
#define clfs
rf = RandomForestClassifier(n_estimators=40)
lr = LogisticRegression()
xg=XGBClassifier(n_estimators=40,n_jobs=10)
rf.name,lr.name,xg.name='rf','lr','xg'
# fit the models
# for clf in [rf,lr,xg]:
# for clf in [rf,lr,xg]:
for clf in [xg]:
    clf.fit(train_x,train_y)
    logger.info('----------\n%s\n----------\n'%clf.name)
    train_pred=clf.predict_proba(train_x)[:,1]
    val_pred=clf.predict_proba(val_x)[:,1]
    test_pred=clf.predict_proba(test_x)[:,1]
    auc, auc_var, ci = auc_ci_Delong(y_true=train_y, y_scores=train_pred)
    logger.info('training: %s, AUROC of %.3f (%%95 CI: %.3f:%.3f)' % (clf.name, auc, ci[0], ci[1]))
    auc, auc_var, ci = auc_ci_Delong(y_true=val_y, y_scores=val_pred)
    logger.info('Validation: %s, AUROC of %.3f (%%95 CI: %.3f:%.3f)'%(clf.name, auc, ci[0],ci[1]))
    auc, auc_var, ci = auc_ci_Delong(y_true=test_y, y_scores=test_pred)
    logger.info('Test: %s, AUROC of %.3f (%%95 CI: %.3f:%.3f)'%(clf.name, auc, ci[0],ci[1]))
    np.savetxt("%sMulTest.csv"%clf.name, test_pred, delimiter=",")





