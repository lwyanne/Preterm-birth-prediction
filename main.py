def classify(df,tb,nul,mul):
    tb.loc['nonhis_white']['preterm_nul'] = nul[(nul['mracehisp'] == 1) & (nul['cat3']==0)].shape[0]
    tb.loc['nonhis_white']['term_nul'] = nul[(nul['mracehisp'] == 1) & (nul['cat3']==1)].shape[0]
    tb.loc['nonhis_black']['preterm_nul'] = nul[(nul['mracehisp'] == 2) & (nul['cat3']==0)].shape[0]
    tb.loc['nonhis_black']['term_nul'] = nul[(nul['mracehisp'] == 2) & (nul['cat3']==1)].shape[0]
    tb.loc['nonhis_ame_ind']['preterm_nul'] = nul[(nul['mracehisp'] ==3) & (nul['cat3']==0)].shape[0]
    tb.loc['nonhis_ame_ind']['term_nul'] = nul[(nul['mracehisp'] ==3) & (nul['cat3']==1)].shape[0]
    tb.loc['nonhis_asian']['preterm_nul'] = nul[(nul['mracehisp'] ==4) & (nul['cat3']==0)].shape[0]
    tb.loc['nonhis_asian']['term_nul'] = nul[(nul['mracehisp'] ==4) & (nul['cat3']==1)].shape[0]
    tb.loc['his']['preterm_nul'] = nul[(nul['mracehisp'] ==7) & (nul['cat3']==0)].shape[0]
    tb.loc['his']['term_nul'] = nul[(nul['mracehisp'] ==7) & (nul['cat3']==1)].shape[0]

    tb.loc['nonhis_white']['preterm_mul'] = mul[(mul['mracehisp'] == 1) & (mul['cat3']==0)].shape[0]
    tb.loc['nonhis_white']['term_mul'] = mul[(mul['mracehisp'] == 1) & (mul['cat3']==1)].shape[0]
    tb.loc['nonhis_black']['preterm_mul'] = mul[(mul['mracehisp'] == 2) & (mul['cat3']==0)].shape[0]
    tb.loc['nonhis_black']['term_mul'] = mul[(mul['mracehisp'] == 2) & (mul['cat3']==1)].shape[0]
    tb.loc['nonhis_ame_ind']['preterm_mul'] = mul[(mul['mracehisp'] ==3) & (mul['cat3']==0)].shape[0]
    tb.loc['nonhis_ame_ind']['term_mul'] = mul[(mul['mracehisp'] ==3) & (mul['cat3']==1)].shape[0]
    tb.loc['nonhis_asian']['preterm_mul'] = mul[(mul['mracehisp'] ==4) & (mul['cat3']==0)].shape[0]
    tb.loc['nonhis_asian']['term_mul'] = mul[(mul['mracehisp'] ==4) & (mul['cat3']==1)].shape[0]
    tb.loc['his']['preterm_mul'] = mul[(mul['mracehisp'] ==7) & (mul['cat3']==0)].shape[0]
    tb.loc['his']['term_mul'] = mul[(mul['mracehisp'] ==7) & (mul['cat3']==1)].shape[0]

    tb.loc['age12-20','preterm_nul']=nul[(nul['mager'] <=20) & (nul['cat3']==0)].shape[0]
    tb.loc['age21-30','preterm_nul']=nul[(nul['mager'] <=30) & (nul['mager'] > 20) & (nul['cat3']==0)].shape[0]
    tb.loc['age31-40','preterm_nul']=nul[(nul['mager'] <=40) & (nul['mager'] > 30) & (nul['cat3']==0)].shape[0]
    tb.loc['age41-50','preterm_nul']=nul[(nul['mager'] <=50) & (nul['mager'] > 40) & (nul['cat3']==0)].shape[0]

    tb.loc['age12-20', 'term_nul'] = nul[(nul['mager'] <= 20) & (nul['cat3'] == 1)].shape[0]
    tb.loc['age21-30', 'term_nul'] = nul[(nul['mager'] <= 30) & (nul['mager'] > 20) & (nul['cat3'] == 1)].shape[0]
    tb.loc['age31-40', 'term_nul'] = nul[(nul['mager'] <= 40) & (nul['mager'] > 30) & (nul['cat3'] == 1)].shape[0]
    tb.loc['age41-50', 'term_nul'] = nul[(nul['mager'] <= 50) & (nul['mager'] > 40) & (nul['cat3'] == 1)].shape[0]

    tb.loc['age12-20','preterm_mul']=mul[(mul['mager'] <=20) & (mul['cat3']==0)].shape[0]
    tb.loc['age21-30','preterm_mul']=mul[(mul['mager'] <=30) & (mul['mager'] > 20) & (mul['cat3']==0)].shape[0]
    tb.loc['age31-40','preterm_mul']=mul[(mul['mager'] <=40) & (mul['mager'] > 30) & (mul['cat3']==0)].shape[0]
    tb.loc['age41-50','preterm_mul']=mul[(mul['mager'] <=50) & (mul['mager'] > 40) & (mul['cat3']==0)].shape[0]

    tb.loc['age12-20', 'term_mul'] = mul[(mul['mager'] <= 20) & (mul['cat3'] == 1)].shape[0]
    tb.loc['age21-30', 'term_mul'] = mul[(mul['mager'] <= 30) & (mul['mager'] > 20) & (mul['cat3'] == 1)].shape[0]
    tb.loc['age31-40', 'term_mul'] = mul[(mul['mager'] <= 40) & (mul['mager'] > 30) & (mul['cat3'] == 1)].shape[0]
    tb.loc['age41-50', 'term_mul'] = mul[(mul['mager'] <= 50) & (mul['mager'] > 40) & (mul['cat3'] == 1)].shape[0]


    tb.loc['high','preterm_nul']=nul[(nul['meduc'] <=3) & (nul['cat3']==0)].shape[0]
    tb.loc['asso','preterm_nul']=nul[((nul['meduc'] ==5) |(nul['meduc']==4)) & (nul['cat3']==0)].shape[0]
    tb.loc['ba','preterm_nul']=nul[(nul['meduc'] ==6) & (nul['cat3']==0)].shape[0]
    tb.loc['ad','preterm_nul']=nul[((nul['meduc'] ==7) | (nul['meduc'] ==8)) & (nul['cat3']==0)].shape[0]

    tb.loc['high','term_nul']=nul[(nul['meduc'] <=3) & (nul['cat3']==1)].shape[0]
    tb.loc['asso','term_nul']=nul[((nul['meduc'] ==5) |(nul['meduc']==4)) & (nul['cat3']==1)].shape[0]
    tb.loc['ba','term_nul']=nul[(nul['meduc'] ==6) & (nul['cat3']==1)].shape[0]
    tb.loc['ad','term_nul']=nul[((nul['meduc'] ==7) | (nul['meduc'] ==8)) & (nul['cat3']==1)].shape[0]

    tb.loc['high','preterm_mul']=mul[(mul['meduc'] <=3) & (mul['cat3']==0)].shape[0]
    tb.loc['asso','preterm_mul']=mul[((mul['meduc'] ==5)|(mul['meduc']==4))  & (mul['cat3']==0)].shape[0]
    tb.loc['ba','preterm_mul']=mul[(mul['meduc'] ==6) & (mul['cat3']==0)].shape[0]
    tb.loc['ad','preterm_mul']=mul[((mul['meduc'] ==7) | (mul['meduc'] ==8)) & (mul['cat3']==0)].shape[0]

    tb.loc['high','term_mul']=mul[(mul['meduc'] <=3) & (mul['cat3']==1)].shape[0]
    tb.loc['asso','term_mul']=mul[((mul['meduc'] ==5)|(mul['meduc']==4))  & (mul['cat3']==1)].shape[0]
    tb.loc['ba','term_mul']=mul[(mul['meduc'] ==6) & (mul['cat3']==1)].shape[0]
    tb.loc['ad','term_mul']=mul[((mul['meduc'] ==7) | (mul['meduc'] ==8)) & (mul['cat3']==1)].shape[0]
    return tb