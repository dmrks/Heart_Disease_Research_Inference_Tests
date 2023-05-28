# import libraries
import codecademylib3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

# load data  + 1
heart = pd.read_csv('heart_disease.csv')
print(heart.head())

#2
sns.boxplot(x = heart.thalach,y = heart.heart_disease)
plt.show()
plt.clf()

#3
thalach_hd = heart.thalach[heart.heart_disease == "presence"]
thalach_no_hd = heart.thalach[heart.heart_disease == "absence"]

#4 Mean difference = 19.11905597473242 / Median difference = 19.0
print(np.mean(thalach_no_hd)-np.mean(thalach_hd ))
print(np.median(thalach_no_hd)-np.median(thalach_hd))

#5 +6 pval = 3.456964908430172e-14 -> Significant
tstat, pval =ttest_ind(thalach_hd, thalach_no_hd)
print(pval)

#8
sns.boxplot(x = heart.cp,y = heart.thalach)
plt.show()
plt.clf()

#9
thalach_typical = heart.thalach[heart.cp == "typical angina"]
thalach_asymptom =heart.thalach[heart.cp == "asymptomatic"]
thalach_nonangin =heart.thalach[heart.cp == "non-anginal pain"]
thalach_atypical =heart.thalach[heart.cp == "atypical angina"]

#10 pval = 1.9065505247705008e-10 -> Significant

Fstat, pval = f_oneway(thalach_typical, thalach_asymptom, thalach_nonangin, thalach_atypical)
print(pval)

#11

tukey_results =pairwise_tukeyhsd(heart.thalach, heart.cp, 0.05)
print(tukey_results)

#12

Xtab =pd.crosstab(heart.cp, heart.heart_disease)
print(Xtab)

#13 pval 1.2517106007837527e-17 => Significance => significant association between chest pain type and whether or not someone is diagnosed with heart disease

chi2, pval, dof, expected =chi2_contingency(Xtab)
print(pval)


