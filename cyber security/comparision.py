
import matplotlib.pyplot as plt
from pylab import *
figure()

###Performance metrics

##### Accuracy
t = [1,2,3,4,5,6,7,8]
tr1 = [86.34,96.24,97.16,95.5,95.19,94.64,95.14,99.23]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('MLP','LSTM','CNN+LSTM','SVM','Bayes','Random Forest','IDCNN','Proposed'))
#xlabel('Data Samples',fontsize=13)
ylabel('Accuracy (%)',fontsize=13)
title('')
plt.xticks(rotation=90)
grid(True)
#ylim([97, 98])
show()
##### Recall
t = [1,2,3,4,5,6,7,8]
tr1 = [86.25,89.89,98.1,98.12,92.84,90.89,90.17,99.22]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('MLP','LSTM','CNN+LSTM','SVM','Bayes','Random Forest','IDCNN','Proposed'))
#xlabel('Data Samples',fontsize=13)
ylabel('Recall (%)',fontsize=13)
title('')
plt.xticks(rotation=90)
grid(True)
#ylim([97, 98])
show()
##### Precision
t = [1,2,3,4,5,6,7,8]
tr1 = [88.47,98.44,97.51,97.72,92.56,90.1,98.14,99.24]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('MLP','LSTM','CNN+LSTM','SVM','Bayes','Random Forest','IDCNN','Proposed'))
#xlabel('Data Samples',fontsize=13)
ylabel('Precision (%)',fontsize=13)
title('')
plt.xticks(rotation=90)
grid(True)
#ylim([97, 98])
show()
#####F1-Score
t = [1,2,3,4,5,6,7,8]
tr1 = [86,96.1,97.06,95,95.1,94,95.1,99.23]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('MLP','LSTM','CNN+LSTM','SVM','Bayes','Random Forest','IDCNN','Proposed'))
#xlabel('Data Samples',fontsize=13)
ylabel('F1-Score (%)',fontsize=13)
title('')
plt.xticks(rotation=90)
grid(True)
#ylim([97, 98])
show()
##### Accuracy
t = [1,2,3,4]
tr1 = [99.13,99.21,99.22,99.23]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('KDD','DAPRA','KDD-CUP99','NSL-KDD'))
xlabel('Dataset',fontsize=13)
ylabel('Accuracy (%)',fontsize=13)
title('')
plt.xticks(rotation=45)
grid(True)
ylim([99, 99.5])
show()
##### Precision
t = [1,2,3,4]
tr1 = [99.21,99.23,99.22,99.24]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('KDD','DAPRA','KDD-CUP99','NSL-KDD'))
xlabel('Dataset',fontsize=13)
ylabel('Precision (%)',fontsize=13)
title('')
plt.xticks(rotation=45)
grid(True)
ylim([99, 99.5])
show()
##### Recall
t = [1,2,3,4]
tr1 = [99.22,99.21,99.21,99.22]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('KDD','DAPRA','KDD-CUP99','NSL-KDD'))
xlabel('Dataset',fontsize=13)
ylabel('Recall (%)',fontsize=13)
title('')
plt.xticks(rotation=45)
grid(True)
ylim([99, 99.5])
show()
##### F1-Score
t = [1,2,3,4]
tr1 = [99.22,99.22,99.23,99.23]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('KDD','DAPRA','KDD-CUP99','NSL-KDD'))
xlabel('Dataset',fontsize=13)
ylabel('F1-Score(%)',fontsize=13)
title('')
plt.xticks(rotation=45)
grid(True)
ylim([99, 99.5])
show()
#####KDD Dataset
t = [1,2,3,4,5,6,7,8]
tr1=[86.3,96.12,97.06,95.3,95,94.54,95.04,99.22]
tr2=[86.15,89.87,98.1,98,92.64,90.79,90.17,99.21]
tr3=[88.37,98.34,97.01,97.62,92.46,90,98.04,99.22]
tr4 = [86,96.1,96.06,94,95,93.9,95,99.23]

plot(t, tr1,marker='o',label='Accuracy', markerfacecolor='blue', markersize=6,linewidth=2.0)

plot(t, tr2,marker='o',label='Recall', markerfacecolor='blue', markersize=6,linewidth=2.0)

plot(t, tr3,marker='o',label='Precision', markerfacecolor='blue', markersize=6,linewidth=2.0)

plot(t, tr4,marker='o',label='F1-Score', markerfacecolor='blue', markersize=6,linewidth=2.0)


xticks(t , ('MLP','LSTM','CNN+LSTM','SVM','Bayes','Random Forest','IDCNN','Proposed'))
#xlabel('Data Samples',fontsize=13)
ylabel('KDD Dataset',fontsize=13)
title('')
legend()
plt.xticks(rotation=90)
grid(True)
#ylim([97, 98])
show()
##### DAPRA
t = [1,2,3,4,5,6,7,8]
tr1=[86.26,96.02,97.05,95.26,94.9,92.34,95.04,99.24]
tr2=[86.05,89.8,98,98,92.54,90.68,90.07,99.23]
tr3=[88.27,98.32,97,97.52,92.36,90,98,99.23]
tr4 = [85.9,96,96,94,95,93.8,95,99.23]

plot(t, tr1,marker='o',label='Accuracy', markerfacecolor='blue', markersize=6,linewidth=2.0)

plot(t, tr2,marker='o',label='Recall', markerfacecolor='blue', markersize=6,linewidth=2.0)

plot(t, tr3,marker='o',label='Precision', markerfacecolor='blue', markersize=6,linewidth=2.0)

plot(t, tr4,marker='o',label='F1-Score', markerfacecolor='blue', markersize=6,linewidth=2.0)


xticks(t , ('MLP','LSTM','CNN+LSTM','SVM','Bayes','Random Forest','IDCNN','Proposed'))
#xlabel('Data Samples',fontsize=13)
ylabel('DAPRA Dataset',fontsize=13)
title('')
legend()
plt.xticks(rotation=90)
grid(True)
#ylim([97, 98])
show()

##### KDD-CUP99
t = [1,2,3,4,5,6,7,8]
tr1=[86.16,96,97,95.16,94.8,92.14,95,99.22]
tr2=[86,89.7,98,98,92.44,90.58,90.05,99.23]
tr3=[88.17,98.22,97,97.42,92.26,90,98,99.23]
tr4 = [95.8,96,96,94.1,95.01,93.8,95,99.24]

plot(t, tr1,marker='o',label='Accuracy', markerfacecolor='blue', markersize=6,linewidth=2.0)

plot(t, tr2,marker='o',label='Recall', markerfacecolor='blue', markersize=6,linewidth=2.0)

plot(t, tr3,marker='o',label='Precision', markerfacecolor='blue', markersize=6,linewidth=2.0)

plot(t, tr4,marker='o',label='F1-Score', markerfacecolor='blue', markersize=6,linewidth=2.0)


xticks(t , ('MLP','LSTM','CNN+LSTM','SVM','Bayes','Random Forest','IDCNN','Proposed'))
#xlabel('Data Samples',fontsize=13)
ylabel('KDD-CUP99 Dataset',fontsize=13)
title('')
legend()
plt.xticks(rotation=90)
grid(True)
#ylim([97, 98])
show()
###performance no of data samples
t = [1,2,3,4,5,6,7,8]
tr1 = [99.22,99.22,99.23,99.23,99.23,99.24,99.245,99.25]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('20k','40k','60k','80k','100k','120k','140k','160k'))
xlabel('No.of Data Samples',fontsize=13)
ylabel('Accuracy(%)',fontsize=13)
title('')
plt.xticks(rotation=45)
grid(True)
ylim([99, 99.5])
show()

t = [1,2,3,4,5,6,7,8]
tr1 = [99.21,99.22,99.233,99.253,99.273,99.244,99.245,99.275]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('20k','40k','60k','80k','100k','120k','140k','160k'))
xlabel('No.of Data Samples',fontsize=13)
ylabel('Precision(%)',fontsize=13)
title('')
plt.xticks(rotation=45)
grid(True)
ylim([99, 99.5])
show()

t = [1,2,3,4,5,6,7,8]
tr1 = [99.31,99.32,99.333,99.353,99.373,99.344,99.345,99.375]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('20k','40k','60k','80k','100k','120k','140k','160k'))
xlabel('No.of Data Samples',fontsize=13)
ylabel('Recall(%)',fontsize=13)
title('')
plt.xticks(rotation=45)
grid(True)
ylim([99, 99.5])
show()
t = [1,2,3,4,5,6,7,8]
tr1 = [99.11,99.12,99.133,99.153,99.173,99.144,99.145,99.175]

plot(t, tr1,marker='o', markerfacecolor='blue', markersize=6,label='',linewidth=2.0)



xticks(t , ('20k','40k','60k','80k','100k','120k','140k','160k'))
xlabel('No.of Data Samples',fontsize=13)
ylabel('F1-Score(%)',fontsize=13)
title('')
plt.xticks(rotation=45)
grid(True)
ylim([99, 99.5])
show()