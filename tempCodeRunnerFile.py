
# Evaluation
acc_fs = accuracy_score(y_test, y_pred_fs)
mcc_fs = matthews_corrcoef(y_test, y_pred_fs)
auc_fs = roc_auc_score(y_test, y_prob_fs)

print(f"Accuracy : {round(acc_fs * 100, 2)}%")
print(f"MCC      : {round(mcc_fs, 4)}")
print(f"AUC      : {round(auc_fs, 4)}")
