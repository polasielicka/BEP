import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import itertools, os

FUSION_TAGS = ["fusion_early", "fusion_late", "fusion_hybrid"]

CLASS_NAMES = {
    0: "Squat (C)", 1: "Squat WT", 2: "Squat FL",
    3: "Ext (C)",   4: "Ext NF",   5: "Ext LL",
    6: "Gait (C)",  7: "Gait NF",  8: "Gait HA"
}

def plot_confusion_matrix(cm, classes, title, savepath=None, normalize=True):
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)
        cm_display = cm * 100.0
        fmt = "{:.1f}%"
        cbar_label = "Percentage"
    else:
        cm_display = cm
        fmt = "{}"
        cbar_label = "Count"

    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(cm_display, interpolation='nearest')
    plt.title(title + (" (Normalized)" if normalize else ""))
    plt.colorbar(label=cbar_label)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm_display.shape[0]), range(cm_display.shape[1])):
        value = cm_display[i, j]
        plt.text(
            j, i, fmt.format(value),
            horizontalalignment="center",
            color="black",
            fontsize=9
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()

def evaluate_tag(tag: str):
    base = f"models/{tag}"
    y_test = np.load(base + "_y_test.npy", allow_pickle=True)
    y_pred = np.load(base + "_y_pred.npy", allow_pickle=True)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    print(f"{tag} | Test Accuracy: {acc:.3f} | Macro-F1: {f1m:.3f}")

    labels_sorted = np.unique(y_test)
    tick_labels = [CLASS_NAMES.get(int(i), str(i)) for i in labels_sorted]

    prec, rec, f1c, supp = precision_recall_fscore_support(
        y_test, y_pred, labels=labels_sorted, zero_division=0
    )
    print("\nPer-class metrics (ID: precision, recall, f1, support)")
    for cid, p, r, f, s in zip(labels_sorted, prec, rec, f1c, supp):
        print(f"{int(cid):2d}: {p:.3f}, {r:.3f}, {f:.3f}, n={s}")

    print("\nResults:\n", classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    plot_confusion_matrix(
        cm, tick_labels,
        title=f"{tag.replace('_',' ').title()} Confusion Matrix",
        savepath=fr"C:\Users\20230931\PycharmProjects\Recommender-Systems-for-Time-Series-Data-in-Knee-Osteoarthritis-Management\figures\{tag}_confusion_matrix.png",
        normalize=True
    )
    return acc, f1m

if __name__ == "__main__":
    results = {}
    for tag in FUSION_TAGS:
        print("\n" + "=" * 70)
        results[tag] = evaluate_tag(tag)
    print("\nSUMMARY:")
    for tag, (acc, f1m) in results.items():
        print(f"{tag:13s} | ACC={acc:.3f} | F1m={f1m:.3f}")
