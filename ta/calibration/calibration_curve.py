import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def get_calibration_figure(model_calibrated, model, x, y, name):
    prob_pos = model_calibrated.predict_proba(x)[:, 1]
    prob_pos_orig = model.predict(x)

    clf_score = brier_score_loss(y, prob_pos, pos_label=1)
    clf_score_orig = brier_score_loss(y, prob_pos_orig, pos_label=1)

    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y, prob_pos, n_bins=10)
    fraction_of_positives_orig, mean_predicted_value_orig = \
        calibration_curve(y, prob_pos_orig, n_bins=10)

    fig_index = 1

    fig = plt.figure(fig_index, figsize=(8, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s (%1.3f)" % ('calibrated', clf_score))
    ax1.plot(mean_predicted_value_orig, fraction_of_positives_orig, "s-",
             label="%s (%1.3f)" % ('original', clf_score_orig))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label='calibrated',
             histtype="step", lw=2)
    ax2.hist(prob_pos_orig, range=(0, 1), bins=10, label='original',
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plots  (reliability curve) {name} dataset')

    ax2.set_yscale('log')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    return fig
