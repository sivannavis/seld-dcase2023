import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

plt.style.use("seaborn")
sn.set_theme(style='white')
plt.rcParams['lines.markersize'] = 14

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 14))

# FOA
df = pd.DataFrame({"Number of Rooms": ["18", "20", "23", "26", "29"],
                   "ER": [0.71, 0.66, 0.66, 0.67, 0.67],
                   "F": [0.20, 0.26, 0.26, 0.24, 0.26],
                   "LE": [47, 48.03, 40.60, 40.80, 40.43],
                   "LR": [0.41, 0.40, 0.37, 0.38, 0.39],
                   })
# set up localization error
ax1 = sn.lineplot(ax=axes[0], x="Number of Rooms", y="LE", data=df, color='red', marker='o', label='LE')
ax1.set_ylabel('Degrees')
ax1.set_xlabel('Number of Rooms in the Training Split')
ax1.yaxis.label.set_color('red')
ax1.yaxis.label.set_fontsize(28)
ax1.xaxis.label.set_fontsize(28)
ax1.tick_params(axis='y', colors='red', labelsize=28)
ax1.tick_params(axis='x', labelsize=20)
# ax1.legend(fontsize=28)

# set up error rate and recall
ax2 = sn.lineplot(x="Number of Rooms", y="ER", data=df, color='blue', marker='^', ax=ax1.twinx(), label='ER')
ax2.set_ylabel('')
ax2.yaxis.label.set_color('blue')
ax2.yaxis.label.set_fontsize(28)
ax2.grid(False)
ax2.tick_params(axis='y', colors='blue', labelsize=28)

ax3 = sn.lineplot(x="Number of Rooms", y="F", data=df, color='blue', marker='s', ax=ax2, label='F')
ax3.grid(False)
ax3.set_ylabel('')
ax4 = sn.lineplot(x="Number of Rooms", y="LR", data=df, color='blue', marker='D', ax=ax2, label='LR')
ax4.grid(False)
ax4.set_ylabel('')

ax1.get_legend().remove()
ax2.get_legend().remove()

# MIC
df = pd.DataFrame({"Number of Rooms": ["18", "20", "23", "26", "29"],
                   "ER": [0.71, 0.66, 0.66, 0.67, 0.67],
                   "F": [0.20, 0.26, 0.26, 0.24, 0.26],
                   "LE": [47, 48.03, 40.60, 40.80, 40.43],
                   "LR": [0.41, 0.40, 0.37, 0.38, 0.39],
                   })
# set up localization error
ax1 = sn.lineplot(ax=axes[1], x="Number of Rooms", y="LE", data=df, color='red', marker='o', label='LE'
                  )
ax1.set_ylabel('Degrees')
ax1.set_xlabel('Number of Rooms in the Training Split')
ax1.yaxis.label.set_color('red')
ax1.yaxis.label.set_fontsize(28)
ax1.xaxis.label.set_fontsize(28)
ax1.tick_params(axis='y', colors='red', labelsize=28)
ax1.tick_params(axis='x', labelsize=28)
# plt.legend(fontsize=24)

# set up error rate and recall
ax2 = sn.lineplot(x="Number of Rooms", y="ER", data=df, color='blue', marker='^', ax=ax1.twinx(), label='ER')
ax2.set_ylabel('')
ax2.yaxis.label.set_color('blue')
ax2.yaxis.label.set_fontsize(28)
ax2.grid(False)
ax2.tick_params(axis='y', colors='blue', labelsize=28)

ax3 = sn.lineplot(x="Number of Rooms", y="F", data=df, color='blue', marker='s', ax=ax2, label='F')
ax3.grid(False)
ax3.set_ylabel('')
ax4 = sn.lineplot(x="Number of Rooms", y="LR", data=df, color='blue', marker='D', ax=ax2, label='LR')
ax4.grid(False)
ax4.set_ylabel('')

# ax1.legend(fontsize=28, loc='right')
sn.move_legend(
    ax1, "lower right",
    bbox_to_anchor=(.3, 1), ncol=1, title=None, frameon=True,
    fontsize=24,
)
sn.move_legend(
    ax2, "lower left",
    bbox_to_anchor=(.3, 1), ncol=3, title=None, frameon=True,
    fontsize=24,
)


plt.savefig('roomplot.pdf', dpi=500, bbox_inches="tight")
plt.show()
