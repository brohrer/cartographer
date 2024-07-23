import matplotlib.pyplot as plt
import matplotlib.patches as patches

BG_COLOR = "#000000"  # black
BG_LINE_COLOR = "#AAAAAA"  # gray
LINE_COLOR = "#DDDDDD"  # light gray
FILL_COLOR = "#DDDDDD"  # light gray
NEG_LINE_COLOR = "#BB2222"  # red
NEG_FILL_COLOR = "#BB2222"  # red

LINEWIDTH = 2
BG_LINEWIDTH = 1

AX_REWARD_LEFT = 0.15
AX_REWARD_BOTTOM = 0.65
AX_REWARD_WIDTH = 0.7
AX_REWARD_HEIGHT = 0.25

REWARD_BAR_WIDTH = 0.6
AX_REWARD_YMIN = -1.1
AX_REWARD_YMAX = 1.1

AX_PREDS_LEFT = AX_REWARD_LEFT
AX_PREDS_BOTTOM = 0.15
AX_PREDS_WIDTH = AX_REWARD_WIDTH
AX_PREDS_HEIGHT = 0.45


class Visualization:
    def __init__(self):
        self.fig = plt.figure(figsize=(6, 6))
        self.ax_bg = self.add_blank_axes((0, 0, 1, 1))
        self.initialized = False

    def initialize(self, predictions, rewards, uncertainties):
        self.n_actions, self.n_features = predictions.shape

        self.ax_reward = self.add_blank_axes(
            (
                AX_REWARD_LEFT,
                AX_REWARD_BOTTOM,
                AX_REWARD_WIDTH,
                AX_REWARD_HEIGHT,
            )
        )
        self.ax_reward.set_xlim(-1, self.n_actions)
        self.ax_reward.set_ylim(AX_REWARD_YMIN, AX_REWARD_YMAX)

        self.ax_reward.plot(
            [-1, self.n_actions],
            [0, 0],
            color=BG_LINE_COLOR,
            linewidth=BG_LINEWIDTH,
        )

        self.reward_bars = []
        # Add a patch for each bar in the bar plot
        for i, reward in enumerate(rewards):
            left = i - REWARD_BAR_WIDTH / 2
            right = i + REWARD_BAR_WIDTH / 2
            path = [
                [left, 0],
                [right, 0],
                [right, reward],
                [left, reward],
            ]

            # Make negative rewards visually distinct from positive ones
            # to intuitively communicate that they are qualitatively different.
            if reward > 0:
                fill_color = FILL_COLOR
                line_color = LINE_COLOR
            else:
                fill_color = NEG_FILL_COLOR
                line_color = NEG_LINE_COLOR

            self.reward_bars.append(
                self.ax_reward.add_patch(
                    patches.Polygon(
                        path,
                        alpha=1 - uncertainties[i],
                        facecolor=fill_color,
                        edgecolor=line_color,
                        linewidth=LINEWIDTH,
                        joinstyle="round",
                    )
                )
            )

        self.ax_preds = self.add_blank_axes(
            (
                AX_PREDS_LEFT,
                AX_PREDS_BOTTOM,
                AX_PREDS_WIDTH,
                AX_PREDS_HEIGHT,
            )
        )
        self.ax_preds.set_xlim(-1, self.n_actions)
        self.ax_preds.set_ylim(-1, self.n_features)

        plt.ion()
        plt.show()
        self.initialized = True

    def update(self, predictions, rewards, uncertainties):
        if not self.initialized:
            self.initialize(predictions, rewards, uncertainties)

        for i, bar in enumerate(self.reward_bars):
            left = i - REWARD_BAR_WIDTH / 2
            right = i + REWARD_BAR_WIDTH / 2
            bar.set_xy(
                [
                    [left, 0],
                    [right, 0],
                    [right, rewards[i]],
                    [left, rewards[i]],
                ]
            )
            if rewards[i] > 0:
                fill_color = FILL_COLOR
                line_color = LINE_COLOR
            else:
                fill_color = NEG_FILL_COLOR
                line_color = NEG_LINE_COLOR
            bar.set_facecolor(fill_color)
            bar.set_edgecolor(line_color)
            bar.set_alpha(uncertainties[i])

        self.fig.canvas.flush_events()

    def add_blank_axes(self, dimensions):
        ax = self.fig.add_axes(dimensions)
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        return ax
