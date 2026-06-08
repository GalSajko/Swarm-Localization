#!/usr/bin/env python3
import os

from plot_social_fusion_rule_comparison import PLOTS_DIR, plot_social_fusion_rule_comparison


if __name__ == "__main__":
    plot_social_fusion_rule_comparison(
        terrains = ("city", "mountains", "rural"),
        output_stem = os.path.join(PLOTS_DIR, "social_fusion_rule_without_desert_comparison"),
    )
