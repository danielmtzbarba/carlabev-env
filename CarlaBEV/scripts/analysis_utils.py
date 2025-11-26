import matplotlib.pyplot as plt
import seaborn as sns

def plot_grouped_metric(df_eval, metric="success_rate", group_by="traffic"):
    """
    Bar plot: mean(metric) grouped by group_by.
    E.g. group_by="reward" → compare CARL vs Shaping.
    """
    plt.figure(figsize=(8,5))
    sns.barplot(
        data=df_eval,
        x=group_by,
        y=metric,
        estimator="mean",
        ci="sd",
        palette="viridis"
    )
    plt.title(f"{metric.replace('_',' ').title()} by {group_by.title()}")
    plt.ylabel(metric.replace('_',' ').title())
    plt.xlabel(group_by.title())
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def heatmap_metric(df_eval, metric="success_rate"):
    """
    Creates a 2D heatmap (Reward × Curriculum) for the selected metric.
    """
    pivot = df_eval.pivot_table(
        values=metric,
        index="reward",
        columns="curriculum",
        aggfunc="mean"
    )

    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma")
    plt.title(f"{metric.replace('_',' ').title()} — Reward × Curriculum")
    plt.xlabel("Curriculum")
    plt.ylabel("Reward Type")
    plt.show()

def plot_training_curve(df_train, tag_filter="rollout/ep_rew_mean", group_by="reward"):
    """
    Plots training curves for all experiments grouped by a condition.
    """
    plt.figure(figsize=(12,6))

    groups = df_train[group_by].unique()
    for g in groups:
        sub = df_train[(df_train[group_by] == g) & (df_train["tag"] == tag_filter)]
        sub = sub.sort_values("global_step")
        plt.plot(sub["global_step"], sub["value"], label=f"{group_by}={g}")

    plt.title(f"Training Curve ({tag_filter}) grouped by {group_by}")
    plt.xlabel("Global Step")
    plt.ylabel(tag_filter.split('/')[-1])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def scatter_success_collision(df_eval):
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df_eval,
        x="collision_rate",
        y="success_rate",
        hue="reward",
        style="input",
        size="traffic",
        sizes=(50,150)
    )
    plt.title("Success vs Collision (Pareto View)")
    plt.xlabel("Collision Rate")
    plt.ylabel("Success Rate")
    plt.grid(alpha=0.3)
    plt.show()

def boxplot_metric(df_eval, metric="success_rate", by="input"):
    plt.figure(figsize=(8,5))
    sns.boxplot(
        data=df_eval,
        x=by,
        y=metric,
        palette="Set3"
    )
    plt.title(f"{metric.replace('_',' ').title()} by {by.title()}")
    plt.show()

def compare_rewards(df_eval, metric="success_rate"):
    plt.figure(figsize=(7,5))
    sns.barplot(
        data=df_eval,
        x="reward",
        y=metric,
        palette={"shaping": "tab:blue", "carl": "tab:red"},
        ci="sd"
    )
    plt.title(f"{metric.replace('_',' ').title()} — Shaping vs CaRL")
    plt.ylabel(metric.replace('_',' ').title())
    plt.show()

def ranking_table(df_eval, top_k=10, metric="success_rate"):
    """
    Returns a sorted table of experiments by metric.
    """
    return df_eval.sort_values(metric, ascending=False).head(top_k)[[
        "exp_id", "traffic", "input", "reward", "curriculum",
        metric, "collision_rate", "mean_return"
    ]]