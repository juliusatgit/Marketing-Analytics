import pandas as pd
import numpy as np
from scipy.stats import entropy

def revenue_trend(group, end):
    """
    Revenue trend: difference between second and first half of rental history
    Positive = growing customer, Negative = declining customer
    """
    # transaction-based trend 
    # def revenue_trend(group):
    #     group = group.sort_values("rentalPeriod.start")
    #     mid = len(group) // 2
    #     if mid == 0:
    #         return 0.0
    #     first_half  = group.iloc[:mid]["revenue"].sum()
    #     second_half = group.iloc[mid:]["revenue"].sum()
    #     return second_half - first_half

    # time-based trend
    group = group.sort_values("rentalPeriod.start")
    midpoint = end - (end - group["rentalPeriod.start"].min()) / 2

    first_half = group[group["rentalPeriod.start"] < midpoint]["revenue"].sum()
    second_half = group[group["rentalPeriod.start"] >= midpoint]["revenue"].sum()

    return second_half - first_half


def get_labels(df, start, end):
    """
    Computes actual revenue per customer in the label period [start, end).
    Customers with no activity in this period get 0.
    Returns a Series indexed by customer.id.
    """
    label_df = df[
        (df["rentalPeriod.start"] >= start) &
        (df["rentalPeriod.start"] <  end)
    ]
    labels = label_df.groupby("customer.id")["revenue"].sum()
    return labels


def build_features(df, end, start=None):
    """Build all customer features using only data before cutoff."""
    if start is not None:
        df = df[(df["rentalPeriod.start"] >= start) & (df["rentalPeriod.start"] < end)].copy()
    else:
        df = df[df["rentalPeriod.start"] < end].copy()

    df["month"] = df["rentalPeriod.start"].dt.month

    df["time_weight"] = np.exp(
        -(end - df["rentalPeriod.start"]).dt.days / 365
    )

    df["weighted_freq"] = df["time_weight"]

    weighted_freq = df.groupby("customer.id")["weighted_freq"].sum().reset_index()
    weighted_freq.columns = ["customer.id", "weighted_frequency"]

    # RFM
    rfm = df.groupby("customer.id").agg(
        recency        = ("rentalPeriod.start", lambda x: (end - x.max()).days),
        frequency      = ("rentalPeriod.start", "count"),
        monetary       = ("revenue", "sum"),
        avg_revenue    = ("revenue", "mean"),
        std_revenue    = ("revenue", "std"),
        avg_duration   = ("duration_days", "mean"),
        weighted_rev   = ("revenue", lambda x: (x * df.loc[x.index, "time_weight"]).sum()),
        first_rental   = ("rentalPeriod.start", "min"),
        last_rental    = ("rentalPeriod.start", "max"),
    ).reset_index()

    rfm["recency_x_monetary"] = rfm["recency"] * rfm["monetary"]
    rfm["recency_x_frequency"] = rfm["recency"] * rfm["frequency"]
    rfm["monetary_x_frequency"] = rfm["monetary"] * rfm["frequency"]

    rfm["tenure_days"]              = (rfm["last_rental"] - rfm["first_rental"]).dt.days
    rfm["avg_days_between_rentals"] = rfm["tenure_days"] / rfm["frequency"].clip(lower=1)
    rfm["recency_ratio"]            = rfm["recency"] / rfm["avg_days_between_rentals"].clip(lower=1)
    rfm["std_revenue"]              = rfm["std_revenue"].fillna(0)
    rfm = rfm.drop(columns=["first_rental", "last_rental"])

    rfm["revenue_per_day"] = rfm["monetary"] / rfm["tenure_days"].clip(lower=1)
    rfm["rentals_per_day"] = rfm["frequency"] / rfm["tenure_days"].clip(lower=1)

    # Seasonality
    seasonality = df.groupby("customer.id").agg(
        n_spring      = ("month", lambda x: x.isin([3, 4, 5]).sum()),
        n_summer      = ("month", lambda x: x.isin([6, 7, 8]).sum()),
        n_autumn      = ("month", lambda x: x.isin([9, 10, 11]).sum()),
        n_winter      = ("month", lambda x: x.isin([12, 1, 2]).sum()),
        active_months = ("month", lambda x: df.loc[x.index, "rentalPeriod.start"].dt.to_period("M").nunique()),
    ).reset_index()

    # Trend
    trend = df.groupby("customer.id").apply(lambda x: revenue_trend(x, end)).reset_index()
    trend.columns = ["customer.id", "revenue_trend"]

    # Weekly rentals
    pct_weekly = df.groupby("customer.id").apply(
        lambda x: (x["duration_days"] <= 7).mean()
    ).reset_index()
    pct_weekly.columns = ["customer.id", "pct_weekly_rentals"]

    # Cluster affinity
    cluster_affinity = (
        df.groupby(["customer.id", "cluster"])["revenue"]
        .sum()
        .unstack(fill_value=0)
    )
    cluster_affinity = cluster_affinity.div(cluster_affinity.sum(axis=1), axis=0)
    cluster_affinity.columns = [f"cluster_affinity_{int(c)}" for c in cluster_affinity.columns]
    cluster_affinity = cluster_affinity.reset_index()

    affinity_cols = [c for c in cluster_affinity.columns if c.startswith("cluster_affinity_")]
    cluster_affinity["n_clusters_rented"] = (cluster_affinity[affinity_cols] > 0).sum(axis=1)
    cluster_affinity["style_entropy"] = cluster_affinity[affinity_cols].apply(
        lambda row: entropy(row + 1e-9), axis=1
    )

    # Price features
    price_features = df.groupby("customer.id").agg(
        avg_retail_price   = ("retailPrice", "mean"),
        max_retail_price   = ("retailPrice", "max"),
        avg_price_per_week = ("pricePerWeek", "mean"),
    ).reset_index()

    features = rfm \
        .merge(seasonality,      on="customer.id", how="left") \
        .merge(trend,            on="customer.id", how="left") \
        .merge(pct_weekly,       on="customer.id", how="left") \
        .merge(cluster_affinity, on="customer.id", how="left") \
        .merge(price_features,   on="customer.id", how="left") \
        .merge(weighted_freq,    on="customer.id", how="left")

    return features


def generate_train_test_splits(df, train_cutoff, label_end, final_cutoff, timeframe_start = None):
    """
    Orchestrates the feature building and label generation for both
    the training fold and the test fold based on the provided cutoffs.
    
    Returns:
    X_train_f, X_test_f, y_train, y_test, X_train, X_test
    """
    # Training Fold
    X_train = build_features(df, start=timeframe_start, end=train_cutoff)
    y_raw_train = get_labels(df, start=train_cutoff, end=label_end)
    
    y_train = pd.DataFrame(
        X_train[["customer.id"]]
        .merge(y_raw_train.reset_index(), on="customer.id", how="left")
        ["revenue"].fillna(0).values,
        columns=["target_revenue"]
    )
    id_train = X_train[["customer.id"]]
    X_train_f = X_train.drop(columns=["customer.id"])

    # Test Fold
    X_test = build_features(df, start=timeframe_start, end=label_end)
    y_raw_test = get_labels(df, start=label_end, end=final_cutoff)
    
    y_test = pd.DataFrame(
        X_test[["customer.id"]]
        .merge(y_raw_test.reset_index(), on="customer.id", how="left")
        ["revenue"].fillna(0).values,
        columns=["target_revenue"]
    )
    id_test = X_test[["customer.id"]]
    X_test_f = X_test.drop(columns=["customer.id"])

    return X_train_f, X_test_f, y_train, y_test, id_train, id_test