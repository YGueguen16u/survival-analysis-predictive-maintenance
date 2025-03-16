import numpy as np
import pandas as pd


def generate_survival_data_extended(n=300, seed=42):
    """
    Generates a synthetic dataset for survival analysis in an industrial context,
    including multiple covariates related to failure risk.

    Modifications:
      - Slight correlation between usage_rate and temp_avg
      - Clamped temp_avg range [45, 85]
      - Increased hazard if usage_rate > 0.75 AND temp_avg > 70

    Parameters
    n : int
        Number of machines (observations)
    seed : int
        Random seed for reproducibility

    Returns
    pandas.DataFrame
        Columns: machine_id, time, status, brand, usage_rate, temp_avg,
                 age_machine, env_humidity, maintenance_freq, operator_experience, shock_events
    """

    np.random.seed(seed)

    # --- 1) Core Covariates ---

    # Brand
    brand_choices = ["Alpha", "Beta", "Gamma"]
    brand = np.random.choice(brand_choices, size=n, p=[0.3, 0.4, 0.3])
    # Usage rate (usage_rate) [0.3, 0.9]
    usage_rate = np.random.uniform(low=0.3, high=0.9, size=n).round(2)
    # Raw temperature (around 65 ± 5) before adjustment
    temp_raw = np.random.normal(loc=65, scale=5, size=n)
    # Slight correlation with usage_rate => temp_avg += 2 * (usage_rate - 0.5)
    temp_corr = temp_raw + 2.0 * (usage_rate - 0.5)
    # Clamp on [45, 85]
    temp_clamped = np.clip(temp_corr, 45, 85)
    # Final rounding
    temp_avg = temp_clamped.round(1)
    # Machine age [1..10]
    age_machine = np.random.randint(low=1, high=11, size=n)

    # --- 2) Additional Covariates ---

    # 1) env_humidity: Environmental humidity (%)
    env_humidity = np.random.uniform(low=30, high=90, size=n).round(1)

    # 2) maintenance_freq: [0.5..2.0], adjusted based on brand
    maintenance_freq = 0.5 + 1.5 * np.random.rand(n)
    for i in range(n):
        if brand[i] == "Beta":
            maintenance_freq[i] *= 0.9  # -10%
        elif brand[i] == "Gamma":
            maintenance_freq[i] *= 1.1  # +10%
    maintenance_freq = maintenance_freq.round(2)

    # 3) operator_experience: [1..5], positively correlated with maintenance_freq
    operator_experience = []
    for i in range(n):
        base_exp = 3 + 0.5 * (maintenance_freq[i] - 1.25)
        exp_value = np.random.normal(loc=base_exp, scale=0.7)
        exp_value = min(max(exp_value, 1), 5)  # borne [1..5]
        operator_experience.append(round(exp_value))
    operator_experience = np.array(operator_experience)

    # 4) shock_events: Binomial(3, p), where p increases if usage_rate > 0.8 & env_humidity > 75
    shock_events = []
    for i in range(n):
        prob_shock = 0.1
        if usage_rate[i] > 0.8:
            prob_shock += 0.2
        if env_humidity[i] > 75:
            prob_shock += 0.15
        p = min(prob_shock, 0.8)
        k = np.random.binomial(3, p)
        shock_events.append(k)
    shock_events = np.array(shock_events)

    # --- 3) Hazard Rate Calculation ---

    # Baseline λ ~ 1/600
    lambda_baseline = 1 / 600

    # brand_factor
    brand_factor = []
    for b in brand:
        if b == "Alpha":
            brand_factor.append(1.0)
        elif b == "Beta":
            brand_factor.append(1.2)  # +20% risk
        else:  # Gamma
            brand_factor.append(0.9)  # -10% risk
    brand_factor = np.array(brand_factor)

    # usage_factor : pivot=0.5 => hazard *= (1 + usage_rate - 0.5)
    usage_factor = 1 + (usage_rate - 0.5)

    # temp_factor : >65 => +risk => hazard *= exp(0.02*(temp_avg-65))
    temp_factor = np.exp(0.02 * (temp_avg - 65))

    # age_factor : hazard *= (1 + 0.06*age_machine)
    age_factor = 1 + 0.06 * age_machine

    # humidity_factor : hazard *= (1 + 0.005*(env_humidity - 50))
    humidity_factor = 1 + 0.005 * (env_humidity - 50)

    # maintenance_factor : hazard *= exp(-0.3*(maintenance_freq - 1))
    maintenance_factor = np.exp(-0.3 * (maintenance_freq - 1))

    # op_exp_factor : hazard *= 0.95^(operator_experience-3)
    op_exp_factor = 0.95 ** (operator_experience - 3)

    # shock_factor : hazard *= (1 + 0.2 * shock_events)
    shock_factor = 1 + 0.2 * shock_events

    # Final product (without synergy)
    hazard_rate = (lambda_baseline * brand_factor * usage_factor * temp_factor *
                   age_factor * humidity_factor * maintenance_factor * op_exp_factor *
                   shock_factor)

    # --- Synergie usage & temp ---
    # If usage_rate > 0.75 AND temp_avg > 70 => +10% hazard
    # => Apply a local multiplier for these cases
    for i in range(n):
        if usage_rate[i] > 0.75 and temp_avg[i] > 70:
            hazard_rate[i] *= 1.1

    # --- 4) Failure Time: Exponential(hazard_rate) ---
    event_time = np.random.exponential(scale=1.0 / hazard_rate)

    # --- 5) Censoring: Uniform[300..1200] ---
    censor_time = np.random.uniform(low=300, high=1200, size=n)

    time = np.minimum(event_time, censor_time)
    status = (event_time <= censor_time).astype(int)

    # --- 6) DataFrame Assembly ---
    df = pd.DataFrame({
        "machine_id": np.arange(1, n + 1),
        "time": np.round(time, 1),
        "status": status,
        "brand": brand,
        "usage_rate": usage_rate,
        "temp_avg": temp_avg,
        "age_machine": age_machine,
        "env_humidity": env_humidity,
        "maintenance_freq": maintenance_freq,
        "operator_experience": operator_experience,
        "shock_events": shock_events
    })

    return df


if __name__ == "__main__":
    # Generate a dataset with 13,800 observations
    data = generate_survival_data_extended(n=13800, seed=123)
    print(data.head(12))

    csv_filename = "survival_industry_extended.csv"
    data.to_csv(csv_filename, index=False)
    print(f"Fichier CSV généré : {csv_filename}")
