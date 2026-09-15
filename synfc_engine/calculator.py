"""
calculator.py — SynFC financial calculator tools
=====================================================
Pure-math financial calculations, meant for the Finance Team (and any
future team that needs a reliable NUMBER, not an LLM's mental
arithmetic). Unlike tools.py, nothing here touches the network -- it's
just formulas, so results are exact and instant, with no risk of a
site changing its HTML and breaking everything.

This file is meant to grow into a small toolbox of different financial
calculations over time, not stay a single function -- see the "room to
grow" section at the bottom for where the next ones would go (FFP
squad-cost ratio, wage-to-revenue ratio, resale-value projection, etc).
Each new calculation should follow the same shape as the one below:
plain numbers in, a dict of results out, no side effects.
"""

from typing import Optional


def calculate_transfer_amortization(
    transfer_fee_eur: float,
    contract_years: int,
    annual_wage_eur: float = 0.0,
    signing_bonus_eur: float = 0.0,
) -> dict:
    """Computes the standard football-finance amortization breakdown for
    signing a player: the transfer fee is spread evenly over the
    contract length (the accounting convention clubs actually use for
    Financial Fair Play purposes), then wages and any signing bonus
    (also typically amortized over the contract) are added on top to
    get the real total annual cost.

    Args:
        transfer_fee_eur: the transfer/bonservis fee, in EUR.
        contract_years: length of the NEW contract being offered, in years.
        annual_wage_eur: the player's annual wage, in EUR (0 if unknown).
        signing_bonus_eur: any one-off signing bonus, in EUR (0 if none/unknown).

    Example return:
        {"transfer_fee_eur": 40000000, "contract_years": 5,
         "annual_amortization_eur": 8000000.0, "annual_wage_eur": 6000000.0,
         "signing_bonus_eur": 2000000, "amortized_bonus_per_year_eur": 400000.0,
         "total_annual_cost_eur": 14400000.0,
         "total_cost_over_contract_eur": 72000000.0}
    """
    if contract_years <= 0:
        return {"error": "contract_years must be a positive integer"}
    if transfer_fee_eur < 0 or annual_wage_eur < 0 or signing_bonus_eur < 0:
        return {"error": "monetary inputs must not be negative"}

    annual_amortization = transfer_fee_eur / contract_years
    amortized_bonus_per_year = signing_bonus_eur / contract_years
    total_annual_cost = annual_amortization + annual_wage_eur + amortized_bonus_per_year
    total_cost_over_contract = total_annual_cost * contract_years

    return {
        "transfer_fee_eur": transfer_fee_eur,
        "contract_years": contract_years,
        "annual_amortization_eur": round(annual_amortization, 2),
        "annual_wage_eur": annual_wage_eur,
        "signing_bonus_eur": signing_bonus_eur,
        "amortized_bonus_per_year_eur": round(amortized_bonus_per_year, 2),
        "total_annual_cost_eur": round(total_annual_cost, 2),
        "total_cost_over_contract_eur": round(total_cost_over_contract, 2),
    }


def parse_money_string_to_eur(text: str) -> Optional[float]:
    """Converts a Transfermarkt-style money string (e.g. "€25.00m",
    "€500k", "€1.2bn") into a plain float in EUR. Returns None for
    unparseable input (e.g. "-", "unknown", "nan") -- callers should
    treat None as "no usable figure", not as zero.

    Example: "€25.00m" -> 25000000.0 ; "€500k" -> 500000.0
    """
    if not text:
        return None
    cleaned = text.strip().lower().replace("€", "").replace(",", "").strip()
    if cleaned in ("", "-", "unknown", "nan", "n/a"):
        return None

    multiplier = 1.0
    if cleaned.endswith("bn"):
        multiplier = 1_000_000_000
        cleaned = cleaned[:-2]
    elif cleaned.endswith("m"):
        multiplier = 1_000_000
        cleaned = cleaned[:-1]
    elif cleaned.endswith("k"):
        multiplier = 1_000
        cleaned = cleaned[:-1]

    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Room to grow: add more pure-math financial calculations here as
# separate functions, following the same "plain numbers in, dict out,
# no network calls" shape, e.g.:
#   - calculate_ffp_squad_cost_ratio(total_wage_bill, total_revenue)
#   - calculate_wage_to_revenue_ratio(...)
#   - project_resale_value(purchase_price, age, contract_years_left)
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # Quick manual test:
    #   python calculator.py 40000000 5 6000000 2000000
    import sys

    args = [float(a) for a in sys.argv[1:]]
    if len(args) < 2:
        args = [40_000_000, 5, 6_000_000, 2_000_000]
    result = calculate_transfer_amortization(*args)
    for k, v in result.items():
        print(f"{k}: {v}")