"""
calculator_mcp_server.py — calculator.py, exposed as an MCP server
========================================================================
A learning exercise: wraps the SAME calculate_transfer_amortization()
and parse_money_string_to_eur() functions already in calculator.py, but
via the Model Context Protocol -- so Claude Desktop (or any other
MCP-compatible client) can call them directly in a normal chat, without
going through engine.py/SynFC's own pipeline at all.

This is NOT wired into SynFC's own graph -- it's a separate, standalone
way to reach the exact same underlying functions. SynFC itself still
calls calculator.py directly (plain Python import), which is the right
choice for a single closed system -- this MCP server exists purely so
you can see MCP tool-calling working live, using code you already
understand.

Setup:
    pip install mcp
    Then point Claude Desktop's config at this file (see bottom of this
    docstring, or ask Claude to explain the config file location for
    your OS).

Claude Desktop config entry (in claude_desktop_config.json):
    {
      "mcpServers": {
        "synfc-calculator": {
          "command": "python",
          "args": ["/full/path/to/calculator_mcp_server.py"]
        }
      }
    }
Restart Claude Desktop after saving the config -- you should then see
these two tools available in a new chat.
"""

from mcp.server.mcpserver import MCPServer

from calculator import calculate_transfer_amortization, parse_money_string_to_eur

mcp = MCPServer("synfc-calculator")


@mcp.tool()
def transfer_amortization(
    transfer_fee_eur: float,
    contract_years: int,
    annual_wage_eur: float = 0.0,
    signing_bonus_eur: float = 0.0,
) -> dict:
    """Computes the standard football-finance amortization breakdown for
    signing a player: transfer fee spread evenly over the contract
    length, plus wages and any signing bonus, to get the real total
    annual cost.

    Args:
        transfer_fee_eur: the transfer/bonservis fee, in EUR.
        contract_years: length of the new contract, in years.
        annual_wage_eur: the player's annual wage, in EUR (0 if unknown).
        signing_bonus_eur: any one-off signing bonus, in EUR (0 if none).
    """
    return calculate_transfer_amortization(
        transfer_fee_eur, contract_years, annual_wage_eur, signing_bonus_eur
    )


@mcp.tool()
def money_string_to_eur(text: str) -> dict:
    """Converts a Transfermarkt-style money string (e.g. "€25.00m",
    "€500k") into a plain float in EUR.

    Args:
        text: the money string to parse, e.g. "€25.00m".
    """
    value = parse_money_string_to_eur(text)
    return {"input": text, "parsed_eur": value}


if __name__ == "__main__":
    mcp.run()
