"""hermes-wallet — crypto wallet for Hermes Agent.

Built on top of hermes-keystore.  Private keys are stored as sealed
secrets — the agent never has direct access.  Transactions go through
a policy engine and optional owner approval.

Scope (v1):
    - Wallet creation, import, listing
    - Native token transfers (ETH, SOL)
    - Balance checks
    - Transaction history (local log)
    - Policy engine (spending limits, rate limits, approval thresholds)
    - CLI + gateway approval flow for high-value transactions

Out of scope (v1):
    - Smart contracts / DeFi / swaps
    - ERC-20 / SPL token transfers
    - Hardware wallets
    - Multi-sig
"""
