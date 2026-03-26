"""Abstract chain provider interface.

Each blockchain (EVM, Solana, etc.) implements this interface.
The wallet manager dispatches through it without knowing chain specifics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional


@dataclass
class Balance:
    """Native token balance for a wallet."""
    chain: str
    address: str
    balance: Decimal          # In native units (ETH, SOL)
    balance_raw: int          # In smallest unit (wei, lamports)
    symbol: str               # "ETH", "SOL"
    decimals: int             # 18 for ETH, 9 for SOL


@dataclass
class TransactionResult:
    """Result of a submitted transaction."""
    tx_hash: str
    chain: str
    status: str               # "submitted" | "confirmed" | "failed"
    explorer_url: str = ""
    gas_used: Optional[int] = None
    error: Optional[str] = None


@dataclass
class GasEstimate:
    """Gas/fee estimate for a transaction."""
    chain: str
    estimated_fee: Decimal    # In native units
    estimated_fee_raw: int    # In smallest unit
    symbol: str


@dataclass
class ChainConfig:
    """Configuration for a blockchain."""
    chain_id: str             # "ethereum", "base", "solana", etc.
    display_name: str
    symbol: str               # Native token symbol
    decimals: int
    rpc_url: str
    explorer_url: str         # Base URL for tx explorer
    is_testnet: bool = False


class ChainProvider(ABC):
    """Abstract interface for blockchain interaction."""

    def __init__(self, config: ChainConfig):
        self.config = config

    @property
    def chain_id(self) -> str:
        return self.config.chain_id

    @abstractmethod
    def get_balance(self, address: str) -> Balance:
        """Get native token balance for an address."""
        ...

    @abstractmethod
    def send_transaction(
        self,
        from_private_key: str,
        to_address: str,
        amount: Decimal,
    ) -> TransactionResult:
        """Sign and broadcast a native token transfer.

        Args:
            from_private_key: Hex-encoded private key (provided by daemon, never by agent)
            to_address: Recipient address
            amount: Amount in native units (ETH, SOL)

        Returns:
            TransactionResult with tx hash and status
        """
        ...

    @abstractmethod
    def estimate_fee(self, from_address: str, to_address: str, amount: Decimal) -> GasEstimate:
        """Estimate transaction fee."""
        ...

    @abstractmethod
    def validate_address(self, address: str) -> bool:
        """Check if an address is valid for this chain."""
        ...

    def explorer_tx_url(self, tx_hash: str) -> str:
        """Return the block explorer URL for a transaction."""
        return f"{self.config.explorer_url}/tx/{tx_hash}"
