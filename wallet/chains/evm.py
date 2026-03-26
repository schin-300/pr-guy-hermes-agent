"""EVM chain provider — Ethereum, Base, Polygon, Arbitrum, etc.

Uses eth-account for key management/signing and web3.py for RPC.
All EVM chains share this single provider with different ChainConfig.
"""

import logging
from decimal import Decimal
from typing import Optional

from wallet.chains import (
    Balance,
    ChainConfig,
    ChainProvider,
    GasEstimate,
    TransactionResult,
)

logger = logging.getLogger(__name__)

try:
    from eth_account import Account
    from web3 import Web3
    _WEB3_AVAILABLE = True
except ImportError:
    _WEB3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pre-built chain configs
# ---------------------------------------------------------------------------

EVM_CHAINS = {
    "ethereum": ChainConfig(
        chain_id="ethereum",
        display_name="Ethereum Mainnet",
        symbol="ETH",
        decimals=18,
        rpc_url="https://eth.llamarpc.com",
        explorer_url="https://etherscan.io",
    ),
    "ethereum-sepolia": ChainConfig(
        chain_id="ethereum-sepolia",
        display_name="Ethereum Sepolia (Testnet)",
        symbol="ETH",
        decimals=18,
        rpc_url="https://rpc.sepolia.org",
        explorer_url="https://sepolia.etherscan.io",
        is_testnet=True,
    ),
    "base": ChainConfig(
        chain_id="base",
        display_name="Base",
        symbol="ETH",
        decimals=18,
        rpc_url="https://mainnet.base.org",
        explorer_url="https://basescan.org",
    ),
    "base-sepolia": ChainConfig(
        chain_id="base-sepolia",
        display_name="Base Sepolia (Testnet)",
        symbol="ETH",
        decimals=18,
        rpc_url="https://sepolia.base.org",
        explorer_url="https://sepolia.basescan.org",
        is_testnet=True,
    ),
    "polygon": ChainConfig(
        chain_id="polygon",
        display_name="Polygon",
        symbol="POL",
        decimals=18,
        rpc_url="https://polygon-rpc.com",
        explorer_url="https://polygonscan.com",
    ),
    "arbitrum": ChainConfig(
        chain_id="arbitrum",
        display_name="Arbitrum One",
        symbol="ETH",
        decimals=18,
        rpc_url="https://arb1.arbitrum.io/rpc",
        explorer_url="https://arbiscan.io",
    ),
    "optimism": ChainConfig(
        chain_id="optimism",
        display_name="Optimism",
        symbol="ETH",
        decimals=18,
        rpc_url="https://mainnet.optimism.io",
        explorer_url="https://optimistic.etherscan.io",
    ),
}


# EVM chain IDs (for transaction signing)
_CHAIN_IDS = {
    "ethereum": 1,
    "ethereum-sepolia": 11155111,
    "base": 8453,
    "base-sepolia": 84532,
    "polygon": 137,
    "arbitrum": 42161,
    "optimism": 10,
}


class EVMProvider(ChainProvider):
    """Provider for all EVM-compatible chains."""

    def __init__(self, config: ChainConfig, rpc_url_override: str = ""):
        if not _WEB3_AVAILABLE:
            raise ImportError(
                "web3 and eth-account are required for EVM wallet support. "
                "Install with: pip install 'hermes-agent[wallet]'"
            )
        super().__init__(config)
        url = rpc_url_override or config.rpc_url
        self._w3 = Web3(Web3.HTTPProvider(url))
        self._evm_chain_id = _CHAIN_IDS.get(config.chain_id)

    def get_balance(self, address: str) -> Balance:
        checksum = Web3.to_checksum_address(address)
        balance_wei = self._w3.eth.get_balance(checksum)
        balance_eth = Decimal(balance_wei) / Decimal(10 ** self.config.decimals)
        return Balance(
            chain=self.config.chain_id,
            address=address,
            balance=balance_eth,
            balance_raw=balance_wei,
            symbol=self.config.symbol,
            decimals=self.config.decimals,
        )

    def send_transaction(
        self,
        from_private_key: str,
        to_address: str,
        amount: Decimal,
    ) -> TransactionResult:
        account = Account.from_key(from_private_key)
        to_checksum = Web3.to_checksum_address(to_address)
        amount_wei = int(amount * Decimal(10 ** self.config.decimals))

        try:
            nonce = self._w3.eth.get_transaction_count(account.address)

            # Build transaction
            tx = {
                "to": to_checksum,
                "value": amount_wei,
                "nonce": nonce,
                "chainId": self._evm_chain_id,
            }

            # Use EIP-1559 if supported, otherwise legacy
            try:
                latest = self._w3.eth.get_block("latest")
                if hasattr(latest, "baseFeePerGas") and latest.baseFeePerGas is not None:
                    # EIP-1559
                    max_priority = self._w3.eth.max_priority_fee
                    base_fee = latest.baseFeePerGas
                    tx["maxFeePerGas"] = base_fee * 2 + max_priority
                    tx["maxPriorityFeePerGas"] = max_priority
                else:
                    tx["gasPrice"] = self._w3.eth.gas_price
            except Exception:
                tx["gasPrice"] = self._w3.eth.gas_price

            # Estimate gas
            tx["gas"] = self._w3.eth.estimate_gas(tx)

            # Sign and send
            signed = self._w3.eth.account.sign_transaction(tx, from_private_key)
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            logger.info("Transaction sent: %s on %s", tx_hash_hex, self.config.chain_id)

            return TransactionResult(
                tx_hash=tx_hash_hex,
                chain=self.config.chain_id,
                status="submitted",
                explorer_url=self.explorer_tx_url(tx_hash_hex),
            )
        except Exception as e:
            logger.error("Transaction failed on %s: %s", self.config.chain_id, e)
            return TransactionResult(
                tx_hash="",
                chain=self.config.chain_id,
                status="failed",
                error=str(e),
            )

    def estimate_fee(self, from_address: str, to_address: str, amount: Decimal) -> GasEstimate:
        from_checksum = Web3.to_checksum_address(from_address)
        to_checksum = Web3.to_checksum_address(to_address)
        amount_wei = int(amount * Decimal(10 ** self.config.decimals))

        try:
            gas_limit = self._w3.eth.estimate_gas({
                "from": from_checksum,
                "to": to_checksum,
                "value": amount_wei,
            })
            gas_price = self._w3.eth.gas_price
            fee_wei = gas_limit * gas_price
            fee_eth = Decimal(fee_wei) / Decimal(10 ** self.config.decimals)

            return GasEstimate(
                chain=self.config.chain_id,
                estimated_fee=fee_eth,
                estimated_fee_raw=fee_wei,
                symbol=self.config.symbol,
            )
        except Exception as e:
            # Return a rough estimate on failure
            rough_fee = Decimal("0.0005")  # ~21000 gas * ~24 gwei
            return GasEstimate(
                chain=self.config.chain_id,
                estimated_fee=rough_fee,
                estimated_fee_raw=int(rough_fee * Decimal(10 ** 18)),
                symbol=self.config.symbol,
            )

    def validate_address(self, address: str) -> bool:
        try:
            Web3.to_checksum_address(address)
            return True
        except (ValueError, Exception):
            return False

    def generate_keypair(self) -> tuple[str, str]:
        """Generate a new EVM keypair. Returns (address, private_key_hex)."""
        account = Account.create()
        return account.address, account.key.hex()

    @staticmethod
    def address_from_key(private_key: str) -> str:
        """Derive address from a private key."""
        account = Account.from_key(private_key)
        return account.address
