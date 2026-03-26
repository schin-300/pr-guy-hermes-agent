"""Solana chain provider.

Uses solders for key management/signing and solana-py for RPC.
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
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.system_program import transfer, TransferParams
    from solders.transaction import Transaction
    from solders.message import Message
    from solders.hash import Hash as SolHash
    from solana.rpc.api import Client as SolanaClient
    from solana.rpc.commitment import Confirmed
    _SOLANA_AVAILABLE = True
except ImportError:
    _SOLANA_AVAILABLE = False


SOLANA_CHAINS = {
    "solana": ChainConfig(
        chain_id="solana",
        display_name="Solana Mainnet",
        symbol="SOL",
        decimals=9,
        rpc_url="https://api.mainnet-beta.solana.com",
        explorer_url="https://explorer.solana.com",
    ),
    "solana-devnet": ChainConfig(
        chain_id="solana-devnet",
        display_name="Solana Devnet (Testnet)",
        symbol="SOL",
        decimals=9,
        rpc_url="https://api.devnet.solana.com",
        explorer_url="https://explorer.solana.com",
        is_testnet=True,
    ),
}

_LAMPORTS_PER_SOL = 1_000_000_000


class SolanaProvider(ChainProvider):
    """Provider for Solana."""

    def __init__(self, config: ChainConfig, rpc_url_override: str = ""):
        if not _SOLANA_AVAILABLE:
            raise ImportError(
                "solders and solana are required for Solana wallet support. "
                "Install with: pip install 'hermes-agent[wallet-solana]'"
            )
        super().__init__(config)
        url = rpc_url_override or config.rpc_url
        self._client = SolanaClient(url)

    def get_balance(self, address: str) -> Balance:
        pubkey = Pubkey.from_string(address)
        resp = self._client.get_balance(pubkey, commitment=Confirmed)
        lamports = resp.value
        sol = Decimal(lamports) / Decimal(_LAMPORTS_PER_SOL)
        return Balance(
            chain=self.config.chain_id,
            address=address,
            balance=sol,
            balance_raw=lamports,
            symbol="SOL",
            decimals=9,
        )

    def send_transaction(
        self,
        from_private_key: str,
        to_address: str,
        amount: Decimal,
    ) -> TransactionResult:
        try:
            # Parse keypair — stored as hex-encoded 64-byte keypair (secret + public)
            key_bytes = bytes.fromhex(from_private_key)
            keypair = Keypair.from_bytes(key_bytes)
            to_pubkey = Pubkey.from_string(to_address)
            lamports = int(amount * Decimal(_LAMPORTS_PER_SOL))

            # Get recent blockhash — use Finalized for reliability on devnet
            from solana.rpc.commitment import Finalized
            blockhash_resp = self._client.get_latest_blockhash(commitment=Finalized)
            recent_blockhash = blockhash_resp.value.blockhash

            # Build transfer instruction
            ix = transfer(TransferParams(
                from_pubkey=keypair.pubkey(),
                to_pubkey=to_pubkey,
                lamports=lamports,
            ))

            # Build and sign transaction
            msg = Message.new_with_blockhash([ix], keypair.pubkey(), recent_blockhash)
            tx = Transaction.new_unsigned(msg)
            tx.sign([keypair], recent_blockhash)

            # Send
            resp = self._client.send_transaction(tx)
            tx_hash = str(resp.value)

            cluster_param = ""
            if self.config.is_testnet:
                cluster_param = "?cluster=devnet"

            logger.info("Solana transaction sent: %s", tx_hash)
            return TransactionResult(
                tx_hash=tx_hash,
                chain=self.config.chain_id,
                status="submitted",
                explorer_url=f"{self.config.explorer_url}/tx/{tx_hash}{cluster_param}",
            )
        except Exception as e:
            logger.error("Solana transaction failed: %s", e)
            return TransactionResult(
                tx_hash="",
                chain=self.config.chain_id,
                status="failed",
                error=str(e),
            )

    def estimate_fee(self, from_address: str, to_address: str, amount: Decimal) -> GasEstimate:
        # Solana has a flat base fee of 5000 lamports per signature
        # Priority fees are optional and variable
        fee_lamports = 5000
        fee_sol = Decimal(fee_lamports) / Decimal(_LAMPORTS_PER_SOL)
        return GasEstimate(
            chain=self.config.chain_id,
            estimated_fee=fee_sol,
            estimated_fee_raw=fee_lamports,
            symbol="SOL",
        )

    def validate_address(self, address: str) -> bool:
        try:
            Pubkey.from_string(address)
            return True
        except (ValueError, Exception):
            return False

    def generate_keypair(self) -> tuple[str, str]:
        """Generate a new Solana keypair. Returns (address, private_key_hex).

        Stores the full 64-byte keypair (secret + public) because
        solders.Keypair.from_bytes() requires it.
        """
        kp = Keypair()
        return str(kp.pubkey()), bytes(kp).hex()

    @staticmethod
    def address_from_key(private_key: str) -> str:
        """Derive address from a private key."""
        key_bytes = bytes.fromhex(private_key)
        kp = Keypair.from_bytes(key_bytes)
        return str(kp.pubkey())

    def explorer_tx_url(self, tx_hash: str) -> str:
        cluster_param = ""
        if self.config.is_testnet:
            cluster_param = "?cluster=devnet"
        return f"{self.config.explorer_url}/tx/{tx_hash}{cluster_param}"
