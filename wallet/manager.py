"""Wallet manager — creates, stores, and operates wallets.

Wallets are stored in the keystore as sealed secrets.  The manager
is the ONLY code that reads private keys; it passes them to chain
providers for signing and never exposes them outside this module.

Wallet metadata is stored alongside the key in the keystore:
    wallet:meta:<wallet_id>   → JSON metadata (label, chain, type, address)
    wallet:<chain>:<address>  → encrypted private key (sealed)
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from keystore.client import KeystoreClient
from wallet.chains import ChainProvider, Balance, TransactionResult, GasEstimate

logger = logging.getLogger(__name__)


@dataclass
class WalletInfo:
    """Public wallet information (no private keys)."""
    wallet_id: str
    label: str
    chain: str
    address: str
    wallet_type: str        # "user" | "agent"
    created_at: str


@dataclass
class TxRecord:
    """Local transaction log entry."""
    tx_id: str
    wallet_id: str
    chain: str
    to_address: str
    amount: str             # Decimal as string for precision
    symbol: str
    tx_hash: str
    status: str             # "submitted" | "confirmed" | "failed" | "rejected"
    policy_result: str      # JSON string
    requested_at: str
    decided_by: str         # "policy_auto" | "owner_cli" | "owner_gateway"


class WalletError(Exception):
    """Base wallet exception."""


class WalletNotFound(WalletError):
    """Wallet ID doesn't exist."""


class WalletManager:
    """Manages wallet lifecycle and transaction execution.

    All private key access goes through the keystore with requester="wallet".
    The agent process never calls this directly — it goes through the
    wallet tools which use session tokens.
    """

    def __init__(self, keystore: KeystoreClient):
        self._ks = keystore
        self._providers: Dict[str, ChainProvider] = {}
        self._tx_log: List[TxRecord] = []  # In-memory for now, persisted to keystore

    # ------------------------------------------------------------------
    # Chain provider management
    # ------------------------------------------------------------------

    def register_provider(self, chain_id: str, provider: ChainProvider) -> None:
        """Register a chain provider."""
        self._providers[chain_id] = provider

    def get_provider(self, chain_id: str) -> ChainProvider:
        """Get the provider for a chain. Raises WalletError if not registered."""
        if chain_id not in self._providers:
            raise WalletError(
                f"No provider registered for chain '{chain_id}'. "
                f"Available: {', '.join(self._providers.keys()) or 'none'}"
            )
        return self._providers[chain_id]

    @property
    def supported_chains(self) -> List[str]:
        return list(self._providers.keys())

    # ------------------------------------------------------------------
    # Wallet CRUD
    # ------------------------------------------------------------------

    def create_wallet(
        self,
        chain: str,
        label: str = "",
        wallet_type: str = "user",
    ) -> WalletInfo:
        """Create a new wallet (generates a fresh keypair).

        Returns public wallet info. The private key is stored as a sealed
        secret in the keystore and never returned.
        """
        provider = self.get_provider(chain)

        # Generate keypair
        if hasattr(provider, "generate_keypair"):
            address, private_key = provider.generate_keypair()
        else:
            raise WalletError(f"Chain '{chain}' doesn't support key generation")

        wallet_id = f"w_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        if not label:
            label = f"{provider.config.display_name} Wallet"

        # Store private key as sealed secret
        key_name = f"wallet:{chain}:{address}"
        self._ks.set_secret(key_name, private_key, category="sealed",
                           description=f"Private key for {label}")

        # Store metadata
        meta = {
            "wallet_id": wallet_id,
            "label": label,
            "chain": chain,
            "address": address,
            "wallet_type": wallet_type,
            "created_at": now,
        }
        meta_name = f"wallet:meta:{wallet_id}"
        self._ks.set_secret(meta_name, json.dumps(meta), category="sealed",
                           description=f"Metadata for {label}")

        logger.info("Created %s wallet '%s' on %s: %s", wallet_type, label, chain, address)

        return WalletInfo(
            wallet_id=wallet_id,
            label=label,
            chain=chain,
            address=address,
            wallet_type=wallet_type,
            created_at=now,
        )

    def import_wallet(
        self,
        chain: str,
        private_key: str,
        label: str = "",
        wallet_type: str = "user",
    ) -> WalletInfo:
        """Import an existing wallet from a private key."""
        provider = self.get_provider(chain)

        # Derive address from key
        if hasattr(provider, "address_from_key"):
            address = provider.address_from_key(private_key)
        else:
            raise WalletError(f"Chain '{chain}' doesn't support key import")

        wallet_id = f"w_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        if not label:
            label = f"Imported {provider.config.display_name} Wallet"

        # Store private key
        key_name = f"wallet:{chain}:{address}"
        self._ks.set_secret(key_name, private_key, category="sealed",
                           description=f"Private key for {label}")

        # Store metadata
        meta = {
            "wallet_id": wallet_id,
            "label": label,
            "chain": chain,
            "address": address,
            "wallet_type": wallet_type,
            "created_at": now,
        }
        meta_name = f"wallet:meta:{wallet_id}"
        self._ks.set_secret(meta_name, json.dumps(meta), category="sealed",
                           description=f"Metadata for {label}")

        logger.info("Imported wallet '%s' on %s: %s", label, chain, address)

        return WalletInfo(
            wallet_id=wallet_id,
            label=label,
            chain=chain,
            address=address,
            wallet_type=wallet_type,
            created_at=now,
        )

    def list_wallets(self) -> List[WalletInfo]:
        """List all wallets (public info only)."""
        wallets = []
        for secret in self._ks.list_secrets():
            if secret.name.startswith("wallet:meta:"):
                meta_json = self._ks.get_secret(secret.name, requester="wallet")
                if meta_json:
                    try:
                        meta = json.loads(meta_json)
                        wallets.append(WalletInfo(**meta))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("Corrupt wallet metadata: %s: %s", secret.name, e)
        return wallets

    def get_wallet(self, wallet_id: str) -> WalletInfo:
        """Get wallet info by ID."""
        meta_json = self._ks.get_secret(f"wallet:meta:{wallet_id}", requester="wallet")
        if not meta_json:
            raise WalletNotFound(f"Wallet '{wallet_id}' not found")
        meta = json.loads(meta_json)
        return WalletInfo(**meta)

    def delete_wallet(self, wallet_id: str) -> bool:
        """Delete a wallet and its private key."""
        try:
            wallet = self.get_wallet(wallet_id)
        except WalletNotFound:
            return False

        # Delete key and metadata
        key_name = f"wallet:{wallet.chain}:{wallet.address}"
        self._ks.delete_secret(key_name)
        self._ks.delete_secret(f"wallet:meta:{wallet_id}")
        logger.info("Deleted wallet '%s' (%s)", wallet.label, wallet.address)
        return True

    # ------------------------------------------------------------------
    # Balance & Transactions
    # ------------------------------------------------------------------

    def get_balance(self, wallet_id: str) -> Balance:
        """Get the balance for a wallet."""
        wallet = self.get_wallet(wallet_id)
        provider = self.get_provider(wallet.chain)
        return provider.get_balance(wallet.address)

    def estimate_fee(self, wallet_id: str, to_address: str, amount: Decimal) -> GasEstimate:
        """Estimate transaction fee."""
        wallet = self.get_wallet(wallet_id)
        provider = self.get_provider(wallet.chain)
        return provider.estimate_fee(wallet.address, to_address, amount)

    def send(
        self,
        wallet_id: str,
        to_address: str,
        amount: Decimal,
        decided_by: str = "owner_cli",
    ) -> TransactionResult:
        """Execute a native token transfer.

        This is the ONLY method that reads the private key from the keystore.
        It should only be called after policy evaluation and approval.
        """
        wallet = self.get_wallet(wallet_id)
        provider = self.get_provider(wallet.chain)

        # Validate recipient
        if not provider.validate_address(to_address):
            return TransactionResult(
                tx_hash="", chain=wallet.chain, status="failed",
                error=f"Invalid address for {wallet.chain}: {to_address}",
            )

        # Read private key (sealed — only wallet requester can access)
        key_name = f"wallet:{wallet.chain}:{wallet.address}"
        private_key = self._ks.get_secret(key_name, requester="wallet")
        if not private_key:
            return TransactionResult(
                tx_hash="", chain=wallet.chain, status="failed",
                error="Failed to retrieve private key from keystore",
            )

        # Execute transaction
        result = provider.send_transaction(private_key, to_address, amount)

        # Log transaction
        tx_record = TxRecord(
            tx_id=f"tx_{uuid.uuid4().hex[:12]}",
            wallet_id=wallet_id,
            chain=wallet.chain,
            to_address=to_address,
            amount=str(amount),
            symbol=provider.config.symbol,
            tx_hash=result.tx_hash,
            status=result.status,
            policy_result="{}",
            requested_at=datetime.now(timezone.utc).isoformat(),
            decided_by=decided_by,
        )
        self._tx_log.append(tx_record)

        return result

    def get_tx_history(self, wallet_id: Optional[str] = None, limit: int = 20) -> List[TxRecord]:
        """Get transaction history from local log."""
        records = self._tx_log
        if wallet_id:
            records = [r for r in records if r.wallet_id == wallet_id]
        return records[-limit:]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_wallet(self, wallet_id: Optional[str] = None, chain: Optional[str] = None) -> WalletInfo:
        """Resolve a wallet — by ID, by chain (first match), or the only wallet."""
        if wallet_id:
            return self.get_wallet(wallet_id)

        wallets = self.list_wallets()
        if not wallets:
            raise WalletNotFound("No wallets found. Create one with 'hermes wallet create'")

        if chain:
            matching = [w for w in wallets if w.chain == chain]
            if not matching:
                raise WalletNotFound(f"No wallet found for chain '{chain}'")
            return matching[0]

        if len(wallets) == 1:
            return wallets[0]

        raise WalletError(
            f"Multiple wallets found. Specify --wallet-id or --chain. "
            f"Wallets: {', '.join(f'{w.label} ({w.chain})' for w in wallets)}"
        )
