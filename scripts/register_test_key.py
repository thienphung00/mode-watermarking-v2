"""
One-time script to register test_batch_001 with the Authority Service.

This script registers the existing watermark key test_batch_001 so that
/api/v1/demo/verify can verify test images generated outside the API.

This is a bootstrap step, not a new feature. It does not modify verification
logic or add demo-specific behavior.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from service.authority import WatermarkAuthorityService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Register test_batch_001 with the Authority Service."""
    key_id = "test_batch_001"
    
    # Instantiate Authority Service
    authority = WatermarkAuthorityService()
    
    # Check if key already exists (idempotent)
    # Try to get the key, but handle decryption errors gracefully
    # (keys encrypted with different encryption keys will fail to decrypt)
    try:
        existing_record = authority.db.get_watermark(key_id)
        if existing_record is not None:
            logger.info(f"Key {key_id} already exists in authority registry. Skipping registration.")
            return
    except Exception as e:
        # If decryption fails (e.g., encryption key mismatch), check if key exists in DB
        # If it exists but can't be decrypted, we'll overwrite it with a new registration
        if key_id in authority.db._db:
            logger.warning(
                f"Key {key_id} exists in database but cannot be decrypted "
                f"(encryption key mismatch). Will register new key."
            )
        # If key doesn't exist, continue with registration
    
    # Register key with default watermark policy configuration
    # This uses the same defaults that generation would normally use
    logger.info(f"Registering {key_id} with Authority Service...")
    policy = authority.create_watermark_policy(key_id=key_id)
    
    logger.info(
        f"Successfully registered {key_id}: "
        f"watermark_version={policy['watermark_version']}"
    )


if __name__ == "__main__":
    main()

