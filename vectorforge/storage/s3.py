"""S3 storage backend.

Stores document content in an S3-compatible object store via httpx.
Uses httpx instead of boto3 to avoid an extra heavy dependency,
falling back to AWS Signature V4 signing.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import UTC, datetime
from urllib.parse import quote

import httpx

from vectorforge.exceptions import StorageError
from vectorforge.storage.base import BaseStorageBackend

logger = logging.getLogger(__name__)


def _sign_v4(
    method: str,
    url: str,
    headers: dict[str, str],
    payload_hash: str,
    region: str,
    access_key: str,
    secret_key: str,
    service: str = "s3",
) -> dict[str, str]:
    """Create AWS Signature V4 headers for an S3 request.

    Args:
        method: HTTP method (PUT, GET, DELETE).
        url: Full request URL.
        headers: Existing request headers.
        payload_hash: SHA-256 hex digest of the request body.
        region: AWS region.
        access_key: AWS access key ID.
        secret_key: AWS secret access key.
        service: AWS service name.

    Returns:
        Updated headers dict with Authorization and related headers.
    """
    now = datetime.now(UTC)
    datestamp = now.strftime("%Y%m%d")
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")

    parsed = httpx.URL(url)
    host = parsed.host
    path = str(parsed.raw_path, "ascii") if isinstance(parsed.raw_path, bytes) else parsed.raw_path

    headers = {**headers}
    headers["x-amz-date"] = amz_date
    headers["x-amz-content-sha256"] = payload_hash
    headers["host"] = str(host)

    signed_header_keys = sorted(headers.keys())
    signed_headers = ";".join(signed_header_keys)
    canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in signed_header_keys)

    canonical_request = (
        f"{method}\n{path}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )

    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n"
        + hashlib.sha256(canonical_request.encode()).hexdigest()
    )

    def _hmac_sha256(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    signing_key = _hmac_sha256(
        _hmac_sha256(
            _hmac_sha256(
                _hmac_sha256(f"AWS4{secret_key}".encode(), datestamp),
                region,
            ),
            service,
        ),
        "aws4_request",
    )

    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()

    headers["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return headers


class S3StorageBackend(BaseStorageBackend):
    """Store document content in S3-compatible object storage.

    Uses httpx with AWS Signature V4 for authentication.
    No boto3 dependency required.

    Args:
        bucket: S3 bucket name.
        region: AWS region.
        access_key: AWS access key ID.
        secret_key: AWS secret access key.
        endpoint_url: Optional custom endpoint for S3-compatible stores.
        prefix: Optional key prefix for all objects.
    """

    def __init__(
        self,
        bucket: str,
        region: str,
        access_key: str,
        secret_key: str,
        endpoint_url: str = "",
        prefix: str = "vectorforge/documents/",
    ) -> None:
        if not bucket:
            msg = "S3 bucket name is required"
            raise StorageError(msg)
        self._bucket = bucket
        self._region = region
        self._access_key = access_key
        self._secret_key = secret_key
        self._endpoint_url = endpoint_url.rstrip("/") if endpoint_url else ""
        self._prefix = prefix
        self._client = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def _build_url(self, key: str) -> str:
        """Build the full S3 URL for a key.

        Args:
            key: The object key.

        Returns:
            Full URL string.
        """
        encoded_key = quote(key, safe="/")
        if self._endpoint_url:
            return f"{self._endpoint_url}/{self._bucket}/{encoded_key}"
        return f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{encoded_key}"

    def _full_key(self, key: str) -> str:
        """Prepend the configured prefix to a key.

        Args:
            key: The bare storage key.

        Returns:
            The prefixed key.
        """
        return f"{self._prefix}{key}"

    async def store(self, key: str, content: bytes) -> str:
        """Upload content to S3.

        Args:
            key: The storage key.
            content: The raw content bytes.

        Returns:
            The full S3 key.

        Raises:
            StorageError: If the upload fails.
        """
        full_key = self._full_key(key)
        url = self._build_url(full_key)
        payload_hash = hashlib.sha256(content).hexdigest()

        headers = _sign_v4(
            method="PUT",
            url=url,
            headers={"content-type": "application/octet-stream"},
            payload_hash=payload_hash,
            region=self._region,
            access_key=self._access_key,
            secret_key=self._secret_key,
        )

        response = await self._client.put(url, content=content, headers=headers)

        if response.status_code not in (200, 201):
            msg = f"S3 PUT failed (status={response.status_code}): {response.text}"
            raise StorageError(msg)

        logger.info("Stored %d bytes in S3 (key=%s)", len(content), full_key)
        return full_key

    async def retrieve(self, key: str) -> bytes:
        """Download content from S3.

        Args:
            key: The storage key (with or without prefix).

        Returns:
            The raw content bytes.

        Raises:
            StorageError: If the download fails.
        """
        full_key = key if key.startswith(self._prefix) else self._full_key(key)
        url = self._build_url(full_key)
        payload_hash = hashlib.sha256(b"").hexdigest()

        headers = _sign_v4(
            method="GET",
            url=url,
            headers={},
            payload_hash=payload_hash,
            region=self._region,
            access_key=self._access_key,
            secret_key=self._secret_key,
        )

        response = await self._client.get(url, headers=headers)

        if response.status_code != 200:
            msg = f"S3 GET failed (status={response.status_code}): {response.text}"
            raise StorageError(msg)

        return response.content

    async def delete(self, key: str) -> None:
        """Delete an object from S3.

        Args:
            key: The storage key (with or without prefix).

        Raises:
            StorageError: If the delete fails.
        """
        full_key = key if key.startswith(self._prefix) else self._full_key(key)
        url = self._build_url(full_key)
        payload_hash = hashlib.sha256(b"").hexdigest()

        headers = _sign_v4(
            method="DELETE",
            url=url,
            headers={},
            payload_hash=payload_hash,
            region=self._region,
            access_key=self._access_key,
            secret_key=self._secret_key,
        )

        response = await self._client.delete(url, headers=headers)

        if response.status_code not in (200, 204):
            msg = f"S3 DELETE failed (status={response.status_code}): {response.text}"
            raise StorageError(msg)

        logger.debug("Deleted from S3 (key=%s)", full_key)
