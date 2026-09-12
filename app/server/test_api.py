"""Smoke-test the Fashion Intelligence API through its public /api proxy.

No third-party Python packages are required. By default, the script targets
http://localhost:3000/api and uses the first catalogue sample as its test image.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TARGETS = ("articleType", "season", "gender", "usage")


class ApiTestError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str, timeout: float, headers: dict[str, str]):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers

    def request(self, method: str, path: str, *, query=None, body=None, headers=None):
        url = f"{self.base_url}/{path.lstrip('/')}"
        if query:
            url += "?" + urlencode(query)
        request = Request(
            url,
            data=body,
            headers={**self.headers, **(headers or {})},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read(), response.headers.get_content_type()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise ApiTestError(
                f"{method} {url} returned HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise ApiTestError(f"{method} {url} failed: {error.reason}") from error

    def json(self, method: str, path: str, **kwargs):
        content, content_type = self.request(method, path, **kwargs)
        if content_type != "application/json":
            raise ApiTestError(
                f"{method} {path} returned {content_type}, expected application/json"
            )
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            raise ApiTestError(f"{method} {path} returned invalid JSON") from error


def require(condition: object, message: str) -> None:
    if not condition:
        raise ApiTestError(message)


def multipart_image(name: str, content: bytes, content_type: str):
    boundary = f"fashion-api-test-{uuid.uuid4().hex}"
    safe_name = Path(name).name.replace('"', "_")
    body = b"".join(
        (
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            content,
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    return body, f"multipart/form-data; boundary={boundary}"


def run_check(name: str, callback):
    started = time.perf_counter()
    result = callback()
    print(
        f"PASS  {name:<27} {(time.perf_counter() - started) * 1000:8.0f} ms",
        flush=True,
    )
    return result


def test_api(client: ApiClient, image_path: Path | None) -> None:
    docs, docs_type = run_check("GET /docs", lambda: client.request("GET", "docs"))
    require(docs_type == "text/html", "/docs did not return HTML")
    require(
        "/api/openapi.json" in docs.decode(errors="replace"),
        "/docs does not reference the public OpenAPI schema URL",
    )

    schema = run_check(
        "GET /openapi.json", lambda: client.json("GET", "openapi.json")
    )
    expected_paths = {
        "/health",
        "/gallery/{item_id}/image",
        "/predict",
        "/search",
        "/analyse",
        "/catalogue/samples",
        "/catalogue",
        "/explain",
    }
    require(isinstance(schema, dict), "OpenAPI response is not an object")
    require(
        expected_paths <= set(schema.get("paths", {})),
        "OpenAPI schema is missing application endpoints",
    )

    health = run_check("GET /health", lambda: client.json("GET", "health"))
    require(health.get("status") == "ok", "Health status is not 'ok'")
    require(
        all(health.get("models", {}).values()),
        "One or more model artifacts are reported unavailable",
    )

    catalogue = run_check(
        "GET /catalogue",
        lambda: client.json("GET", "catalogue", query={"page_size": 3}),
    )
    require(isinstance(catalogue.get("items"), list), "Catalogue has no item list")
    require(catalogue.get("total", 0) > 0, "Catalogue is empty")

    samples = run_check(
        "GET /catalogue/samples",
        lambda: client.json("GET", "catalogue/samples"),
    )
    require(samples.get("items"), "The sample gallery is empty")
    sample_id = str(samples["items"][0].get("id", ""))
    require(sample_id.isdigit(), "The first sample has no numeric ID")

    gallery_image, gallery_type = run_check(
        "GET /gallery/{id}",
        lambda: client.request("GET", f"gallery/{sample_id}/image"),
    )
    require(gallery_type.startswith("image/"), "Gallery endpoint did not return an image")
    require(gallery_image, "Gallery endpoint returned an empty image")

    if image_path:
        require(image_path.is_file(), f"Image does not exist: {image_path}")
        image = image_path.read_bytes()
        image_name = image_path.name
        image_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    else:
        image, image_name, image_type = (
            gallery_image,
            f"catalogue-{sample_id}.jpg",
            gallery_type,
        )
    body, content_type = multipart_image(image_name, image, image_type)
    upload = {"body": body, "headers": {"Content-Type": content_type}}

    prediction = run_check(
        "POST /predict",
        lambda: client.json("POST", "predict", query={"top_k": 3}, **upload),
    )
    require(
        set(prediction.get("predictions", {})) == set(TARGETS),
        "Prediction response does not contain all four targets",
    )

    search = run_check(
        "POST /search",
        lambda: client.json("POST", "search", query={"top_k": 3}, **upload),
    )
    require(len(search.get("results", [])) == 3, "Search did not return three results")

    analysis = run_check(
        "POST /analyse",
        lambda: client.json(
            "POST",
            "analyse",
            query={"prediction_top_k": 3, "search_top_k": 3},
            **upload,
        ),
    )
    require(
        set(analysis.get("predictions", {})) == set(TARGETS),
        "Analysis response does not contain all four targets",
    )
    require(
        len(analysis.get("similar_items", [])) == 3,
        "Analysis did not return three similar items",
    )

    for target in TARGETS:
        explanation = run_check(
            f"POST /explain ({target})",
            lambda target=target: client.json(
                "POST", "explain", query={"target": target}, **upload
            ),
        )
        require(explanation.get("method") == "Grad-CAM", "Unexpected explanation method")
        require(
            str(explanation.get("heatmap", "")).startswith("data:image/png;base64,"),
            f"{target} explanation has no PNG heatmap",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://localhost:3000/api",
        help="Public API prefix (default: %(default)s)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional JPG/PNG/WebP input; otherwise a catalogue sample is used",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Timeout per request in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--cf-access-client-id",
        default=os.environ.get("CF_ACCESS_CLIENT_ID"),
        help="Cloudflare Access client ID (or CF_ACCESS_CLIENT_ID)",
    )
    parser.add_argument(
        "--cf-access-client-secret",
        default=os.environ.get("CF_ACCESS_CLIENT_SECRET"),
        help="Cloudflare Access secret (or CF_ACCESS_CLIENT_SECRET)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.cf_access_client_id) != bool(args.cf_access_client_secret):
        print("ERROR Both Cloudflare Access token values must be provided.", file=sys.stderr)
        return 2
    headers = {"User-Agent": "fashion-intelligence-api-test/1.0"}
    if args.cf_access_client_id:
        headers.update(
            {
                "CF-Access-Client-Id": args.cf_access_client_id,
                "CF-Access-Client-Secret": args.cf_access_client_secret,
            }
        )
    print(f"Testing {args.base_url.rstrip('/')}\n", flush=True)
    try:
        test_api(ApiClient(args.base_url, args.timeout, headers), args.image)
    except (ApiTestError, OSError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        return 1
    print("\nAll API endpoint checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
