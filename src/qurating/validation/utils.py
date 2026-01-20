from pathlib import Path
from urllib.parse import urlparse
from tempfile import TemporaryDirectory

from cloudpathlib import AzureBlobClient, CloudPath
import pymupdf


def download_resource(
    url: str,
    resource_folder: Path,
    client: AzureBlobClient,
    skip_existing: bool = True,
    scheme: str = "az",
) -> Path:
    """Download a resource from Azure Blob Storage.

    Args:
        url: Full URL to the resource
        resource_folder: Local folder to download to
        client: Azure Blob Client instance
        skip_existing: Skip download if file already exists

    Returns:
        Path to the downloaded file
    """
    purl = urlparse(url)
    full_path = purl.path.lstrip("/")
    cloud_path = CloudPath(f"{scheme}://{full_path}", client=client)
    resource_path = resource_folder / cloud_path.name
    if not (skip_existing and resource_path.exists()):
        cloud_path.download_to(resource_path)
    return resource_path


def process_resource(
    url,
    client=None,
    scheme=None,
    verbose=1,
    dl=True,
    client_kwargs=None,
    extract_images=False,
):
    if client_kwargs is None:
        client_kwargs = {}
    purl = urlparse(url)
    full_path = purl.path.lstrip("/")
    if scheme is None:
        scheme = purl.scheme
    if client is None:
        if scheme == "az":
            client = AzureBlobClient(**client_kwargs)
        else:
            raise NotImplementedError(
                "Auto client currently only supported for 'az' scheme"
            )
    cloud_path = CloudPath(f"{scheme}://{full_path}", client=client)
    sz = cloud_path.stat().st_size * (1e-6 * (1 / 8))
    if verbose > 0:
        print(sz)

    if dl:
        with TemporaryDirectory() as tempdir:
            local_path = download_resource(
                url, Path(tempdir), client, skip_existing=True
            )
            if verbose > 0:
                print(f"  -> Saved to: {local_path}")

            try:
                # with open(local_path, "rb") as f:
                #     pdf_bytes = f.read()
                # print(full_path)
                # print(sz)
                with pymupdf.open(local_path) as f:
                    pages_text = [p.get_text() for p in f.pages()]
                    if extract_images:
                        pages_images = [
                            [b for b in p.get_text("dict")["blocks"] if b["type"] == 1]
                            for p in f.pages()
                        ]
                    else:
                        pages_images = []
            except Exception as e:
                print(e)
                # pdf_bytes = b""
                pages_text = []
                pages_images = []
    else:
        # pdf_bytes = b""
        pages_text = []
        pages_images = []

    return pages_text, pages_images, sz
