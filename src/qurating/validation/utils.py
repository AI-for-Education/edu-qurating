from pathlib import Path
from urllib.parse import urlparse
from tempfile import TemporaryDirectory
import zipfile

from cloudpathlib import AzureBlobClient, GSClient, CloudPath
from cloudpathlib.client import Client
import pymupdf


def download_resource(
    url: str,
    resource_folder: Path,
    client: Client,
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
    i=0,
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
        elif scheme == "gs":
            client = GSClient(**client_kwargs)
        else:
            raise NotImplementedError(
                "Auto client currently only supported for 'az' or 'gs' scheme"
            )
    cloud_path = CloudPath(f"{scheme}://{full_path}", client=client)
    try:
        sz = cloud_path.stat().st_size * (1e-6 * (1 / 8))
    except Exception:
        sz = None
    if verbose > 0:
        print(sz)

    if dl:
        with TemporaryDirectory() as tempdir:
            local_path = download_resource(
                url, Path(tempdir), client, skip_existing=True, scheme=scheme
            )
            if verbose > 0:
                print(f"  -> Saved to: {local_path}")

            try:
                # with open(local_path, "rb") as f:
                #     pdf_bytes = f.read()
                # print(i)
                # print(full_path)
                # print(sz)
                filetype = full_path.split(".")[-1]
                # print(filetype)
                if filetype == "zip":
                    pages_text = {}
                    with zipfile.ZipFile(local_path, mode="r") as f:
                        for file in f.namelist():
                            subfiletype = file.split(".")[-1]
                            if subfiletype in ["pdf", "docx", "txt"]:
                                filebytes = f.read(file)
                                with pymupdf.open(None, stream=filebytes, filetype=subfiletype) as pdf:
                                    pages_text[file] = [p.get_text() for p in pdf.pages()]
                    pages_images = []
                #############
                elif filetype in ["pdf", "docx", "txt"]:
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
