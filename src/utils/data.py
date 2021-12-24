import gzip
import hashlib
import shutil


def unzip_gz(filepath: str) -> None:
    # Open file with gzip encoding
    with gzip.open(filepath, 'rb') as f_in:
        # Uncompress the file and remove .gz ending
        with open(filepath[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def get_checksum(filepath: str) -> str:
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filepath, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()
