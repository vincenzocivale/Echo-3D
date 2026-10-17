import os, pytest
from usmotion4d.data import tusrec, split

pytestmark = None
data_available = os.path.isdir(tusrec.ROOT)


@pytest.fixture(scope="session")
def scan():
    if not data_available: pytest.skip("TUS-REC2024 not available")
    return tusrec.Scan(tusrec.scan_paths(["000"])[0])
