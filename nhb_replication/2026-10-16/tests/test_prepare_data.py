"""Archive-boundary and integrity tests for the public replication installer."""
from pathlib import Path
import hashlib
import json
import stat
import sys
import tempfile
import unittest
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_data import canonical_path, member_allowed, restore_archive, sha256_file


class ArchiveRestorationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "checkout"
        self.root.mkdir()
        self.archive = self.base / "agent_study_arrays.zip"
        self.member = "data/results/c0000.npz"
        self.contents = b"fixture payload; the loader does not execute its contents"

    def tearDown(self):
        self.temp.cleanup()

    def create(self, members=None):
        members = members if members is not None else [(self.member, self.contents)]
        with zipfile.ZipFile(self.archive, "w") as packed:
            for name, data in members:
                packed.writestr(name, data)
        records = [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in members]
        asset = {"name": self.archive.name, "bytes": self.archive.stat().st_size, "sha256": sha256_file(self.archive)}
        return asset, records

    def test_valid_payload_and_idempotence(self):
        asset, records = self.create()
        self.assertEqual(restore_archive(self.archive, asset, records, self.root), 1)
        self.assertEqual((self.root / self.member).read_bytes(), self.contents)
        self.assertEqual(restore_archive(self.archive, asset, records, self.root), 0)

    def test_corrupt_archive_rejected_before_extraction(self):
        asset, records = self.create()
        self.archive.write_bytes(self.archive.read_bytes() + b"unexpected data")
        with self.assertRaisesRegex(ValueError, "byte-size mismatch"):
            restore_archive(self.archive, asset, records, self.root)
        self.assertFalse((self.root / self.member).exists())

    def test_wrong_member_hash_does_not_replace_existing_file(self):
        asset, records = self.create()
        target = self.root / self.member
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing verified snapshot")
        records[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Member SHA-256 mismatch"):
            restore_archive(self.archive, asset, records, self.root)
        self.assertEqual(target.read_bytes(), b"existing verified snapshot")
        self.assertEqual(list(target.parent.glob("*.part")), [])

    def test_unlisted_extra_member_rejected(self):
        asset, records = self.create([(self.member, self.contents), ("paper.pdf", b"excluded")])
        with self.assertRaisesRegex(ValueError, "Duplicate or missing"):
            restore_archive(self.archive, asset, records[:1], self.root)
        self.assertFalse((self.root / self.member).exists())

    def test_traversal_and_wrong_payload_class_rejected(self):
        for name in ["../paper.pdf", "/absolute.npz", "data/results/../paper.npz", "data\\results\\c0000.npz", "data/results/.hidden.npz", "C:/paper.npz", "data//results/c0000.npz"]:
            with self.subTest(name=name):
                self.assertFalse(member_allowed(self.archive.name, name))
                with self.assertRaises(ValueError):
                    canonical_path(name)
        self.assertFalse(member_allowed(self.archive.name, "article/results.tex"))
        self.assertFalse(member_allowed("replication_metadata.zip", self.member))
        self.assertFalse(member_allowed("source_data_workbooks.zip", "NHB_Source_Data/paper.pdf"))
        self.assertTrue(member_allowed("replication_metadata.zip", "baseline/nhb_smallworld_20261012/results/f0008.json"))

    def test_symlinked_destination_rejected(self):
        asset, records = self.create()
        outside = self.base / "elsewhere"
        outside.mkdir()
        (self.root / "data").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            restore_archive(self.archive, asset, records, self.root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_symlink_archive_member_rejected(self):
        info = zipfile.ZipInfo(self.member)
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive, "w") as packed:
            packed.writestr(info, self.contents)
        asset = {"name": self.archive.name, "bytes": self.archive.stat().st_size, "sha256": sha256_file(self.archive)}
        records = [{"path": self.member, "bytes": len(self.contents), "sha256": hashlib.sha256(self.contents).hexdigest()}]
        with self.assertRaisesRegex(ValueError, "Disallowed archive member"):
            restore_archive(self.archive, asset, records, self.root)

    def test_duplicate_archive_members_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            asset, records = self.create([(self.member, self.contents), (self.member, self.contents)])
        with self.assertRaisesRegex(ValueError, "Duplicate or missing"):
            restore_archive(self.archive, asset, records[:1], self.root)


if __name__ == "__main__":
    unittest.main()
