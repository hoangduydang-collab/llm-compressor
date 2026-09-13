import csv
import hashlib
import io
import stat
import zipfile
from pathlib import Path
from types import MappingProxyType

import pytest

from pipeline import aa_lcr_v11 as A


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    archive = path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, contents in members.items():
            zf.writestr(name, contents)
    return archive


def make_duplicate_zip(path: Path, name: str) -> Path:
    archive = path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(name, b"first")
            zf.writestr(name, b"second")
    return archive


def make_unflagged_utf8_zip(path: Path, members: dict[str, bytes]) -> Path:
    """Create a fixture whose UTF-8 names incorrectly lack ZIP bit 11."""
    archive = make_zip(path, members)
    contents = bytearray(archive.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = 0
        while (offset := contents.find(signature, start)) != -1:
            flags = int.from_bytes(
                contents[offset + flag_offset : offset + flag_offset + 2]
            )
            contents[offset + flag_offset : offset + flag_offset + 2] = (
                flags & ~0x800
            ).to_bytes(2, "little")
            start = offset + len(signature)
    archive.write_bytes(contents)
    return archive


def make_special_zip(path: Path, name: str, mode: int) -> Path:
    archive = path / "special.zip"
    member = zipfile.ZipInfo(name)
    member.external_attr = mode << 16
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, b"x")
    return archive


def make_nul_filename_zip(path: Path) -> Path:
    archive = make_zip(path, {"nulXmember.txt": b"x"})
    archive.write_bytes(
        archive.read_bytes().replace(b"nulXmember.txt", b"nul\x00member.txt")
    )
    return archive


def question(question_id: int) -> A.Question:
    return A.Question(
        question_id=question_id,
        category="category",
        document_set_id="set-1",
        question="Question?",
        official_answer="Answer",
        document_filenames=("document.txt",),
        prompt="prompt",
        cl100k_tokens=1,
    )


def fixture_payloads(
    *,
    include_official_extra: bool = True,
    include_trailing_question: bool = False,
    include_trailing_answer: bool = False,
) -> dict[str, bytes]:
    documents = {"b.txt": b"B", "a.txt": b"A"}
    rows = []
    for question_id in range(1, 101):
        filenames = "b.txt;a.txt" if question_id == 1 else "a.txt"
        question_text = f"Question {question_id}?"
        if include_trailing_question and question_id == 1:
            question_text += " \u200b "
        answer = f"Answer {question_id}"
        if include_trailing_answer and question_id == 1:
            answer += "  "
        prompt = A.build_candidate_prompt(
            [documents[filename].decode("utf-8") for filename in filenames.split(";")],
            question_text,
        )
        rows.append(
            {
                "": str(question_id - 1),
                "document_category": "synthetic",
                "document_set_id": f"set-{(question_id - 1) % 30 + 1}",
                "question_id": question_id,
                "question": question_text,
                "answer": answer,
                "data_source_filenames": filenames,
                "data_source_urls": "",
                "input_tokens": len(
                    A.tiktoken.get_encoding("cl100k_base").encode(prompt)
                ),
            }
        )
    csv_file = io.StringIO(newline="")
    writer = csv.DictWriter(csv_file, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        for set_id in range(1, 31):
            zf.writestr(f"lcr/synthetic/set-{set_id}/a.txt", documents["a.txt"])
        zf.writestr("lcr/synthetic/set-1/b.txt", documents["b.txt"])
        if include_official_extra:
            zf.writestr(A.OFFICIAL_UNREFERENCED_MEMBER, b"official extra")
    return {
        A.CSV_FILENAME: csv_file.getvalue().encode("utf-8"),
        A.ZIP_FILENAME: archive.getvalue(),
    }


class FixtureResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class LocalCl100kEncoding:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


def prepare_synthetic_100_question_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_official_extra=True,
    include_trailing_question=False,
    include_trailing_answer=False,
) -> A.PreparedDataset:
    encoding = LocalCl100kEncoding()
    monkeypatch.setattr(A.tiktoken, "get_encoding", lambda name: encoding)
    monkeypatch.setattr(A, "PINNED_INPUT_TOKEN_DISCREPANCIES", MappingProxyType({}))
    payloads = fixture_payloads(
        include_official_extra=include_official_extra,
        include_trailing_question=include_trailing_question,
        include_trailing_answer=include_trailing_answer,
    )
    monkeypatch.setattr(
        A,
        "ZIP_SHA256",
        hashlib.sha256(payloads[A.ZIP_FILENAME]).hexdigest(),
    )

    def opener(url: str):
        filename = url.rsplit("/", 1)[-1]
        payload = next(
            contents
            for path, contents in payloads.items()
            if path.rsplit("/", 1)[-1] == filename
        )
        return FixtureResponse(payload)

    return A.prepare_dataset(tmp_path / "prepared", opener=opener)


def test_safe_extract_rejects_parent_traversal(tmp_path):
    archive = make_zip(tmp_path, {"../escape.txt": b"x"})
    with pytest.raises(A.DatasetIntegrityError, match="unsafe archive member"):
        A.safe_extract_zip(archive, tmp_path / "out")


@pytest.mark.parametrize(
    "member",
    [
        "/absolute.txt",
        "C:/drive-qualified.txt",
        "//server/share.txt",
        r"folder\..\escape.txt",
    ],
)
def test_safe_extract_rejects_cross_platform_unsafe_paths(tmp_path, member):
    archive = make_zip(tmp_path, {member: b"x"})
    with pytest.raises(A.DatasetIntegrityError, match="unsafe archive member"):
        A.safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_rejects_nul_filename(tmp_path):
    archive = make_nul_filename_zip(tmp_path)
    with pytest.raises(A.DatasetIntegrityError, match="unsafe archive member"):
        A.safe_extract_zip(archive, tmp_path / "out")


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o644])
def test_safe_extract_rejects_non_regular_unix_members(tmp_path, mode):
    archive = make_special_zip(tmp_path, "lcr/category/set/document.txt", mode)
    with pytest.raises(A.DatasetIntegrityError, match="unsafe archive member"):
        A.safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_rejects_duplicate_members(tmp_path):
    archive = make_duplicate_zip(tmp_path, "lcr/A/x.txt")
    with pytest.raises(A.DatasetIntegrityError, match="duplicate archive member"):
        A.safe_extract_zip(archive, tmp_path / "out")


def make_ambiguous_cp437_zip(path: Path) -> tuple[Path, str, str]:
    placeholder_name = "lcr/category/set/nameXX.txt"
    raw_suffix = b"\xc2\xa2"
    cp437_name = f"lcr/category/set/name{raw_suffix.decode('cp437')}.txt"
    utf8_name = "lcr/category/set/name¢.txt"
    archive = make_zip(path, {placeholder_name: b"document"})
    archive.write_bytes(
        archive.read_bytes().replace(b"nameXX.txt", b"name" + raw_suffix + b".txt")
    )
    return archive, cp437_name, utf8_name


def test_safe_extract_preserves_valid_unflagged_cp437_name_without_contract(tmp_path):
    archive, cp437_name, _ = make_ambiguous_cp437_zip(tmp_path)

    extracted = A.safe_extract_zip(archive, tmp_path / "out")

    assert [path.relative_to(tmp_path / "out").as_posix() for path in extracted] == [
        cp437_name
    ]


def test_safe_extract_prefers_expected_valid_unflagged_cp437_name(tmp_path):
    archive, cp437_name, _ = make_ambiguous_cp437_zip(tmp_path)

    extracted = A.safe_extract_zip(
        archive, tmp_path / "out", expected_members={cp437_name}
    )

    assert [path.relative_to(tmp_path / "out").as_posix() for path in extracted] == [
        cp437_name
    ]


def test_safe_extract_recovers_unflagged_utf8_name_and_normalizes_nfc(tmp_path):
    raw_name = "lcr/Marketing/mkt_gaming/402813954_17. 260-275 Sinem Bas\u0327ev.txt"
    expected_name = "lcr/Marketing/mkt_gaming/402813954_17. 260-275 Sinem Başev.txt"
    archive = make_unflagged_utf8_zip(tmp_path, {raw_name: b"document"})

    extracted = A.safe_extract_zip(
        archive, tmp_path / "out", expected_members={expected_name}
    )

    assert [path.relative_to(tmp_path / "out").as_posix() for path in extracted] == [
        expected_name
    ]


def test_safe_extract_rejects_canonical_name_collision(tmp_path):
    expected_name = "lcr/Marketing/mkt_gaming/Sinem Başev.txt"
    archive = make_unflagged_utf8_zip(
        tmp_path,
        {
            "lcr/Marketing/mkt_gaming/Sinem Başev.txt": b"first",
            "lcr/Marketing/mkt_gaming/Sinem Bas\u0327ev.txt": b"second",
        },
    )

    with pytest.raises(A.DatasetIntegrityError, match="duplicate archive member"):
        A.safe_extract_zip(
            archive, tmp_path / "out", expected_members={expected_name}
        )


def test_safe_extract_rejects_member_outside_expected_contract(tmp_path):
    archive = make_zip(
        tmp_path,
        {
            "lcr/category/set/expected.txt": b"expected",
            "lcr/category/set/unexpected.txt": b"unexpected",
        },
    )
    with pytest.raises(A.DatasetIntegrityError, match="unexpected archive member"):
        A.safe_extract_zip(
            archive,
            tmp_path / "out",
            expected_members={"lcr/category/set/expected.txt"},
        )


def test_prepare_dataset_allows_only_pinned_official_unreferenced_member(
    tmp_path, monkeypatch
):
    prepared = prepare_synthetic_100_question_fixture(
        tmp_path, monkeypatch, include_official_extra=True
    )

    assert A.OFFICIAL_UNREFERENCED_MEMBER in prepared.file_sha256


def test_prepare_fixture_with_official_headers_loads_questions(tmp_path, monkeypatch):
    prepared = prepare_synthetic_100_question_fixture(tmp_path, monkeypatch)

    first = prepared.questions[0]
    assert first.category == "synthetic"
    assert first.official_answer == "Answer 1"


def test_prepare_fixture_preserves_raw_question_trailing_whitespace(
    tmp_path, monkeypatch
):
    prepared = prepare_synthetic_100_question_fixture(
        tmp_path, monkeypatch, include_trailing_question=True
    )

    assert prepared.questions[0].question.endswith(" \u200b ")
    assert (
        "START QUESTION\n\nQuestion 1? \u200b \n\nEND QUESTION"
        in prepared.questions[0].prompt
    )


def test_prepare_fixture_preserves_raw_official_answer_trailing_whitespace(
    tmp_path, monkeypatch
):
    prepared = prepare_synthetic_100_question_fixture(
        tmp_path, monkeypatch, include_trailing_answer=True
    )

    assert prepared.questions[0].official_answer == "Answer 1  "


def test_pinned_input_token_discrepancies_are_accepted_exactly():
    A._validate_input_token_discrepancies(A.PINNED_INPUT_TOKEN_DISCREPANCIES)


def test_unexpected_input_token_discrepancy_is_rejected():
    observed = dict(A.PINNED_INPUT_TOKEN_DISCREPANCIES)
    observed[1] = (1, 2)

    with pytest.raises(A.DatasetIntegrityError, match="unexpected"):
        A._validate_input_token_discrepancies(observed)


def test_missing_pinned_input_token_discrepancy_is_rejected():
    observed = dict(A.PINNED_INPUT_TOKEN_DISCREPANCIES)
    observed.pop(5)

    with pytest.raises(A.DatasetIntegrityError, match="missing"):
        A._validate_input_token_discrepancies(observed)


def test_validate_questions_requires_exact_population():
    with pytest.raises(A.DatasetIntegrityError, match="question IDs 1..100"):
        A.validate_questions([question(question_id=1)])


def test_prepare_fixture_preserves_csv_filename_order(tmp_path, monkeypatch):
    prepared = prepare_synthetic_100_question_fixture(tmp_path, monkeypatch)

    assert len(prepared.questions) == 100
    assert {item.question_id for item in prepared.questions} == set(range(1, 101))
    assert {item.document_set_id for item in prepared.questions} == {
        f"set-{number}" for number in range(1, 31)
    }
    q1 = prepared.questions[0]
    assert q1.document_filenames == ("b.txt", "a.txt")
    assert "BEGIN DOCUMENT 1:\nB" in q1.prompt
    assert q1.prompt.index("BEGIN DOCUMENT 1:\nB") < q1.prompt.index(
        "BEGIN DOCUMENT 2:\nA"
    )
    assert q1.cl100k_tokens == len(
        A.tiktoken.get_encoding("cl100k_base").encode(q1.prompt)
    )
    assert set(prepared.file_sha256) == {
        A.CSV_FILENAME,
        A.ZIP_FILENAME,
        *(f"lcr/synthetic/set-{set_id}/a.txt" for set_id in range(1, 31)),
        "lcr/synthetic/set-1/b.txt",
        A.OFFICIAL_UNREFERENCED_MEMBER,
    }
    payloads = fixture_payloads()
    assert (
        prepared.file_sha256[A.CSV_FILENAME]
        == hashlib.sha256(payloads[A.CSV_FILENAME]).hexdigest()
    )
    assert all(
        (
            tmp_path
            / "prepared"
            / "extracted_text"
            / "lcr"
            / "synthetic"
            / "set-1"
            / filename
        ).is_file()
        for filename in ("a.txt", "b.txt")
    )
    assert not (tmp_path / "escape.txt").exists()
