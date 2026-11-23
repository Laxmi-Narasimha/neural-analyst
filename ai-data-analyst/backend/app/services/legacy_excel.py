"""Read selected BIFF worksheets without building an in-memory xlrd sheet.

xlrd decodes formats, formulas and older BIFF dialects. Its normal worksheet and
BIFF8 shared-string containers are replaced by per-scan, disk-backed storage.
"""
from __future__ import annotations

from contextlib import closing
from itertools import groupby
from pathlib import Path
import shutil
import sqlite3
import struct
from sys import getsizeof
from tempfile import TemporaryDirectory

import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES
import xlrd
from compoundfiles import CompoundFileReader
from xlrd.biffh import XL_CONTINUE, XL_WORKBOOK_GLOBALS, XL_WORKSHEET, XLRDError
from xlrd.book import Book, SUPPORTED_VERSIONS
from xlrd.sheet import Sheet


_STRING_REF = 7  # private storage tag, outside xlrd's cell-type enumeration


class _SharedStringIds:
    def __init__(self, count):
        self.count = count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise XLRDError("Invalid Excel shared-string reference")
        return index


class _StringRecords:
    """Read SST and CONTINUE payloads, including compression changes mid-string."""
    def __init__(self, book, data):
        self.book = book
        self.data = data
        self.pos = 8

    def advance(self):
        kind, size, self.data = self.book.get_record_parts_conditional(XL_CONTINUE)
        if kind is None or not size:
            raise XLRDError("Incomplete Excel shared-string table")
        self.pos = 0

    def read(self, size):
        output = bytearray()
        while size:
            if self.pos == len(self.data):
                self.advance()
            n = min(size, len(self.data) - self.pos)
            output.extend(self.data[self.pos:self.pos + n])
            self.pos += n
            size -= n
        return output

    def string(self):
        units, flags = struct.unpack("<HB", self.read(3))
        runs = int.from_bytes(self.read(2), "little") if flags & 8 else 0
        extended = int.from_bytes(self.read(4), "little") if flags & 4 else 0
        utf16 = bytearray()
        while units:
            if self.pos == len(self.data):
                self.advance()
                flags = self.read(1)[0]
            width = 2 if flags & 1 else 1
            n = min(units, (len(self.data) - self.pos) // width)
            if not n:
                raise XLRDError("Invalid character boundary in Excel shared string")
            part = self.read(n * width)
            utf16.extend(part if width == 2 else part.decode("latin1").encode("utf-16-le"))
            units -= n
        skip = runs * 4 + extended
        while skip:
            n = min(skip, 8192)
            self.read(n)
            skip -= n
        return utf16.decode("utf-16-le")


class _DiskBook(Book):
    def __init__(self, db, selected):
        super().__init__()
        self.db = db
        self.selected = selected

    def handle_sst(self, data):
        count = struct.unpack_from("<I", data, 4)[0]
        source = _StringRecords(self, data)
        self.db.executemany("INSERT INTO strings VALUES (?, ?)", ((i, source.string()) for i in range(count)))
        self._sharedstrings = _SharedStringIds(count)

    def get_sheet(self, sh_number, update_pos=True):
        if self.selected is None and sh_number != 0:
            return
        if self.selected is not None and self.selected != "all" and self._sheet_names[sh_number] not in self.selected:
            return
        if update_pos:
            self._position = self._sh_abs_posn[sh_number]
        self.getbof(XL_WORKSHEET)
        sheet = Sheet(self, self._position, self._sheet_names[sh_number], sh_number)
        def store(row, col, kind, value, xf):
            if kind is None:
                kind = sheet._xf_index_to_xl_type_map[xf]
            if kind == xlrd.XL_CELL_TEXT and isinstance(value, int):
                kind = _STRING_REF
            if kind not in {xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK}:
                self.db.execute("INSERT OR REPLACE INTO cells VALUES (?, ?, ?, ?, ?)", (sh_number, row, col, kind, value))
        sheet.put_cell = store
        sheet.read(self)
        self._sheet_list[sh_number] = sheet
        return sheet

    def decode(self, path):
        self.biff2_8_load(filename=str(path), use_mmap=True, on_demand=True, ragged_rows=True)
        self.biff_version = self.getbof(XL_WORKBOOK_GLOBALS)
        if self.biff_version not in SUPPORTED_VERSIONS:
            raise XLRDError("Unsupported binary Excel workbook version")
        if self.biff_version <= 40:
            self.fake_globals_get_sheet()
        else:
            self.parse_globals()
            if not self._sheet_names:
                raise XLRDError("No worksheets found")
            if self.biff_version != 45:  # BIFF4W reads its embedded sheets while parsing globals.
                self._sheet_list = [None] * len(self._sheet_names)
                for idx in range(len(self._sheet_names)):
                    self.get_sheet(idx)


def _rows(db, index, width, epoch):
    next_row = 0
    cells = db.execute("SELECT r,c,kind,CASE WHEN kind=? THEN s.value ELSE x.value END "
                       "FROM cells x LEFT JOIN strings s ON kind=? AND s.id=x.value WHERE sheet=? ORDER BY r,c",
                       (_STRING_REF, _STRING_REF, index))
    for index, group in groupby(cells, key=lambda cell: cell[0]):
        while next_row < index:
            yield [None] * width
            next_row += 1
        row = [None] * width
        for _, col, kind, value in group:
            if kind == xlrd.XL_CELL_DATE:
                try:
                    value = xlrd.xldate_as_datetime(value, epoch)
                except OverflowError:
                    row[col] = value
                    continue
                if value.date().isoformat() == ("1904-01-01" if epoch else "1899-12-31"):
                    value = value.time()
            elif kind == xlrd.XL_CELL_BOOLEAN:
                value = bool(value)
            elif kind == xlrd.XL_CELL_ERROR or value == "":
                value = None
            elif index and isinstance(value, str) and value in STR_NA_VALUES:
                value = None
            elif kind == xlrd.XL_CELL_NUMBER and isinstance(value, float) and value.is_integer():
                value = int(value)
            row[col] = value
        yield row
        next_row += 1


def xls_sheets(path: Path, batch_rows: int, selected):
    """Yield (name, batches) in workbook order. Consume each iterator before advancing."""
    with TemporaryDirectory(prefix="analyst-xls-") as work, closing(sqlite3.connect(Path(work) / "sheet.db")) as db:
        db.executescript("PRAGMA journal_mode=OFF; PRAGMA cache_size=-2048; "
                         "CREATE TABLE strings(id INTEGER PRIMARY KEY, value TEXT); "
                         "CREATE TABLE cells(sheet INTEGER,r INTEGER,c INTEGER,kind INTEGER,value,PRIMARY KEY(sheet,r,c)) WITHOUT ROWID;")
        with path.open("rb") as src:
            signature = src.read(8)
        stream = path
        if signature == bytes.fromhex("d0cf11e0a1b11ae1"):
            # A fragmented OLE chain must not be concatenated into one Python bytes object.
            stream = Path(work) / "workbook.biff"
            with CompoundFileReader(str(path)) as ole:
                name = next((n for n in ("Workbook", "Book") if n in ole), None)
                if name is None:
                    raise XLRDError("No Excel workbook stream found")
                with ole.open(name) as source, stream.open("wb") as target:
                    shutil.copyfileobj(source, target, 64 * 1024)
        book = _DiskBook(db, selected)
        try:
            book.decode(stream)
            epoch = book.datemode
        finally:
            book.release_resources()
        for index, name in enumerate(book._sheet_names):
            if selected == "all" or (selected is None and index == 0) or (selected and name in selected):
                yield name, _sheet_batches(db, index, epoch, batch_rows)


def _sheet_batches(db, index, epoch, batch_rows):
    width = db.execute("SELECT max(c)+1 FROM cells WHERE sheet=?", (index,)).fetchone()[0]
    if width is None:
        return
    rows = _rows(db, index, width, epoch)
    header = next(rows)
    names = []
    originals = {str(v) for v in header if v is not None}
    for i, val in enumerate(header):
        base = str(val) if val is not None else f"Unnamed: {i}"
        name, suffix = base, 1
        while name in names or (suffix > 1 and name in originals):
            name = f"{base}.{suffix}"
            suffix += 1
        names.append(name)
    yield pd.DataFrame(columns=names)
    batch, size = [], 0
    for row in rows:
        batch.append(row)
        size += getsizeof(row) + sum(map(getsizeof, row))
        if len(batch) >= batch_rows or size >= 4 * 1024 * 1024:
            yield pd.DataFrame.from_records(batch, columns=names)
            batch, size = [], 0
    if batch:
        yield pd.DataFrame.from_records(batch, columns=names)


def xls_frames(path: Path, batch_rows: int):
    for _, batches in xls_sheets(path, batch_rows, None):
        yield from (frame for frame in batches if len(frame))
