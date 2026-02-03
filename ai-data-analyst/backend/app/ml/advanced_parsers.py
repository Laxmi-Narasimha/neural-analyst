# AI Enterprise Data Analyst - Advanced File Parsers
# Fixed-width, XML streaming, multi-encoding support

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, BinaryIO, Optional, Iterator
import pandas as pd
import io
from app.core.logging import get_logger
try:
    from app.core.exceptions import FileParseException
except ImportError:
    class FileParseException(Exception): pass

logger = get_logger(__name__)


@dataclass
class ColumnSpec:
    """Fixed-width column specification."""
    name: str
    start: int
    end: int
    dtype: str = "str"


class FixedWidthParser:
    """Parse fixed-width (positional) files from mainframes."""
    
    def parse(
        self, file_data: BinaryIO, column_specs: list[ColumnSpec], 
        encoding: str = "utf-8", skip_rows: int = 0
    ) -> pd.DataFrame:
        content = file_data.read().decode(encoding)
        lines = content.strip().split('\n')[skip_rows:]
        
        data = []
        for line in lines:
            row = {}
            for spec in column_specs:
                value = line[spec.start:spec.end].strip()
                row[spec.name] = value
            data.append(row)
        
        df = pd.DataFrame(data)
        
        for spec in column_specs:
            if spec.dtype == "int":
                df[spec.name] = pd.to_numeric(df[spec.name], errors='coerce').astype('Int64')
            elif spec.dtype == "float":
                df[spec.name] = pd.to_numeric(df[spec.name], errors='coerce')
            elif spec.dtype == "date":
                df[spec.name] = pd.to_datetime(df[spec.name], errors='coerce')
        
        return df
    
    def auto_detect_columns(
        self, file_data: BinaryIO, encoding: str = "utf-8", sample_lines: int = 100
    ) -> list[ColumnSpec]:
        """Auto-detect column boundaries from whitespace patterns."""
        content = file_data.read().decode(encoding)
        file_data.seek(0)
        lines = content.strip().split('\n')[:sample_lines]
        
        if not lines:
            return []
        
        max_len = max(len(line) for line in lines)
        space_counts = [0] * max_len
        
        for line in lines:
            for i, char in enumerate(line):
                if char == ' ':
                    space_counts[i] += 1
        
        threshold = len(lines) * 0.8
        boundaries = [0]
        in_gap = False
        
        for i, count in enumerate(space_counts):
            if count >= threshold and not in_gap:
                in_gap = True
            elif count < threshold and in_gap:
                boundaries.append(i)
                in_gap = False
        
        boundaries.append(max_len)
        
        specs = []
        for i in range(len(boundaries) - 1):
            specs.append(ColumnSpec(
                name=f"col_{i+1}",
                start=boundaries[i],
                end=boundaries[i+1]
            ))
        
        return specs


class XMLStreamingParser:
    """Stream large XML files without loading into memory."""
    
    def parse(
        self, file_path: str, record_tag: str, 
        fields: list[str] = None, max_records: int = None
    ) -> Iterator[dict]:
        try:
            import xml.etree.ElementTree as ET
            
            context = ET.iterparse(file_path, events=('end',))
            count = 0
            
            for event, elem in context:
                if elem.tag == record_tag:
                    record = {}
                    
                    for child in elem:
                        if fields is None or child.tag in fields:
                            record[child.tag] = child.text
                    
                    for attr, value in elem.attrib.items():
                        if fields is None or attr in fields:
                            record[f"@{attr}"] = value
                    
                    yield record
                    count += 1
                    
                    elem.clear()
                    
                    if max_records and count >= max_records:
                        break
                        
        except Exception as e:
            logger.error(f"XML parsing error: {e}")
            raise FileParseException(filename=file_path, parse_errors=[str(e)])
    
    def to_dataframe(
        self, file_path: str, record_tag: str, 
        fields: list[str] = None, max_records: int = 100000
    ) -> pd.DataFrame:
        records = list(self.parse(file_path, record_tag, fields, max_records))
        return pd.DataFrame(records)


class MultiEncodingParser:
    """Handle files with mixed or unknown encodings."""
    
    ENCODINGS = [
        'utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252',
        'shift_jis', 'gb2312', 'gbk', 'gb18030', 'big5',
        'euc-kr', 'euc-jp', 'utf-16', 'utf-16-le', 'utf-16-be'
    ]
    
    def detect_encoding(self, file_data: BinaryIO) -> str:
        sample = file_data.read(10000)
        file_data.seek(0)
        
        # Check for BOM
        if sample.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        elif sample.startswith(b'\xff\xfe'):
            return 'utf-16-le'
        elif sample.startswith(b'\xfe\xff'):
            return 'utf-16-be'
        
        # Try chardet if available
        try:
            import chardet
            result = chardet.detect(sample)
            if result['confidence'] > 0.7:
                return result['encoding']
        except ImportError:
            pass
        
        # Fallback: try each encoding
        for enc in self.ENCODINGS:
            try:
                sample.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue
        
        return 'utf-8'
    
    def read_with_fallback(
        self, file_data: BinaryIO, preferred_encoding: str = None
    ) -> tuple[str, str]:
        """Read file content with encoding fallback. Returns (content, encoding_used)."""
        content = file_data.read()
        file_data.seek(0)
        
        encodings = [preferred_encoding] if preferred_encoding else []
        encodings.extend(self.ENCODINGS)
        
        for enc in encodings:
            if enc is None:
                continue
            try:
                decoded = content.decode(enc)
                return decoded, enc
            except (UnicodeDecodeError, LookupError):
                continue
        
        return content.decode('utf-8', errors='replace'), 'utf-8'


class ChunkedCSVParser:
    """Parse large CSV files in chunks for memory efficiency."""
    
    def parse_chunks(
        self, file_path: str, chunk_size: int = 100000, 
        **read_csv_kwargs
    ) -> Iterator[pd.DataFrame]:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, **read_csv_kwargs):
            yield chunk
    
    def aggregate_chunks(
        self, file_path: str, agg_func: callable, 
        chunk_size: int = 100000, **read_csv_kwargs
    ) -> Any:
        """Apply aggregation function across chunks."""
        results = []
        for chunk in self.parse_chunks(file_path, chunk_size, **read_csv_kwargs):
            results.append(agg_func(chunk))
        return results
    
    def count_rows(self, file_path: str) -> int:
        """Count rows without loading entire file."""
        count = 0
        for chunk in pd.read_csv(file_path, chunksize=100000, usecols=[0]):
            count += len(chunk)
        return count


def get_fixed_width_parser() -> FixedWidthParser:
    return FixedWidthParser()

def get_xml_parser() -> XMLStreamingParser:
    return XMLStreamingParser()

def get_multi_encoding_parser() -> MultiEncodingParser:
    return MultiEncodingParser()

def get_chunked_csv_parser() -> ChunkedCSVParser:
    return ChunkedCSVParser()
