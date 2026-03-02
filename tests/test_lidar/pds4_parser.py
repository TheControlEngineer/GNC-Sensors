"""
Minimal helpers for loading OLA PDS4 binary tables.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def parse_ola_label(xml_path):
    """
    Parse a PDS4 label and return column descriptors plus record length.

    :param xml_path: Path to the .xml PDS4 label.
    :return: (columns, record_length)
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    table = root.find(".//{*}Table_Binary")
    if table is None:
        raise ValueError(f"No Table_Binary found in label: {xml_path}")

    record_bin = table.find("{*}Record_Binary")
    if record_bin is None:
        raise ValueError(f"No Record_Binary found in label: {xml_path}")

    record_length_tag = record_bin.find("{*}record_length")
    if record_length_tag is None:
        raise ValueError(f"No record_length found in label: {xml_path}")
    record_length = int(record_length_tag.text)

    columns = []
    for field in record_bin.findall("{*}Field_Binary"):
        name_tag = field.find("{*}name")
        loc_tag = field.find("{*}field_location")
        len_tag = field.find("{*}field_length")
        type_tag = field.find("{*}data_type")
        if name_tag is None or loc_tag is None or len_tag is None or type_tag is None:
            continue
        name = str(name_tag.text).strip()
        offset = int(loc_tag.text) - 1
        length = int(len_tag.text)
        pds_type = str(type_tag.text).strip()
        dtype = _pds4_to_numpy_dtype(pds_type, length)
        columns.append(
            {
                "name": name,
                "offset": offset,
                "length": length,
                "pds4_type": pds_type,
                "dtype": dtype,
            }
        )

    if not columns:
        raise ValueError(f"No Field_Binary entries found in label: {xml_path}")

    return columns, record_length


def _pds4_to_numpy_dtype(pds4_type, length):
    mapping = {
        "IEEE754LSBSingle": "<f4",
        "IEEE754LSBDouble": "<f8",
        "IEEE754MSBSingle": ">f4",
        "IEEE754MSBDouble": ">f8",
        "SignedByte": "i1",
        "UnsignedByte": "u1",
        "SignedLSB2": "<i2",
        "UnsignedLSB2": "<u2",
        "SignedLSB4": "<i4",
        "UnsignedLSB4": "<u4",
        "SignedLSB8": "<i8",
        "UnsignedLSB8": "<u8",
        "SignedMSB2": ">i2",
        "UnsignedMSB2": ">u2",
        "SignedMSB4": ">i4",
        "UnsignedMSB4": ">u4",
        "SignedMSB8": ">i8",
        "UnsignedMSB8": ">u8",
        "ComplexLSB8": "<c8",
        "ComplexLSB16": "<c16",
        "ComplexMSB8": ">c8",
        "ComplexMSB16": ">c16",
    }
    if pds4_type in mapping:
        return mapping[pds4_type]
    return f"S{int(length)}"


def load_ola_binary(dat_path, columns, record_length, max_records=None):
    """
    Load a PDS4 binary table into a structured numpy array.

    :param dat_path: Path to the .dat file.
    :param columns: Column descriptors from parse_ola_label.
    :param record_length: Bytes per record.
    :param max_records: Optional row limit.
    :return: Structured numpy array.
    """
    dat_path = Path(dat_path)
    dtype = np.dtype(
        {
            "names": [sanitize_name(col["name"]) for col in columns],
            "formats": [col["dtype"] for col in columns],
            "offsets": [int(col["offset"]) for col in columns],
            "itemsize": int(record_length),
        }
    )
    arr = np.fromfile(dat_path, dtype=dtype, count=-1 if max_records is None else int(max_records))
    arr = _decode_ascii_columns(arr)
    return arr


def load_ola_with_pds4_tools(xml_path):
    """
    Try loading with pds4_tools when available.

    :param xml_path: Path to .xml label.
    :return: Structured numpy array or None.
    """
    try:
        from pds4_tools import pds4_read
    except Exception:
        return None

    try:
        product = pds4_read(str(xml_path), quiet=True)
    except Exception:
        return None

    for item in product:
        try:
            if hasattr(item, "data") and isinstance(item.data, np.ndarray):
                arr = item.data
                if arr.dtype.names:
                    return _decode_ascii_columns(arr)
        except Exception:
            continue
    return None


def load_ola_dataset(xml_path, dat_path=None, max_records=None):
    """
    Load OLA data from xml and dat with pds4_tools fallback to manual parser.

    :param xml_path: Path to label file.
    :param dat_path: Optional explicit .dat path.
    :param max_records: Optional row limit for manual parse.
    :return: Structured numpy array.
    """
    xml_path = Path(xml_path)

    arr = load_ola_with_pds4_tools(xml_path)
    if arr is not None:
        if max_records is not None:
            return arr[: int(max_records)]
        return arr

    columns, record_length = parse_ola_label(xml_path)
    if dat_path is None:
        dat_guess = xml_path.with_suffix(".dat")
        if not dat_guess.exists():
            raise FileNotFoundError(f"Could not infer .dat file from label: {xml_path}")
        dat_path = dat_guess
    return load_ola_binary(dat_path, columns, record_length, max_records=max_records)


def sanitize_name(name):
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def _decode_ascii_columns(arr):
    if arr.dtype.names is None:
        return arr

    decoded_dtype = []
    has_bytes = False
    for name in arr.dtype.names:
        dt = arr.dtype.fields[name][0]
        if dt.kind == "S":
            decoded_dtype.append((name, f"U{dt.itemsize}"))
            has_bytes = True
        else:
            decoded_dtype.append((name, dt))

    if not has_bytes:
        return arr

    out = np.empty(arr.shape, dtype=np.dtype(decoded_dtype))
    for name in arr.dtype.names:
        dt = arr.dtype.fields[name][0]
        if dt.kind == "S":
            out[name] = np.char.decode(arr[name], "utf-8", errors="ignore")
        else:
            out[name] = arr[name]
    return out
