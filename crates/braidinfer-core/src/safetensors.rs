use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::types::DType;

#[derive(Debug)]
pub enum SafeTensorsError {
    Io(std::io::Error),
    InvalidHeader(&'static str),
    UnknownDtype(String),
    TensorNotFound(String),
}

impl std::fmt::Display for SafeTensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafeTensorsError::Io(e) => write!(f, "IO error: {e}"),
            SafeTensorsError::InvalidHeader(msg) => write!(f, "Invalid header: {msg}"),
            SafeTensorsError::UnknownDtype(s) => write!(f, "Unknown dtype: {s}"),
            SafeTensorsError::TensorNotFound(s) => write!(f, "Tensor not found: {s}"),
        }
    }
}

impl std::error::Error for SafeTensorsError {}

impl From<std::io::Error> for SafeTensorsError {
    fn from(e: std::io::Error) -> Self {
        SafeTensorsError::Io(e)
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data_offset: usize,
    pub data_len: usize,
}

pub struct SafeTensors {
    data: Vec<u8>,
    header_size: usize,
    tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self, SafeTensorsError> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        if data.len() < 8 {
            return Err(SafeTensorsError::InvalidHeader("file too small"));
        }

        let header_size = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;

        if data.len() < 8 + header_size {
            return Err(SafeTensorsError::InvalidHeader("file truncated"));
        }

        let header_bytes = &data[8..8 + header_size];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|_| SafeTensorsError::InvalidHeader("header not utf8"))?;

        let tensors = parse_header(header_str)?;

        Ok(SafeTensors { data, header_size, tensors })
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        let base = 8 + self.header_size;
        let start = base + info.data_offset;
        let end = start + info.data_len;
        self.data.get(start..end)
    }

    pub fn tensor_as_f32(&self, name: &str) -> Option<Vec<f32>> {
        let info = self.tensors.get(name)?;
        let raw = self.tensor_data(name)?;
        Some(convert_to_f32(raw, info.dtype))
    }

    /// Return raw u16 values for BF16/F16 tensors (no conversion).
    /// For F32 tensors, converts to BF16 (truncates mantissa).
    pub fn tensor_as_u16(&self, name: &str) -> Option<Vec<u16>> {
        let info = self.tensors.get(name)?;
        let raw = self.tensor_data(name)?;
        Some(convert_to_u16(raw, info.dtype))
    }
}

fn convert_to_f32(raw: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        DType::F16 => raw
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes(b.try_into().unwrap());
                f16_to_f32(bits)
            })
            .collect(),
        DType::BF16 => raw
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes(b.try_into().unwrap());
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        DType::I8 => raw.iter().map(|&b| b as i8 as f32).collect(),
        DType::I4 => raw
            .iter()
            .flat_map(|&b| {
                let lo = (b & 0x0f) as i8;
                let hi = ((b >> 4) & 0x0f) as i8;
                [lo as f32, hi as f32]
            })
            .collect(),
    }
}

fn convert_to_u16(raw: &[u8], dtype: DType) -> Vec<u16> {
    match dtype {
        DType::BF16 | DType::F16 => raw
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        DType::F32 => raw
            .chunks_exact(4)
            .map(|b| {
                let bits = u32::from_le_bytes(b.try_into().unwrap());
                (bits >> 16) as u16 // FP32 → BF16 truncation
            })
            .collect(),
        _ => panic!("tensor_as_u16 unsupported for {:?}", dtype),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = (bits >> 10) & 0x1f;
    let mant = (bits & 0x3ff) as u32;
    let (f32_exp, f32_mant) = if exp == 0 {
        if mant == 0 {
            (0u32, 0u32)
        } else {
            let mut e = 127u32 - 14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (e, (m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        (255u32, mant << 13)
    } else {
        ((exp as u32) + (127 - 15), mant << 13)
    };
    f32::from_bits(sign | (f32_exp << 23) | f32_mant)
}

// Minimal JSON parser for safetensors headers.
// Header is: { "__metadata__": {...}, "name": {"dtype": "...", "shape": [...], "data_offsets": [start, end]}, ... }
fn parse_header(s: &str) -> Result<HashMap<String, TensorInfo>, SafeTensorsError> {
    let s = s.trim();
    if !s.starts_with('{') {
        return Err(SafeTensorsError::InvalidHeader("expected object"));
    }

    let mut tensors = HashMap::new();
    let mut pos = 1usize; // skip '{'

    loop {
        pos = skip_ws(s, pos);
        if pos >= s.len() {
            break;
        }
        if s.as_bytes()[pos] == b'}' {
            break;
        }

        // Parse key
        let (key, next) = parse_string(s, pos)?;
        pos = next;
        pos = skip_ws(s, pos);
        if pos >= s.len() || s.as_bytes()[pos] != b':' {
            return Err(SafeTensorsError::InvalidHeader("expected ':'"));
        }
        pos += 1;
        pos = skip_ws(s, pos);

        // Parse value (object)
        if s.as_bytes()[pos] != b'{' {
            // skip non-object values (shouldn't happen except __metadata__)
            pos = skip_value(s, pos)?;
        } else {
            let (dtype, shape, offsets, next) = parse_tensor_object(s, pos)?;
            pos = next;
            if key != "__metadata__" {
                if let (Some(dtype), Some(shape), Some([start, end])) = (dtype, shape, offsets) {
                    let data_len = end - start;
                    tensors.insert(
                        key.clone(),
                        TensorInfo {
                            name: key,
                            dtype,
                            shape,
                            data_offset: start,
                            data_len,
                        },
                    );
                }
            }
        }

        pos = skip_ws(s, pos);
        if pos < s.len() && s.as_bytes()[pos] == b',' {
            pos += 1;
        }
    }

    Ok(tensors)
}

fn skip_ws(s: &str, mut pos: usize) -> usize {
    let b = s.as_bytes();
    while pos < b.len() && (b[pos] == b' ' || b[pos] == b'\t' || b[pos] == b'\n' || b[pos] == b'\r') {
        pos += 1;
    }
    pos
}

fn parse_string(s: &str, pos: usize) -> Result<(String, usize), SafeTensorsError> {
    let b = s.as_bytes();
    if b[pos] != b'"' {
        return Err(SafeTensorsError::InvalidHeader("expected '\"'"));
    }
    let mut i = pos + 1;
    let mut result = String::new();
    while i < b.len() {
        if b[i] == b'\\' && i + 1 < b.len() {
            result.push(b[i + 1] as char);
            i += 2;
        } else if b[i] == b'"' {
            return Ok((result, i + 1));
        } else {
            result.push(b[i] as char);
            i += 1;
        }
    }
    Err(SafeTensorsError::InvalidHeader("unterminated string"))
}

fn parse_u64(s: &str, pos: usize) -> Result<(u64, usize), SafeTensorsError> {
    let b = s.as_bytes();
    let mut i = pos;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == pos {
        return Err(SafeTensorsError::InvalidHeader("expected number"));
    }
    let n: u64 = s[pos..i].parse().map_err(|_| SafeTensorsError::InvalidHeader("number parse failed"))?;
    Ok((n, i))
}

fn parse_usize_array(s: &str, pos: usize) -> Result<(Vec<usize>, usize), SafeTensorsError> {
    let b = s.as_bytes();
    if b[pos] != b'[' {
        return Err(SafeTensorsError::InvalidHeader("expected '['"));
    }
    let mut pos = pos + 1;
    let mut vals = Vec::new();
    loop {
        pos = skip_ws(s, pos);
        if b[pos] == b']' {
            return Ok((vals, pos + 1));
        }
        let (n, next) = parse_u64(s, pos)?;
        vals.push(n as usize);
        pos = skip_ws(s, next);
        if b[pos] == b',' {
            pos += 1;
        }
    }
}

fn skip_value(s: &str, pos: usize) -> Result<usize, SafeTensorsError> {
    let b = s.as_bytes();
    match b[pos] {
        b'"' => {
            let (_, next) = parse_string(s, pos)?;
            Ok(next)
        }
        b'{' => {
            let mut depth = 1;
            let mut i = pos + 1;
            while i < b.len() && depth > 0 {
                match b[i] {
                    b'"' => {
                        let (_, next) = parse_string(s, i)?;
                        i = next;
                        continue;
                    }
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            Ok(i)
        }
        b'[' => {
            let mut depth = 1;
            let mut i = pos + 1;
            while i < b.len() && depth > 0 {
                match b[i] {
                    b'"' => {
                        let (_, next) = parse_string(s, i)?;
                        i = next;
                        continue;
                    }
                    b'[' => depth += 1,
                    b']' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            Ok(i)
        }
        _ => {
            let mut i = pos;
            while i < b.len() && b[i] != b',' && b[i] != b'}' && b[i] != b']' {
                i += 1;
            }
            Ok(i)
        }
    }
}

fn parse_tensor_object(
    s: &str,
    pos: usize,
) -> Result<(Option<DType>, Option<Vec<usize>>, Option<[usize; 2]>, usize), SafeTensorsError> {
    let b = s.as_bytes();
    assert_eq!(b[pos], b'{');
    let mut pos = pos + 1;

    let mut dtype: Option<DType> = None;
    let mut shape: Option<Vec<usize>> = None;
    let mut offsets: Option<[usize; 2]> = None;

    loop {
        pos = skip_ws(s, pos);
        if pos >= b.len() {
            break;
        }
        if b[pos] == b'}' {
            pos += 1;
            break;
        }

        let (key, next) = parse_string(s, pos)?;
        pos = next;
        pos = skip_ws(s, pos);
        if pos >= b.len() || b[pos] != b':' {
            return Err(SafeTensorsError::InvalidHeader("expected ':'"));
        }
        pos += 1;
        pos = skip_ws(s, pos);

        match key.as_str() {
            "dtype" => {
                let (val, next) = parse_string(s, pos)?;
                dtype = Some(parse_dtype(&val)?);
                pos = next;
            }
            "shape" => {
                let (v, next) = parse_usize_array(s, pos)?;
                shape = Some(v);
                pos = next;
            }
            "data_offsets" => {
                let (v, next) = parse_usize_array(s, pos)?;
                if v.len() != 2 {
                    return Err(SafeTensorsError::InvalidHeader("data_offsets must have 2 elements"));
                }
                offsets = Some([v[0], v[1]]);
                pos = next;
            }
            _ => {
                pos = skip_value(s, pos)?;
            }
        }

        pos = skip_ws(s, pos);
        if pos < b.len() && b[pos] == b',' {
            pos += 1;
        }
    }

    Ok((dtype, shape, offsets, pos))
}

fn parse_dtype(s: &str) -> Result<DType, SafeTensorsError> {
    match s {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I8" => Ok(DType::I8),
        "I32" | "I64" | "I4" => Ok(DType::I8), // map to closest
        "U8" | "BOOL" => Ok(DType::I8),
        _ => Err(SafeTensorsError::UnknownDtype(s.to_string())),
    }
}

// Multi-file support via index JSON

pub struct SafeTensorSet {
    shards: Vec<SafeTensors>,
    index: HashMap<String, usize>, // tensor name -> shard index
}

impl SafeTensorSet {
    pub fn open_directory(dir: &Path) -> Result<Self, SafeTensorsError> {
        // Try to find index file
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            Self::open_with_index(dir, &index_path)
        } else {
            // Single file fallback
            let single = dir.join("model.safetensors");
            let st = SafeTensors::open(&single)?;
            let names: Vec<String> = st.tensors.keys().cloned().collect();
            let mut index = HashMap::new();
            for name in names {
                index.insert(name, 0);
            }
            Ok(SafeTensorSet { shards: vec![st], index })
        }
    }

    fn open_with_index(dir: &Path, index_path: &Path) -> Result<Self, SafeTensorsError> {
        let mut file = File::open(index_path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;

        // Parse index JSON: {"metadata": {...}, "weight_map": {"name": "file", ...}}
        let weight_map = parse_index_json(&content)?;

        // Collect unique filenames preserving order
        let mut file_order: Vec<String> = Vec::new();
        let mut file_to_idx: HashMap<String, usize> = HashMap::new();
        for filename in weight_map.values() {
            if !file_to_idx.contains_key(filename) {
                file_to_idx.insert(filename.clone(), file_order.len());
                file_order.push(filename.clone());
            }
        }

        let mut shards = Vec::with_capacity(file_order.len());
        for filename in &file_order {
            let path = dir.join(filename);
            shards.push(SafeTensors::open(&path)?);
        }

        let mut index = HashMap::new();
        for (tensor_name, filename) in weight_map {
            let shard_idx = file_to_idx[&filename];
            index.insert(tensor_name, shard_idx);
        }

        Ok(SafeTensorSet { shards, index })
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.index.keys().map(|s| s.as_str()).collect()
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let &shard_idx = self.index.get(name)?;
        self.shards[shard_idx].tensor_info(name)
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let &shard_idx = self.index.get(name)?;
        self.shards[shard_idx].tensor_data(name)
    }

    pub fn tensor_as_f32(&self, name: &str) -> Option<Vec<f32>> {
        let &shard_idx = self.index.get(name)?;
        self.shards[shard_idx].tensor_as_f32(name)
    }

    pub fn tensor_as_u16(&self, name: &str) -> Option<Vec<u16>> {
        let &shard_idx = self.index.get(name)?;
        self.shards[shard_idx].tensor_as_u16(name)
    }
}

fn parse_index_json(s: &str) -> Result<HashMap<String, String>, SafeTensorsError> {
    // Parse {"metadata": {...}, "weight_map": {"tensor": "file", ...}}
    let s = s.trim();
    if !s.starts_with('{') {
        return Err(SafeTensorsError::InvalidHeader("index: expected object"));
    }
    let mut pos = 1;
    let mut weight_map = HashMap::new();

    loop {
        pos = skip_ws(s, pos);
        let b = s.as_bytes();
        if pos >= b.len() || b[pos] == b'}' {
            break;
        }

        let (key, next) = parse_string(s, pos)?;
        pos = next;
        pos = skip_ws(s, pos);
        if s.as_bytes()[pos] != b':' {
            return Err(SafeTensorsError::InvalidHeader("index: expected ':'"));
        }
        pos += 1;
        pos = skip_ws(s, pos);

        if key == "weight_map" {
            // parse object of string->string
            if s.as_bytes()[pos] != b'{' {
                return Err(SafeTensorsError::InvalidHeader("weight_map must be object"));
            }
            pos += 1;
            loop {
                pos = skip_ws(s, pos);
                let b = s.as_bytes();
                if pos >= b.len() || b[pos] == b'}' {
                    pos += 1;
                    break;
                }
                let (tname, next) = parse_string(s, pos)?;
                pos = next;
                pos = skip_ws(s, pos);
                if s.as_bytes()[pos] != b':' {
                    return Err(SafeTensorsError::InvalidHeader("weight_map entry: expected ':'"));
                }
                pos += 1;
                pos = skip_ws(s, pos);
                let (fname, next) = parse_string(s, pos)?;
                pos = next;
                weight_map.insert(tname, fname);
                pos = skip_ws(s, pos);
                if pos < s.len() && s.as_bytes()[pos] == b',' {
                    pos += 1;
                }
            }
        } else {
            pos = skip_value(s, pos)?;
        }

        pos = skip_ws(s, pos);
        if pos < s.len() && s.as_bytes()[pos] == b',' {
            pos += 1;
        }
    }

    Ok(weight_map)
}
