use std::collections::HashMap;
use std::fs;
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;

#[derive(Debug)]
pub enum SafeTensorsError {
    Io(std::io::Error),
    Parse(safetensors::SafeTensorError),
    TensorNotFound(String),
}

impl std::fmt::Display for SafeTensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafeTensorsError::Io(e) => write!(f, "IO error: {e}"),
            SafeTensorsError::Parse(e) => write!(f, "SafeTensors parse error: {e}"),
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

impl From<safetensors::SafeTensorError> for SafeTensorsError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        SafeTensorsError::Parse(e)
    }
}

/// Single mmap'd safetensors file. Holds the mmap and parsed metadata together.
pub struct MmapSafeTensors {
    mmap: Mmap,
    header_size: usize,
    metadata: safetensors::tensor::Metadata,
}

impl MmapSafeTensors {
    pub fn open(path: &Path) -> Result<Self, SafeTensorsError> {
        let file = fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let (header_size, metadata) = SafeTensors::read_metadata(&mmap)?;
        Ok(MmapSafeTensors {
            mmap,
            header_size,
            metadata,
        })
    }

    pub fn mmap(&self) -> &Mmap {
        &self.mmap
    }

    pub fn names(&self) -> Vec<String> {
        self.metadata.tensors().keys().cloned().collect()
    }

    pub fn tensor_info(&self, name: &str) -> Option<&safetensors::tensor::TensorInfo> {
        self.metadata.info(name)
    }

    /// Get raw tensor data directly from mmap, zero-copy.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8], SafeTensorsError> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| SafeTensorsError::TensorNotFound(name.to_string()))?;
        let data_start = 8 + self.header_size;
        let start = data_start + info.data_offsets.0;
        let end = data_start + info.data_offsets.1;
        Ok(&self.mmap[start..end])
    }
}

/// Multi-shard safetensors loader with mmap.
pub struct SafeTensorSet {
    shards: Vec<MmapSafeTensors>,
    index: HashMap<String, usize>,
}

impl SafeTensorSet {
    pub fn open_directory(dir: &Path) -> Result<Self, SafeTensorsError> {
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            Self::open_with_index(dir, &index_path)
        } else {
            let single = MmapSafeTensors::open(&dir.join("model.safetensors"))?;
            let mut index = HashMap::new();
            for name in single.names() {
                index.insert(name, 0);
            }
            Ok(SafeTensorSet {
                shards: vec![single],
                index,
            })
        }
    }

    fn open_with_index(dir: &Path, index_path: &Path) -> Result<Self, SafeTensorsError> {
        let content = fs::read_to_string(index_path)?;
        let parsed: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            SafeTensorsError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;

        let weight_map = parsed
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                SafeTensorsError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "missing weight_map in index",
                ))
            })?;

        let mut file_order: Vec<String> = Vec::new();
        let mut file_to_idx: HashMap<String, usize> = HashMap::new();
        for filename in weight_map.values() {
            if let Some(f) = filename.as_str() {
                if !file_to_idx.contains_key::<str>(f) {
                    file_to_idx.insert(f.to_string(), file_order.len());
                    file_order.push(f.to_string());
                }
            }
        }

        let mut shards = Vec::with_capacity(file_order.len());
        for filename in &file_order {
            shards.push(MmapSafeTensors::open(&dir.join(filename))?);
        }

        let mut index = HashMap::new();
        for (tensor_name, filename) in weight_map {
            if let Some(f) = filename.as_str() {
                index.insert(tensor_name.clone(), file_to_idx[f]);
            }
        }

        Ok(SafeTensorSet { shards, index })
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.index.keys().map(|s| s.as_str()).collect()
    }

    /// Get raw tensor data as &[u8], zero-copy from mmap.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8], SafeTensorsError> {
        let &shard_idx = self
            .index
            .get(name)
            .ok_or_else(|| SafeTensorsError::TensorNotFound(name.to_string()))?;
        self.shards[shard_idx].tensor_data(name)
    }

    /// Get tensor info (dtype, shape, offsets).
    pub fn tensor_info(&self, name: &str) -> Option<&safetensors::tensor::TensorInfo> {
        let &shard_idx = self.index.get(name)?;
        self.shards[shard_idx].tensor_info(name)
    }

    /// Get the mmap for the shard containing the named tensor (for hipHostRegister).
    pub fn shard_mmap(&self, name: &str) -> Option<&Mmap> {
        let &shard_idx = self.index.get(name)?;
        Some(self.shards[shard_idx].mmap())
    }

    /// Get all shard mmaps (for bulk hipHostRegister).
    pub fn shard_mmaps(&self) -> impl Iterator<Item = &Mmap> {
        self.shards.iter().map(|s| s.mmap())
    }
}
