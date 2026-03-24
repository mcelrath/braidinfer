use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u32);

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu:{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I8 = 3,
    I4 = 4,
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 => 1,
            DType::I4 => 1, // packed, 2 values per byte
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: DeviceId,
}

impl TensorDesc {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        let n = self.numel();
        match self.dtype {
            DType::I4 => (n + 1) / 2,
            _ => n * self.dtype.size_bytes(),
        }
    }
}
