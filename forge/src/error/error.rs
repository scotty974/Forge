
use core::fmt;

#[derive(Debug)]
pub enum ForgeError {
    EmptyData { context: &'static str },
    LengthMismatch { context: &'static str, left: usize, right: usize },
    ZeroVariance { column: usize },
    NotFitted,
    Diverged { epoch: usize },
    Parse { line: usize, column: usize, value: String },
    InvalidParameter { context: &'static str, name: &'static str, value: f32, expected: &'static str },
    Io(std::io::Error),
    Json(serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ForgeError>;


impl fmt::Display for ForgeError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForgeError::EmptyData { context} => write!(f, "{context} : data is empty"),
            ForgeError::NotFitted => write!(f, "Model not trained for now"),
            ForgeError::LengthMismatch { context, left, right } => write!(f, "{context} : different length ({left}|{right})"),
            ForgeError::ZeroVariance { column } => write!(f, "{column} has the same value"),
            ForgeError::Parse { line, column, value } => write!(f, "Line : {line}, Column : {column}, value : {value} is not a number"),
            ForgeError::Diverged { epoch }=> write!(f, "the training has diverged at {epoch} epoch : reduce the learning rate"),
            ForgeError::Io(e) => write!(f, "file error : {e}"),
            ForgeError::Json(e) => write!(f, "JSON error : {e}"),
            ForgeError::InvalidParameter { context, name, value, expected } => write!(f, "{context} : invalid {name} ({value}), expected {expected}"),
        }
    }
}

impl std::error::Error for ForgeError {}

impl From<std::io::Error> for ForgeError {
    fn from(e: std::io::Error) -> Self {
        ForgeError::Io(e)
    }
}

impl From<serde_json::Error> for ForgeError {
    fn from(e: serde_json::Error) -> Self {
        ForgeError::Json(e)
    }
}