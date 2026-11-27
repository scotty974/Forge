use crate::error::error::ForgeError;

pub fn mae(y: &[f32], y_pred: &[f32]) -> Result<f32, ForgeError> {
    if y.len() != y_pred.len() {
        return Err(ForgeError::LengthMismatch {
            context: "mae",
            left: y.len(),
            right: y_pred.len(),
        });
    }
    if y.is_empty() {
        return Err(ForgeError::EmptyData { context: "mae" });
    }

    let n = y.len() as f32;
    let mae = y
        .iter()
        .zip(y_pred)
        .map(|(ya, yi)| (ya - yi).abs())
        .sum::<f32>()
        / n;

    Ok(mae)
}
